#!/usr/bin/env python3
"""Final settings run: settle every open setting of the production model, in one resumable command.

Goal. Freeze the production model used for the parameter sweeps (v3-reference). Each setting is
chosen by a rule fixed in 3D_Code/configs/finalise.json before the run; this script only applies
the rules and writes the evidence. Simulations use the carputils production model
(3D_Code/production/slab.py); analysis uses the unchanged pipeline (run_boundary.py) plus the
exact contributing-tissue field (Shared_Pipeline/contribution_field.py).

Fixed before the run (not tested here; finalise.json):
  forward model -> openCARP pseudo-bidomain, phi_e from openCARP (your decision, DL-2026-11-02-01)
  bath          -> 5 mm of blood on every side of the slab, no-flux outer walls, a 1 mm^3 grounded
                   region at one bath corner (as Leenknegt et al. 2023; an assumption, DL-2026-11-02-02)

Stages, and the final-model parameter each one decides (each writes <out>/<stage>/stage.json; a
rerun skips completed stages and simulations):
  preflight     -> nothing; records versions so the result is reproducible
  recompute     -> output step, time step, beats, domain (existing readiness runs; no simulation)
  tune          -> conductivities: 0.90 m/s at 0.5 and 0.25 mm (strips, minutes)
  run050        -> 0.5 mm openCARP run (intact + lesion); gate: openCARP's phi_e matches the verified solve
  run025        -> the same at 0.25 mm
  decide        -> mesh spacing (0.5 vs 0.25 mm at matched CV); writes productionModel.json, report.md

Usage (from the repository root, inside the efov-runner container):
    export MPIEXEC_BIN=/usr/local/lib/opencarp/lib/petsc/bin/mpiexec
    python3 3D_Code/production/finalise.py plan --np 24
    nohup python3 3D_Code/production/finalise.py run --np 24 > 3D_Code/runs/final_v2_console.log 2>&1 &
    python3 3D_Code/production/finalise.py status
"""
import argparse
import copy
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import time
import traceback

import numpy as np

HERE = Path(__file__).resolve().parent
REPOSITORY = HERE.parents[1]
for folder in ("3D_Code", "Shared_Pipeline", "3D_Code/readiness", "3D_Code/production"):
    if str(REPOSITORY / folder) not in sys.path:
        sys.path.insert(0, str(REPOSITORY / folder))

import boundary_setup  # noqa: E402
import contribution_field as cf  # noqa: E402
import igb_io  # noqa: E402
import pipeline_run as pr  # noqa: E402

STAGES = ["preflight", "recompute", "tune", "run050", "run025", "decide"]
SOURCE = "sigmaM"                           # openCARP pseudo-bidomain (DL-2026-11-02-01)
SOURCE_TEXT = {"sigmaM": "openCARP pseudo-bidomain (source sigma_m)", "sigmaI": "bidomain physics (source sigma_i)"}
POINT_KEYS = ["condition", "noiseRepeat", "centreYmm", "gapMm", "kind", "label"]
FIT_KEYS = ["condition", "gapMm", "noiseRepeat", "kind", "label", "distanceConvention", "preprocessing"]


# ----------------------------------------------------------------- small helpers
def now():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def save(path, values):
    import common                                   # readiness helper: creates folders, stores numpy values
    common.saveJson(path, values)


def load(path):
    return json.loads(Path(path).read_text())


def log(message):
    print(f"[{now()}] {message}", flush=True)


class Context:
    def __init__(self, out, processCount):
        self.out = Path(out).resolve()
        self.np = processCount
        # FINALISE_SETTINGS lets a test (or a deliberate variant) use another settings file; preflight records it.
        self.settingsPath = Path(os.environ.get("FINALISE_SETTINGS", REPOSITORY / "3D_Code/configs/finalise.json"))
        self.settings = load(self.settingsPath)
        self.kit = load(REPOSITORY / self.settings["kitSettings"])
        self.base = load(REPOSITORY / self.settings["baseConfig"])
        self.tol = self.settings["tolerances"]

    def stage(self, name, completeOnly=True):
        path = self.out / name / "stage.json"
        if not path.exists():
            return {}
        record = load(path)
        return record if (record.get("status") == "complete" or not completeOnly) else {}

    def decisions(self):
        return self.stage("recompute").get("decisions", {})


# ----------------------------------------------------------------- simulation
def carputilsEnvironment(ctx):
    env = dict(os.environ)
    if not env.get("CARPUTILS_SETTINGS"):
        env["CARPUTILS_SETTINGS"] = str(ctx.out / "carputils_settings.yaml")
    return env


def ensureCarputilsSettings(ctx):
    """Use the user's carputils settings if set; otherwise write a run-local file (MPIEXEC = MPIEXEC_BIN)."""
    if os.environ.get("CARPUTILS_SETTINGS"):
        return os.environ["CARPUTILS_SETTINGS"]
    path = ctx.out / "carputils_settings.yaml"
    if not path.exists():
        carp = shutil.which("openCARP") or os.environ.get("OPENCARP_BIN")
        if not carp:
            raise RuntimeError("openCARP not found on PATH")
        mpiexec = os.environ.get("MPIEXEC_BIN", "mpiexec")
        code = ("from carputils.settings.config import new; "
                f"new({str(path)!r}, software_root={str(Path(carp).resolve().parent)!r}, mpiexec={mpiexec!r})")
        subprocess.run([sys.executable, "-c", code], check=True)
    return str(path)


def expectedFrames(spec, fromMs=0.0):
    pacing = spec["pacing"]
    duration = spec.get("durationMs", pacing["cycleLengthMs"] * pacing["beatCount"])
    return int(round((duration - fromMs) / spec["sampleStepMs"])) + 1


def jobComplete(job, spec):
    phieStart = spec.get("restartMs") or 0.0
    for name, start in [("vm.igb", 0.0), ("phie.igb", phieStart)]:
        if not (job / name).exists() or igb_io.header(job / name)[2] != expectedFrames(spec, start):
            return False
    if spec.get("restartMs") and not (job / "restart.json").exists():
        return False
    return (job / "act-thresh.dat").exists()


def restartTime(ctx, beatCount):
    """Prepacing checkpoint: early enough that phi_e covers every frame the analysis reads (the last beat's
    earliest possible window start minus twice the filter history), less a margin; on the 10 ms grid."""
    pace, rule = ctx.base["pacing"], ctx.settings["windowRule"]
    earliestWindow = (beatCount - 1) * pace["cycleLengthMs"] + pace["durationMs"] + rule["windowMinimumAfterStimulusMs"]
    restart = earliestWindow - 2 * ctx.base["analysis"]["filterHistoryMs"] - ctx.settings["prepacing"]["restartMarginMs"]
    restart = np.floor(restart / 10.0) * 10.0
    return float(restart) if restart > 0 else None


def runSlab(ctx, job, spec):
    """carputils slab.py for one condition; reuses a complete run, refuses an incomplete one."""
    job = Path(job)
    specPath = job.parent / f"{job.name}.json"
    if job.exists() and jobComplete(job, spec):
        if load(specPath) != spec:
            raise ValueError(f"{job} was run with a different spec; move it aside")
        return load(job.parent / f"{job.name}.timing.json")
    if job.exists():
        raise FileExistsError(f"{job} is incomplete (interrupted run?). Move it aside, then rerun.")
    job.parent.mkdir(parents=True, exist_ok=True)
    save(specPath, spec)
    log(f"openCARP: {job.name}")
    started = time.time()
    with open(job.parent / f"{job.name}.log", "w") as stream:
        code = subprocess.call([sys.executable, str(HERE / "slab.py"), "--spec", str(specPath), "--np", str(ctx.np),
                                "--ID", str(job), "--overwrite-behaviour", "error", "--silent",
                                "--flavor", solverFlavor(ctx)],
                               stdout=stream, stderr=subprocess.STDOUT, env=carputilsEnvironment(ctx))
    if code != 0 or not jobComplete(job, spec):
        raise RuntimeError(f"{job.name}: slab.py failed (exit {code}); see {job.parent / (job.name + '.log')}")
    launches = (job.parent / f"{job.name}.log").read_text(errors="replace").count("*** Launching simulation ***")
    expected = 2 if spec.get("restartMs") else 1                     # prepace + restart, or one run
    if launches != expected:
        raise RuntimeError(f"{job.name}: {launches} launch messages, expected {expected} (MPI launcher mismatch?)")
    timing = dict(wallSeconds=round(time.time() - started, 1), processCount=ctx.np)
    save(job.parent / f"{job.name}.timing.json", timing)
    return timing


def solverFlavor(ctx):
    """carputils solver flavour chosen by preflight (boomeramg if this openCARP's PETSc has hypre)."""
    return ctx.stage("preflight").get("solverFlavor", ctx.settings["solver"]["fallbackFlavor"])


def flavorSmokeTest(ctx, flavor):
    """5 ms pseudo-bidomain run of a tiny slab with this flavour: does the build support it?"""
    folder = ctx.out / "preflight" / f"smoke_{flavor}"
    if folder.exists():
        shutil.rmtree(folder)
    folder.mkdir(parents=True)
    state = folder / "conditioning" / "Courtemanche.sv"
    boundary_setup.conditionCells(folder, dict(timeStepUs=20.0, conditioning=dict(enabled=True, beatCount=1),
                                               pacing=dict(cycleLengthMs=500.0, durationMs=2.0, strength=50.0)))
    spec = dict(condition="intact", tissueSizeMm=[4.0, 4.0, 1.0], spacingMm=0.5, bathThicknessMm=1.0, bidomain=2,
                conductivitiesSPerM=ctx.kit["cvTuning"]["baseConductivitiesSPerM"], lesionConductivitySPerM=1e-6,
                bloodSPerM=0.74, boundaryYmm=0.0, timeStepUs=20.0, sampleStepMs=0.2,
                pacing=dict(cycleLengthMs=500.0, beatCount=1, durationMs=2.0, strength=50.0, stripWidthMm=0.5),
                initialState=str(state), meshRoot=str(folder / "meshes"), durationMs=5.0)
    save(folder / "job.json", spec)
    with open(folder / "job.log", "w") as stream:
        code = subprocess.call([sys.executable, str(HERE / "slab.py"), "--spec", str(folder / "job.json"), "--np",
                                str(min(ctx.np, 2)), "--ID", str(folder / "job"), "--silent", "--flavor", flavor],
                               stdout=stream, stderr=subprocess.STDOUT, env=carputilsEnvironment(ctx))
    return code == 0 and (folder / "job" / "phie.igb").exists()


def conditioning(ctx, folder, timeStepUs, strength):
    """bench limit cycle (openCARP tutorial workflow), shared by every slab run with this time step."""
    pace = ctx.base["pacing"]
    config = dict(timeStepUs=timeStepUs, conditioning=dict(enabled=True, beatCount=ctx.kit["cvTuning"]["conditioningBeats"]),
                  pacing=dict(cycleLengthMs=pace["cycleLengthMs"], durationMs=pace["durationMs"], strength=strength))
    folder.mkdir(parents=True, exist_ok=True)
    boundary_setup.conditionCells(folder, config)
    return folder / "conditioning" / "Courtemanche.sv"


def slabSpec(ctx, condition, spacing, tuned, state, extra=None):
    d, t, bath = ctx.decisions(), ctx.settings["tissue"], ctx.settings["bath"]
    domain = ctx.settings["domains"][d["domain"]]
    pace = ctx.base["pacing"]
    spec = dict(condition=condition, tissueSizeMm=[domain[0], domain[1], t["wallMm"]], spacingMm=spacing,
                bathThicknessMm=bath["thicknessMm"], bathAllSides=bath["allSides"], groundCubeMm=bath["groundCubeMm"],
                bidomain=2, conductivitiesSPerM=tuned["conductivitiesSPerM"],
                lesionConductivitySPerM=t["lesionConductivitySPerM"], bloodSPerM=t["bloodSPerM"],
                boundaryYmm=t["boundaryYmm"], timeStepUs=d["timeStepUs"], sampleStepMs=d["sampleStepMs"],
                pacing=dict(cycleLengthMs=pace["cycleLengthMs"], beatCount=d["beatCount"], durationMs=pace["durationMs"],
                            strength=tuned["stimulusStrength"], stripWidthMm=pace["stripWidthMm"]),
                initialState=str(state), meshRoot=str(ctx.out / "meshes"))
    if tuned.get("cellSurfVolRatioPerUm") is not None:
        spec["cellSurfVolRatioPerUm"] = tuned["cellSurfVolRatioPerUm"]
    spec.update(extra or {})
    spec["phieOutputHeightMm"] = ctx.settings["phieOutputHeightMm"]
    if "durationMs" not in spec and ctx.settings["prepacing"]["enabled"]:
        restart = restartTime(ctx, d["beatCount"])
        if restart is not None:
            spec["restartMs"] = restart
    return spec


def analysisConfig(ctx, spacing, tuned):
    d = ctx.decisions()
    domain = ctx.settings["domains"][d["domain"]]
    config = copy.deepcopy(ctx.base)
    config["slabSizeMm"] = [domain[0], domain[1], ctx.settings["tissue"]["wallMm"]]
    config["meshSpacingMm"] = spacing
    config["timeStepUs"] = float(d["timeStepUs"])
    config["sampleStepMs"] = float(d["sampleStepMs"])
    config["pacing"]["beatCount"] = int(d["beatCount"])
    config["pacing"]["strength"] = float(tuned["stimulusStrength"])
    config["acceptance"]["responseWindowMs"] = list(ctx.settings["physiologyResponseWindowMs"])
    config["analysis"]["beatWindowMs"] = [25.0, 100.0]          # placeholder; replaced by the derived window
    config["analysis"]["groundTruth"] = "external"
    config["tissue"] = {"conductivitiesSPerM": dict(tuned["conductivitiesSPerM"]),
                        "cellSurfVolRatioPerUm": tuned.get("cellSurfVolRatioPerUm"),
                        "source": f"finalise tune stage, {spacing} mm; CV calibration: {tuned.get('calibration')}"}
    config["_purpose"] = "Final settings run; production-model candidate."
    boundary_setup.validateConfig(config)
    return config


# ----------------------------------------------------------------- one analysed run
def variantName(spacing):
    return f"dx{spacing:.2f}"


def analyseVariant(ctx, folder, spacing, tuned, jobs):
    """Pipeline run on openCARP's phi_e, forward-model gate and contribution field for one intact + lesion pair."""
    folder = Path(folder)
    summaryPath = folder / "variant.json"
    if summaryPath.exists():
        return load(summaryPath)
    config = pr.prepare(folder, analysisConfig(ctx, spacing, tuned), jobs, ctx.settings["windowRule"])
    t = ctx.settings["tissue"]
    models, check = {}, {}
    for condition, job in jobs.items():
        meshBase = pr.jobRecord(job)["meshBase"]
        # Same mesh, conductivities and ground as the openCARP run (ground.vtx written by slab.py).
        model = cf.BathModel(meshBase, tuned["conductivitiesSPerM"], t["lesionConductivitySPerM"], t["bloodSPerM"],
                             SOURCE, groundNodes=pr.readVertices(Path(job) / "ground.vtx"))
        models[condition] = model
        interpolator, layout = pr.recoveryInterpolator(meshBase, folder)
        check[condition] = forwardModelCheck(ctx, job, model, interpolator, config, folder)
        log(f"electrograms: {folder.name} {condition}")
        pr.writeElectrograms(folder, config, condition, pr.openCarpSignals(job, interpolator, layout),
                             "openCARP pseudo-bidomain phie.igb")
    log(f"pipeline: {folder.name}")
    status = pr.runPipeline(folder, 1)
    log(f"contribution field: {folder.name}")
    table = pr.groundTruth(folder, config, models, ctx.settings["contributionField"]["voxelMm"],
                           ctx.settings["contributionField"]["curveStepMm"])
    summary = dict(folder=str(folder), spacingMm=spacing, bath=ctx.settings["bath"], source=SOURCE,
                   electrograms="openCARP", pilotStatus=status,
                   window=config["analysis"]["beatWindowMs"], forwardModelCheck=check,
                   groundTruthCheck=dict(shareSumMaxDeviation=float((table.shareSum - 1).abs().max()),
                                         reconstructionResidualMax=float(table.reconstructionResidual.max()),
                                         scaleOverGainRange=[float(table.scaleOverGain.min()), float(table.scaleOverGain.max())]))
    save(summaryPath, summary)
    return summary


def forwardModelCheck(ctx, job, model, interpolator, config, folder, frameCount=25):
    """DL-2026-10-28-11: openCARP phi_e = sigma_m extracellular solve at the electrode samples."""
    times, frames = pr.analysisFrames(folder, config)
    chosen = np.unique(np.linspace(frames.start, frames.stop - 1, frameCount).astype(int))
    carp = np.stack([interpolator @ pr.readPhie(job, slice(f, f + 1)).T[:, 0] for f in chosen], axis=1)
    fem = np.stack([interpolator @ model.solver.solve(model.fullVm(igb_io.read(Path(job) / "vm.igb", slice(f, f + 1)).T))[:, 0]
                    for f in chosen], axis=1)
    rms = lambda x: np.sqrt(np.mean(x ** 2, axis=1))
    relative = rms(carp - fem) / np.maximum(rms(fem), 1e-30)
    centred = lambda x: x - x.mean(axis=1, keepdims=True)
    a, b = centred(carp), centred(fem)
    correlation = np.sum(a * b, axis=1) / np.maximum(np.sqrt(np.sum(a * a, axis=1) * np.sum(b * b, axis=1)), 1e-30)
    result = dict(frames=len(chosen), points=int(len(carp)), relativeRmsMax=float(relative.max()),
                  relativeRmsMedian=float(np.median(relative)), correlationMin=float(correlation.min()))
    result["passed"] = bool(result["relativeRmsMax"] <= ctx.tol["forwardModelMatchRelativeRms"]
                            and result["correlationMin"] >= ctx.tol["forwardModelMatchCorrelation"])
    return result


# ----------------------------------------------------------------- comparison
def loadRun(folder):
    import pandas as pd
    folder = Path(folder)
    results = folder / "results"
    gain = load(results / "calibration.json")["gain"]
    points = pd.concat([pd.read_csv(results / f"{c}Points.csv") for c in ["intact", "lesion"]], ignore_index=True)
    points["rawCleanPp"] = points.cleanAtSelectedWeightsPpMv / gain
    fits = pd.read_csv(results / "clinical" / "changePointFits.csv")
    if "selected" in fits:
        fits = fits[fits.selected.astype(str).str.lower() == "true"]
    # Profiles too short to fit carry only their fitStatus (e.g. insufficientDetectedDistances).
    for column in ["changePointMm", "model"]:
        if column not in fits:
            fits[column] = np.nan
    physiology = pd.read_csv(folder / "physiologyChecks.csv")
    last = physiology[physiology.condition == "intact"].beatNumber.max()
    speed = float(physiology[(physiology.condition == "intact") & (physiology.beatNumber == last)].speedMmPerMs.iloc[0])
    field = dict(np.load(results / "contributionField.npz"))
    return dict(gain=gain, points=points, fits=fits, speed=speed, field=field,
                status=load(results / "pilotStatus.json"), physiology=load(folder / "physiologyStatus.json")["passed"])


def compareRuns(ctx, folderA, folderB, label, ppMode="raw", useCv=True, useChangePoint=True):
    """A vs B (B = finer / thicker / reference). DL-05 tolerances plus the contribution field."""
    a, b = loadRun(folderA), loadRun(folderB)
    tol = ctx.tol
    points = a["points"].merge(b["points"], on=POINT_KEYS, suffixes=("A", "B"))
    clean = points[points.noiseRepeat == 0]
    relative = {mode: ((clean[c + "A"] - clean[c + "B"]).abs() / clean[c + "B"])
                for mode, c in [("raw", "rawCleanPp"), ("calibrated", "ppMv")]}
    ppRelative = relative[ppMode]
    fits = a["fits"].merge(b["fits"], on=FIT_KEYS, suffixes=("A", "B"))
    primary = fits[(fits.preprocessing == "clinicalTrimmed") & (fits.distanceConvention == "recordingAnchor")]
    supportedA = primary.fitStatusA == "supportedDescriptiveChangePoint"
    supportedB = primary.fitStatusB == "supportedDescriptiveChangePoint"
    either = primary[supportedA | supportedB]
    cpShift = float((either.changePointMmA - either.changePointMmB).abs().max()) if len(either) else 0.0
    ids = {i: k for k, i in enumerate(b["field"]["recordingId"])}
    distances = [cf.fieldDistance(a["field"]["fields"][k], b["field"]["fields"][ids[i]])
                 for k, i in enumerate(a["field"]["recordingId"]) if i in ids]
    row = dict(label=label, a=str(folderA), b=str(folderB), ppMode=ppMode,
               cvA=a["speed"], cvB=b["speed"], cvRelative=abs(a["speed"] - b["speed"]) / b["speed"],
               ppRelativeMax=float(ppRelative.max()), ppRelativeMedian=float(ppRelative.median()), matchedSignals=int(len(clean)),
               ppRawRelativeMax=float(relative["raw"].max()), ppCalibratedRelativeMax=float(relative["calibrated"].max()),
               gainA=a["gain"], gainB=b["gain"],
               changePointMaxShiftMmSupported=cpShift, supportStatusChanges=int((supportedA != supportedB).sum()),
               primaryFits=int(len(primary)), supportedFits=int(len(either)),
               fieldL1Max=float(max(distances)) if distances else None,
               fieldL1Median=float(np.median(distances)) if distances else None, fieldRecordings=len(distances),
               physiologyA=a["physiology"], physiologyB=b["physiology"],
               acceptanceA=a["status"]["pilotAcceptancePassed"], acceptanceB=b["status"]["pilotAcceptancePassed"])
    checks = {"pp": row["ppRelativeMax"] <= tol["ppRelative"],
              "field": distances != [] and row["fieldL1Max"] <= tol["contributionFieldL1"]}
    if useCv:
        checks["cv"] = row["cvRelative"] <= tol["cvRelative"]
    if useChangePoint:
        checks["changePoint"] = cpShift <= tol["changePointMm"] and row["supportStatusChanges"] == 0
    # The signals behind the PP maximum (the gate is the per-signal maximum, so small signals can dominate it).
    worst = clean.assign(ppRelative=ppRelative).nlargest(5, "ppRelative")
    row["worstPpSignals"] = [dict(condition=w["condition"], kind=w["kind"], label=w["label"], centreYmm=float(w["centreYmm"]),
                                  gapMm=float(w["gapMm"]), ppMvA=float(w["ppMvA"]), ppMvB=float(w["ppMvB"]),
                                  rawPpA=float(w["rawCleanPpA"]), rawPpB=float(w["rawCleanPpB"]),
                                  relative=float(w["ppRelative"])) for w in worst.to_dict("records")]
    row["checks"] = checks
    row["passed"] = bool(all(checks.values()) and a["physiology"] and b["physiology"])
    return row


# ----------------------------------------------------------------- stages
def stagePreflight(ctx):
    record = dict(timeUtc=now(), settingsFile=str(ctx.settingsPath), settings=ctx.settings, python=sys.version, platform=platform.platform(), cpuCount=os.cpu_count(),
                  processCount=ctx.np, mpiexecBin=os.environ.get("MPIEXEC_BIN"))
    for name in ["openCARP", "mesher", "bench"]:
        path = shutil.which(name)
        if not path:
            raise RuntimeError(f"{name} not on PATH")
        record[name] = path
    record["openCarpBuild"] = subprocess.run(["openCARP", "-buildinfo"], capture_output=True, text=True).stdout[-1500:]
    if ctx.np > 1 and not os.environ.get("MPIEXEC_BIN"):
        raise RuntimeError("Set MPIEXEC_BIN to openCARP's own launcher (/usr/local/lib/opencarp/lib/petsc/bin/mpiexec)")
    import scipy, pandas
    record["packages"] = dict(numpy=np.__version__, scipy=scipy.__version__, pandas=pandas.__version__)
    try:
        import pyamg
        record["packages"]["pyamg"] = pyamg.__version__
    except ImportError:
        record["packages"]["pyamg"] = "not installed (large extracellular solves fall back to ILU-GMRES; pip install pyamg)"
    record["carputilsSettings"] = ensureCarputilsSettings(ctx)
    probe = subprocess.run([sys.executable, "-c", "import carputils, os; print(os.path.dirname(carputils.__file__))"],
                           capture_output=True, text=True, env=carputilsEnvironment(ctx))
    if probe.returncode:
        raise RuntimeError("carputils is not importable: " + probe.stderr[-500:])
    record["carputils"] = probe.stdout.strip()
    ladderFolder = (REPOSITORY / ctx.settings["readinessLadderFolder"]).resolve()
    if not (ladderFolder / "ladderSummary.json").exists():
        raise FileNotFoundError(f"{ladderFolder}: readiness ladder results are needed by the recompute stage")
    record["readinessLadder"] = str(ladderFolder)
    record["repositoryCommit"] = subprocess.run(["git", "-C", str(REPOSITORY), "rev-parse", "HEAD"],
                                                capture_output=True, text=True).stdout.strip()
    record["diskFreeGb"] = round(shutil.disk_usage(ctx.out).free / 1e9, 1)
    preferred, fallback = ctx.settings["solver"]["preferredFlavor"], ctx.settings["solver"]["fallbackFlavor"]
    record["solverFlavor"] = preferred if flavorSmokeTest(ctx, preferred) else fallback
    if record["solverFlavor"] != preferred:
        record["solverNote"] = f"{preferred} failed on this build (see preflight/smoke_{preferred}/job.log); using {fallback}"
    return record


def stageRecompute(ctx):
    import ladder
    folder = REPOSITORY / ctx.settings["readinessLadderFolder"]
    tol = ctx.tol
    rows = {}
    for item in ctx.kit["ladder"]["comparisons"]:
        coarse, fine = folder / item["coarse"], folder / item["fine"]
        if not ((coarse / "results" / "contributions.csv").exists() and (fine / "results" / "contributions.csv").exists()):
            rows[item["label"]] = dict(label=item["label"], skipped="one or both runs not analysed"); continue
        rows[item["label"]] = ladder.compare(coarse, fine, item["label"])
    def passes(label, changePoint):
        row = rows.get(label, {})
        if "cvRelative" not in row:
            raise RuntimeError(f"Comparison {label} is not available: {row}")
        ok = row["cvRelative"] <= tol["cvRelative"] and row["ppRelativeMaxAll"] <= tol["ppRelative"]
        if changePoint:
            ok = ok and (row["changePointMaxAbsMmSupported"] or 0) <= tol["changePointMm"] and row["supportStatusChanges"] == 0
        return bool(ok)
    verdicts = {"output_0.20_vs_0.10": passes("output_0.20_vs_0.10", False),
                "timestep_20_vs_10": passes("timestep_20_vs_10", False),
                "beats_4_vs_8": passes("beats_4_vs_8", False),
                "domain_40x60_vs_60x90": passes("domain_40x60_vs_60x90", True)}
    decisions = dict(sampleStepMs=0.2 if verdicts["output_0.20_vs_0.10"] else 0.1,
                     timeStepUs=20.0 if verdicts["timestep_20_vs_10"] else 10.0,
                     beatCount=4 if verdicts["beats_4_vs_8"] else 8,
                     domain="pilot" if verdicts["domain_40x60_vs_60x90"] else "large")
    return dict(comparisons=rows, verdicts=verdicts, decisions=decisions,
                note="Fixed-conductivity, unbounded-medium ladder runs. FOV80 reported in comparisons, not used.")


def stageTune(ctx):
    import strip_cv
    settings = copy.deepcopy(ctx.kit["cvTuning"])
    settings["timeStepUs"] = ctx.decisions()["timeStepUs"]
    target = ctx.settings["tissue"]["cvTargetMPerS"]
    out = {}
    for spacing in ctx.settings["spacingsMm"]:
        path = ctx.out / "tune" / f"tuned_dx{spacing:.2f}.json"
        if not path.exists():
            scale, strength, history, converged = strip_cv.tune(ctx.out / "tune", f"dx{spacing:g}", target, spacing,
                                                               settings, float(settings["stimulus"]["strength"]), ctx.np,
                                                               allowCaptureSearch=True)
            if not converged:
                raise RuntimeError(f"CV tuning at {spacing} mm did not converge: {history}")
            save(path, dict(conductivitiesSPerM=strip_cv.scaledConductivities(settings["baseConductivitiesSPerM"], scale),
                            stimulusStrength=strength, scaleOfRobertsScher=scale, targetMPerS=target,
                            achievedMPerS=history[-1]["speedMPerS"], spacingMm=spacing, timeStepUs=settings["timeStepUs"],
                            history=history, interpretation="Calibration (DL-04): Roberts and Scher ratios, magnitudes tuned at this spacing."))
        out[f"{spacing:.2f}"] = load(path)
    return dict(tuned=out)


def tuned(ctx, spacing):
    return load(ctx.out / "tune" / f"tuned_dx{spacing:.2f}.json")


OPENCARP_DEFAULT_BETA_PER_UM = 0.14         # imp_region[].cellSurfVolRatio default (openCARP +Help), used by the tune strips


def tissueFor(ctx, spacing):
    """Conductivities, beta and stimulus of the production model at one spacing (DL-2026-11-02-03).

    The tune stage reaches the CV target by scaling the Roberts and Scher conductivities by s. Monodomain Vm
    depends on sigma / beta only (checked: Vm identical within 4e-5 mV, VL-2026-11-02-03), so the same
    propagation is obtained with measured conductivities and beta = 0.14 / s ("surfaceToVolume"), or with
    conductivities x s and the default beta ("conductivity"). Only the extracellular field differs: beta does
    not enter it, the conductivity magnitudes (against blood) do.
    """
    t = tuned(ctx, spacing)
    mode = ctx.settings["tissue"]["cvCalibration"]
    if mode == "surfaceToVolume":
        base = ctx.kit["cvTuning"]["baseConductivitiesSPerM"]
        return dict(conductivitiesSPerM=dict(base), stimulusStrength=t["stimulusStrength"],
                    cellSurfVolRatioPerUm=OPENCARP_DEFAULT_BETA_PER_UM / t["scaleOfRobertsScher"], calibration=mode,
                    scaleOfRobertsScher=t["scaleOfRobertsScher"], achievedMPerS=t["achievedMPerS"])
    if mode == "conductivity":
        return dict(conductivitiesSPerM=t["conductivitiesSPerM"], stimulusStrength=t["stimulusStrength"],
                    cellSurfVolRatioPerUm=None, calibration=mode,
                    scaleOfRobertsScher=t["scaleOfRobertsScher"], achievedMPerS=t["achievedMPerS"])
    raise ValueError("tissue.cvCalibration must be 'surfaceToVolume' or 'conductivity'")


def runPair(ctx, spacing):
    t = tissueFor(ctx, spacing)
    state = conditioning(ctx, ctx.out / "conditioning" / f"dt{ctx.decisions()['timeStepUs']:g}_s{t['stimulusStrength']:g}",
                         ctx.decisions()["timeStepUs"], t["stimulusStrength"])
    jobs, specs, timing = {}, {}, {}
    for condition in ("intact", "lesion"):
        spec = slabSpec(ctx, condition, spacing, t, state)
        job = ctx.out / "sims" / f"{variantName(spacing)}_{condition}"
        timing[condition] = runSlab(ctx, job, spec)
        jobs[condition], specs[condition] = job, spec
    return jobs, specs, timing


def stageRun(ctx, spacing):
    """openCARP pseudo-bidomain run at one spacing, analysed; gate: openCARP phi_e = the verified sigma_m solve."""
    jobs, specs, timing = runPair(ctx, spacing)
    variant = analyseVariant(ctx, ctx.out / "variants" / variantName(spacing), spacing, tissueFor(ctx, spacing), jobs)
    if not all(c["passed"] for c in variant["forwardModelCheck"].values()):
        raise RuntimeError(f"openCARP phi_e does not match the sigma_m solve at {spacing} mm: {variant['forwardModelCheck']}")
    return dict(timing=timing, nodes={c: pr.jobRecord(j)["sets"]["nodes"] for c, j in jobs.items()}, variant=variant)


def stageRun050(ctx):
    return stageRun(ctx, 0.5)


def stageRun025(ctx):
    return stageRun(ctx, 0.25)


def stageDecide(ctx):
    """Decides the mesh spacing (matched CV) and writes the final settings."""
    d = ctx.decisions()
    # Matched CV: the meshes have different tuned conductivity magnitudes, so raw amplitudes differ by design;
    # the gate uses calibrated PP (spatial pattern) and reports raw PP.
    mesh = compareRuns(ctx, ctx.out / "variants" / variantName(0.5), ctx.out / "variants" / variantName(0.25),
                       "mesh_0.50_vs_0.25", ppMode="calibrated")
    spacing = 0.5 if mesh["passed"] else 0.25
    flags = []
    if spacing == 0.25:
        flags.append("0.5 vs 0.25 mm failed: 0.25 mm is used, but 0.25 vs 0.125 mm is not yet shown.")
    flags.append("Forward model: openCARP pseudo-bidomain is your decision and overrides the 1 Nov rule outcome "
                 "(DL-2026-11-02-01).")
    flags.append("Bath: modelling assumption, not tested (DL-2026-11-02-02).")
    if "PROPOSED" in ctx.settings["tolerances"].get("_contributionFieldL1", ""):
        flags.append("contributionFieldL1 tolerance (0.10) is still PROPOSED: confirm it, or the mesh decision rests on it.")
    flags.append("Scar type (R10): your decision; the pilot's insulating scar is used here.")
    bath = ctx.settings["bath"]
    t = tissueFor(ctx, spacing)
    model = dict(
        status="PROPOSED by the final settings run; becomes v3-reference once you confirm the flags",
        createdUtc=now(), domainMm=ctx.settings["domains"][d["domain"]] + [ctx.settings["tissue"]["wallMm"]],
        spacingMm=spacing, timeStepUs=d["timeStepUs"], sampleStepMs=d["sampleStepMs"], beatCount=d["beatCount"],
        bath=dict(thicknessMm=bath["thicknessMm"], allSides=bath["allSides"], groundCubeMm=bath["groundCubeMm"],
                  outerBoundary="no flux", bloodSPerM=ctx.settings["tissue"]["bloodSPerM"]),
        forwardModel=SOURCE_TEXT[SOURCE], conductivitiesSPerM=t["conductivitiesSPerM"], stimulusStrength=t["stimulusStrength"],
        cellSurfVolRatioPerUm=t["cellSurfVolRatioPerUm"] or OPENCARP_DEFAULT_BETA_PER_UM, cvCalibration=t["calibration"],
        cvTargetMPerS=ctx.settings["tissue"]["cvTargetMPerS"], scar="insulating (pilot) unless you choose otherwise (R10)",
        prepacingRestartMs=restartTime(ctx, d["beatCount"]), phieOutputHeightMm=ctx.settings["phieOutputHeightMm"],
        measurementWindow="derived per run (DL-2026-10-28-08)", groundTruth="exact contribution field (contribution_field.py)",
        productionSpec=slabSpec(ctx, "<intact|lesion>", spacing, t, "<conditioning state>"),
        flags=flags)
    return dict(meshComparison=mesh, productionModel=model)


STAGE_FUNCTIONS = dict(preflight=stagePreflight, recompute=stageRecompute, tune=stageTune, run050=stageRun050,
                       run025=stageRun025, decide=stageDecide)


# ----------------------------------------------------------------- report
def fmt(value, digits=3):
    if value is None:
        return "-"
    if isinstance(value, bool):
        return "pass" if value else "FAIL"
    if isinstance(value, float):
        return f"{value:.{digits}g}"
    return str(value)


def writeReport(ctx):
    lines = ["# Final settings run: report", "", f"Generated {now()}. Rules: 3D_Code/configs/finalise.json.", ""]
    r = ctx.stage("recompute")
    if r:
        lines += ["## Recompute (existing readiness runs, no simulation)", "", "| Comparison | CV | PP max | change point (supported) | FOV80 (reported) | verdict |",
                  "| --- | --- | --- | --- | --- | --- |"]
        for label, row in r.get("comparisons", {}).items():
            if "cvRelative" in row:
                lines.append(f"| {label} | {fmt(row['cvRelative'])} | {fmt(row['ppRelativeMaxAll'])} | "
                             f"{fmt(row.get('changePointMaxAbsMmSupported'))} mm | {fmt(row.get('fov80MaxAbsMm'))} mm | "
                             f"{fmt(r.get('verdicts', {}).get(label, '-'))} |")
            else:
                lines.append(f"| {label} | {row.get('skipped', row.get('error'))} | | | | |")
        lines += ["", "Decisions: " + ", ".join(f"{k} = {v}" for k, v in r["decisions"].items()), ""]
    tu = ctx.stage("tune")
    if tu:
        lines += ["## CV tuning", "", f"CV calibration: {ctx.settings['tissue']['cvCalibration']} (DL-2026-11-02-03).", "",
                  "| Spacing (mm) | CV (m/s) | scale s of Roberts-Scher | production sigma_il | production sigma_el | production beta (1/um) |",
                  "| --- | --- | --- | --- | --- | --- |"]
        for key, v in tu["tuned"].items():
            p = tissueFor(ctx, float(key))
            g = p["conductivitiesSPerM"]
            lines.append(f"| {key} | {fmt(v['achievedMPerS'])} | {fmt(v['scaleOfRobertsScher'])} | {fmt(g['g_il'])} | {fmt(g['g_el'])} | "
                         f"{fmt(p['cellSurfVolRatioPerUm'] or OPENCARP_DEFAULT_BETA_PER_UM, 4)} |")
        lines.append("")
    for stage in ["run050", "run025"]:
        s = ctx.stage(stage)
        if s:
            lines += [f"## {stage} (openCARP wall time)", "", ", ".join(f"{c} {v['wallSeconds']:.0f} s" for c, v in s["timing"].items()), ""]
    def table(title, decides, rows):
        out = [f"## {title}", "", f"Decides: {decides}", "",
               "| Comparison | CV | PP max (gated) | PP raw / calibrated | change point shift (supported) | support changes | field L1 max | verdict |",
               "| --- | --- | --- | --- | --- | --- | --- | --- |"]
        for row in rows:
            out.append(f"| {row['label']} | {fmt(row['cvRelative'])} | {fmt(row['ppRelativeMax'])} ({row['ppMode']}) | "
                       f"{fmt(row['ppRawRelativeMax'])} / {fmt(row['ppCalibratedRelativeMax'])} | "
                       f"{fmt(row['changePointMaxShiftMmSupported'])} mm ({row['supportedFits']} fits) | {row['supportStatusChanges']} | "
                       f"{fmt(row['fieldL1Max'])} | {fmt(row['passed'])} |")
        return out + [""]
    bath = ctx.settings["bath"]
    lines += ["## Fixed before the run", "",
              f"- Forward model: {SOURCE_TEXT[SOURCE]}; electrograms are openCARP's phi_e. "
              f"{ctx.settings['forwardModelDecision']['reason']}",
              f"- Bath: {bath['thicknessMm']:g} mm on every side, no-flux outer walls, {bath['groundCubeMm']:g} mm "
              f"grounded cube at a bath corner. {bath['_source']}", ""]
    for stage in ["run050", "run025"]:
        s = ctx.stage(stage)
        if s:
            lines += [f"Gate {stage} (openCARP phi_e vs sigma_m solve): " + ", ".join(
                f"{c} relative RMS {v['relativeRmsMax']:.2g}, correlation {v['correlationMin']:.8f}"
                for c, v in s["variant"]["forwardModelCheck"].items()), ""]
    de = ctx.stage("decide")
    if de:
        lines += table("Mesh spacing (matched CV)", "0.5 mm if it passes, else 0.25 mm", [de["meshComparison"]])
        lines += ["## Proposed production model", "", "```json", json.dumps(de["productionModel"], indent=2), "```", ""]
    (ctx.out / "report.md").write_text("\n".join(lines) + "\n")


# ----------------------------------------------------------------- commands
def run(ctx, only=None):
    ctx.out.mkdir(parents=True, exist_ok=True)
    for name in ([only] if only else STAGES):
        previous = ctx.stage(name)
        if previous:
            log(f"[{name}] complete - skipping")
            continue
        log(f"[{name}] started")
        started = now()
        try:
            result = STAGE_FUNCTIONS[name](ctx)
            save(ctx.out / name / "stage.json", dict(status="complete", started=started, finished=now(), **result))
            log(f"[{name}] complete")
        except Exception as error:                                   # keep the error with the stage, then stop
            save(ctx.out / name / "stage.json", dict(status="failed", started=started, finished=now(),
                                                    error=f"{type(error).__name__}: {error}", traceback=traceback.format_exc()))
            log(f"[{name}] FAILED: {error}")
            writeReport(ctx)
            return 1
        writeReport(ctx)
        if name == "decide":
            save(ctx.out / "productionModel.json", ctx.stage("decide")["productionModel"])
    return 0


def plan(ctx):
    d = ctx.decisions() or dict(domain="pilot (until recompute)", beatCount=4, sampleStepMs=0.2, timeStepUs=20.0)
    domains = ctx.settings["domains"]
    domain = domains.get(d["domain"], domains["pilot"])
    wall = ctx.settings["tissue"]["wallMm"]
    bath = ctx.settings["bath"]
    B = bath["thicknessMm"]
    def nodes(spacing):
        sides = 2 if bath["allSides"] else 0
        return int(np.prod([round((domain[0] + sides * B) / spacing) + 1, round((domain[1] + sides * B) / spacing) + 1,
                            round((wall + (2 if bath["allSides"] else 1) * B) / spacing) + 1]))
    print(f"Domain {domain[0]:g} x {domain[1]:g} x {wall:g} mm ({d['domain']}); beats {d['beatCount']}; output {d['sampleStepMs']} ms")
    print(f"Bath {B:g} mm" + (" on every side" if bath["allSides"] else " on top") + f", grounded {bath['groundCubeMm']:g} mm corner cube")
    print("openCARP runs (pseudo-bidomain), each intact + lesion:")
    print(f"  run050  0.5 mm: {nodes(0.5):,} nodes")
    print(f"  run025  0.25 mm: {nodes(0.25):,} nodes")
    for name in STAGES:
        print(f"  {name:14s} {ctx.stage(name, completeOnly=False).get('status', 'not started')}")


def status(ctx):
    for name in STAGES:
        s = ctx.stage(name, completeOnly=False)
        print(f"{name:14s} {s.get('status', 'not started'):12s} {s.get('finished', '')} {s.get('error', '')[:200]}")
    logs = sorted(ctx.out.glob("sims/*.log"), key=lambda p: p.stat().st_mtime)[-1:] + \
        sorted(ctx.out.glob("variants/*/analysis.log"), key=lambda p: p.stat().st_mtime)[-1:]
    for path in logs:
        tail = path.read_text(errors="replace").strip().splitlines()[-1:] if path.stat().st_size else []
        print(f"latest: {path.relative_to(ctx.out)}: {tail[0] if tail else ''}")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("command", choices=["plan", "run", "status", "report"])
    parser.add_argument("--np", type=int, default=1)
    parser.add_argument("--out", default=str(REPOSITORY / "3D_Code" / "runs" / "final_v2"))
    parser.add_argument("--only", choices=STAGES)
    args = parser.parse_args()
    ctx = Context(args.out, args.np)
    if args.command == "plan":
        return plan(ctx)
    if args.command == "status":
        return status(ctx)
    if args.command == "report":
        return writeReport(ctx)
    return run(ctx, args.only)


if __name__ == "__main__":
    sys.exit(main() or 0)
