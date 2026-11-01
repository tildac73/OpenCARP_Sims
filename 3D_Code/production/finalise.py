#!/usr/bin/env python3
"""Final settings run: settle every open setting of the production model, in one resumable command.

Goal. Freeze the production model used for the parameter sweeps (v3-reference). Each setting is
chosen by a rule fixed in 3D_Code/configs/finalise.json before the run; this script only applies
the rules and writes the evidence. Simulations use the carputils production model
(3D_Code/production/slab.py); analysis uses the unchanged pipeline (run_boundary.py) plus the
exact contributing-tissue field (Shared_Pipeline/contribution_field.py).

Stages, and the final-model parameter each one decides (each writes <out>/<stage>/stage.json; a
rerun skips completed stages and simulations):
  preflight     -> nothing; records versions so the result is reproducible
  recompute     -> output step, time step, beats, domain (existing readiness runs; no simulation)
  tune          -> conductivities: 0.90 m/s at 0.5 and 0.25 mm (strips, minutes)
  run050        -> the 0.5 mm Vm used by the next two stages; gate: openCARP's bath is set up right
  forwardModel  -> openCARP pseudo-bidomain or the sigma_i source (same Vm, both fields compared)
  bathStudy     -> bath thickness (5 / 10 / 20 mm, chosen model, on the run050 Vm); scar effect for R10
  run025        -> the 0.25 mm level of the mesh decision (chosen bath and model)
  decide        -> mesh spacing (0.5 vs 0.25 mm at matched CV); writes productionModel.json, report.md
Optional, only if forwardModel chooses the sigma_i source: --only bidomainCheck (cost of full bidomain).

Usage (from the repository root, inside the efov-runner container):
    export MPIEXEC_BIN=/usr/local/lib/opencarp/lib/petsc/bin/mpiexec
    python3 3D_Code/production/finalise.py plan --np 24
    nohup python3 3D_Code/production/finalise.py run --np 24 > 3D_Code/runs/final_console.log 2>&1 &
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

STAGES = ["preflight", "recompute", "tune", "run050", "forwardModel", "bathStudy", "run025", "decide"]
OPTIONAL_STAGES = ["bidomainCheck"]         # only if the sigma_i source is needed (finalise.py run --only bidomainCheck)
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


def slabSpec(ctx, condition, spacing, bath, bidomain, tuned, state, extra=None):
    d, t = ctx.decisions(), ctx.settings["tissue"]
    domain = ctx.settings["domains"][d["domain"]]
    pace = ctx.base["pacing"]
    spec = dict(condition=condition, tissueSizeMm=[domain[0], domain[1], t["wallMm"]], spacingMm=spacing,
                bathThicknessMm=bath, bidomain=bidomain, conductivitiesSPerM=tuned["conductivitiesSPerM"],
                lesionConductivitySPerM=t["lesionConductivitySPerM"], bloodSPerM=t["bloodSPerM"],
                boundaryYmm=t["boundaryYmm"], timeStepUs=d["timeStepUs"], sampleStepMs=d["sampleStepMs"],
                pacing=dict(cycleLengthMs=pace["cycleLengthMs"], beatCount=d["beatCount"], durationMs=pace["durationMs"],
                            strength=tuned["stimulusStrength"], stripWidthMm=pace["stripWidthMm"]),
                initialState=str(state), meshRoot=str(ctx.out / "meshes"))
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
                        "source": f"finalise tune stage, {spacing} mm"}
    config["_purpose"] = "Final settings run; production-model candidate."
    boundary_setup.validateConfig(config)
    return config


def meshFor(ctx, spec):
    """Mesh of a spec with another bath thickness: mesher with the options carputils Block generates
    (mesh.generate itself needs a running carputils experiment)."""
    os.environ.setdefault("CARPUTILS_SETTINGS", carputilsEnvironment(ctx)["CARPUTILS_SETTINGS"])
    import slab
    name = f"{spec['condition']}_dx{spec['spacingMm']:g}_B{spec['bathThicknessMm']:g}_{spec['tissueSizeMm'][0]:g}x{spec['tissueSizeMm'][1]:g}x{spec['tissueSizeMm'][2]:g}"
    folder = Path(spec["meshRoot"]) / "bathStudy" / name
    base = folder / "block"
    if not Path(f"{base}.elem").exists():
        folder.mkdir(parents=True, exist_ok=True)
        options = [str(o) for o in slab.geometry(spec).mesher_opts(str(base))]
        with open(folder / "mesher.log", "w") as stream:
            subprocess.run([shutil.which("mesher") or "mesher"] + options, check=True, stdout=stream, stderr=subprocess.STDOUT)
    return str(base)


# ----------------------------------------------------------------- one analysed variant
def variantName(spacing, bath, source, scar="insulating"):
    return f"dx{spacing:.2f}_B{bath:g}_{source}" + ("" if scar == "insulating" else f"_{scar}")


def analyseVariant(ctx, folder, spacing, tuned, jobs, specs, bath, source, scar="insulating", fromOpenCarp=False):
    """Pipeline run + contribution field for one (mesh, bath, forward model, scar) on the jobs' Vm."""
    folder = Path(folder)
    summaryPath = folder / "variant.json"
    if summaryPath.exists():
        return load(summaryPath)
    config = pr.prepare(folder, analysisConfig(ctx, spacing, tuned), jobs, ctx.settings["windowRule"])
    t = ctx.settings["tissue"]
    models, check = {}, {}
    jobMesh = {c: pr.jobRecord(j)["meshBase"] for c, j in jobs.items()}
    for condition, job in jobs.items():
        sameMesh = specs[condition]["bathThicknessMm"] == bath
        meshBase = jobMesh[condition] if sameMesh else meshFor(ctx, {**specs[condition], "bathThicknessMm": bath})
        tissuePositions = None
        if not sameMesh:
            positions, elements, tags, _ = cf.readMesh(jobMesh[condition])
            tissuePositions = positions[np.unique(elements[tags != cf.BATH_TAG])]
        model = cf.BathModel(meshBase, tuned["conductivitiesSPerM"], t["lesionConductivitySPerM"], t["bloodSPerM"],
                             source, scar=scar, vmTissuePositions=tissuePositions)
        models[condition] = model
        interpolator, layout = pr.recoveryInterpolator(meshBase, folder)
        if fromOpenCarp:
            if not sameMesh or source != "sigmaM" or scar != "insulating":
                raise ValueError("openCARP electrograms exist only for the run's own bath, sigma_m, insulating scar")
            check[condition] = forwardModelCheck(ctx, job, model, interpolator, config, folder)
            signals, text = pr.openCarpSignals(job, interpolator, layout), "openCARP pseudo-bidomain phie.igb"
        else:
            signals = pr.femSignals(job, model, interpolator, layout, ctx.np)
            text = f"extracellular solve ({source}, scar {scar}, bath {bath} mm; per-electrode reciprocity)"
        log(f"electrograms: {folder.name} {condition}")
        pr.writeElectrograms(folder, config, condition, signals, text)
    log(f"pipeline: {folder.name}")
    status = pr.runPipeline(folder, 1)
    log(f"contribution field: {folder.name}")
    table = pr.groundTruth(folder, config, models, ctx.settings["contributionField"]["voxelMm"],
                           ctx.settings["contributionField"]["curveStepMm"])
    summary = dict(folder=str(folder), spacingMm=spacing, bathMm=bath, source=source, scar=scar,
                   electrograms="openCARP" if fromOpenCarp else "extracellular solve", pilotStatus=status,
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


def runPair(ctx, label, spacing, bath, bidomain=2, conditions=("intact", "lesion"), extra=None):
    t = tuned(ctx, spacing)
    state = conditioning(ctx, ctx.out / "conditioning" / f"dt{ctx.decisions()['timeStepUs']:g}_s{t['stimulusStrength']:g}",
                         ctx.decisions()["timeStepUs"], t["stimulusStrength"])
    jobs, specs, timing = {}, {}, {}
    for condition in conditions:
        spec = slabSpec(ctx, condition, spacing, bath, bidomain, t, state, extra)
        job = ctx.out / "sims" / f"{label}_{condition}"
        timing[condition] = runSlab(ctx, job, spec)
        jobs[condition], specs[condition] = job, spec
    return jobs, specs, timing


def stageRun050(ctx):
    """Base 0.5 mm run: the Vm every later 0.5 mm decision is computed on; gate that openCARP's bath is set up right."""
    bath = ctx.settings["runBathMm"]
    jobs, specs, timing = runPair(ctx, "dx0.50_pseudo", 0.5, bath)
    variant = analyseVariant(ctx, ctx.out / "variants" / variantName(0.5, bath, "sigmaM"), 0.5, tuned(ctx, 0.5),
                             jobs, specs, bath, "sigmaM", fromOpenCarp=True)
    if not all(c["passed"] for c in variant["forwardModelCheck"].values()):
        raise RuntimeError(f"openCARP phi_e does not match the sigma_m solve: {variant['forwardModelCheck']}")
    return dict(timing=timing, variant=variant)


def stageForwardModel(ctx):
    """Decides the forward model: openCARP pseudo-bidomain (sigma_m) vs bidomain physics (sigma_i), same Vm."""
    jobs, specs = pairFor(ctx, "dx0.50_pseudo")
    bath = ctx.settings["runBathMm"]
    analyseVariant(ctx, ctx.out / "variants" / variantName(0.5, bath, "sigmaI"), 0.5, tuned(ctx, 0.5), jobs, specs, bath, "sigmaI")
    row = compareRuns(ctx, ctx.out / "variants" / variantName(0.5, bath, "sigmaM"),
                      ctx.out / "variants" / variantName(0.5, bath, "sigmaI"), "forwardModel_sigmaM_vs_sigmaI",
                      ppMode="calibrated", useCv=False)
    return dict(comparison=row, chosen="sigmaM" if row["passed"] else "sigmaI")


def chosenSource(ctx):
    return ctx.stage("forwardModel")["chosen"]


def pairFor(ctx, label):
    jobs = {c: ctx.out / "sims" / f"{label}_{c}" for c in ["intact", "lesion"]}
    specs = {c: load(ctx.out / "sims" / f"{label}_{c}.json") for c in jobs}
    return jobs, specs


def stageBathStudy(ctx):
    """Decides the bath thickness (chosen forward model only); reports the scar effect for R10."""
    jobs, specs = pairFor(ctx, "dx0.50_pseudo")
    runBath, source, t050 = ctx.settings["runBathMm"], chosenSource(ctx), tuned(ctx, 0.5)
    baths = sorted(ctx.settings["bathStudyMm"])
    for bath in baths:
        analyseVariant(ctx, ctx.out / "variants" / variantName(0.5, bath, source), 0.5, t050, jobs, specs, bath, source,
                       fromOpenCarp=(source == "sigmaM" and bath == runBath))
    analyseVariant(ctx, ctx.out / "variants" / variantName(0.5, runBath, source, "conducting"), 0.5, t050, jobs, specs,
                   runBath, source, "conducting")
    comparisons = {}
    for thin, thick in zip(baths[:-1], baths[1:]):
        label = f"bath_{thin:g}_vs_{thick:g}"
        comparisons[label] = compareRuns(ctx, ctx.out / "variants" / variantName(0.5, thin, source),
                                         ctx.out / "variants" / variantName(0.5, thick, source), label, useCv=False)
    chosen, flag = baths[-1], "bath not converged at the thickest bath tested"
    for thin, thick in zip(baths[:-1], baths[1:]):
        if comparisons[f"bath_{thin:g}_vs_{thick:g}"]["passed"]:
            chosen, flag = thin, None
            break
    scar = compareRuns(ctx, ctx.out / "variants" / variantName(0.5, runBath, source, "conducting"),
                       ctx.out / "variants" / variantName(0.5, runBath, source), "scar_conducting_vs_insulating", useCv=False)
    return dict(source=source, comparisons=comparisons, chosenBathMm=chosen, flag=flag, scarEffect=scar)


def stageRun025(ctx):
    """0.25 mm run with the chosen bath and forward model: the finer level of the mesh decision."""
    bath, source = ctx.stage("bathStudy")["chosenBathMm"], chosenSource(ctx)
    jobs, specs, timing = runPair(ctx, f"dx0.25_pseudo_B{bath:g}", 0.25, bath)
    t025 = tuned(ctx, 0.25)
    folder = ctx.out / "variants" / variantName(0.25, bath, source)
    variant = analyseVariant(ctx, folder, 0.25, t025, jobs, specs, bath, source, fromOpenCarp=(source == "sigmaM"))
    check = variant["forwardModelCheck"]
    if source == "sigmaI":                          # the bath gate is defined on openCARP's own (sigma_m) field
        t = ctx.settings["tissue"]
        config = load(folder / "config.json")
        check = {}
        for condition, job in jobs.items():
            meshBase = pr.jobRecord(job)["meshBase"]
            model = cf.BathModel(meshBase, t025["conductivitiesSPerM"], t["lesionConductivitySPerM"], t["bloodSPerM"], "sigmaM")
            check[condition] = forwardModelCheck(ctx, job, model, pr.recoveryInterpolator(meshBase, folder)[0], config, folder)
    if not all(c["passed"] for c in check.values()):
        raise RuntimeError(f"openCARP phi_e does not match the sigma_m solve at 0.25 mm: {check}")
    return dict(timing=timing, variant=variant, forwardModelCheck=check)


def stageDecide(ctx):
    """Decides the mesh spacing (matched CV) and writes the final settings."""
    d, source = ctx.decisions(), chosenSource(ctx)
    bath = ctx.stage("bathStudy")["chosenBathMm"]
    # Matched CV: the meshes have different tuned conductivity magnitudes, so raw amplitudes differ by design;
    # the gate uses calibrated PP (spatial pattern) and reports raw PP.
    mesh = compareRuns(ctx, ctx.out / "variants" / variantName(0.5, bath, source),
                       ctx.out / "variants" / variantName(0.25, bath, source), "mesh_0.50_vs_0.25", ppMode="calibrated")
    spacing = 0.5 if mesh["passed"] else 0.25
    flags = []
    if spacing == 0.25:
        flags.append("0.5 vs 0.25 mm failed: 0.25 mm is used, but 0.25 vs 0.125 mm is not yet shown.")
    if ctx.stage("bathStudy").get("flag"):
        flags.append(ctx.stage("bathStudy")["flag"])
    if source == "sigmaI":
        flags.append("sigma_m and sigma_i differ: production needs the sigma_i source. Choose the route (openCARP full "
                     "bidomain, or the verified sigma_i solve on openCARP Vm); run --only bidomainCheck for its cost.")
    flags.append("Scar type (R10): see the scar effect in report.md.")
    model = dict(
        status="PROPOSED by the final settings run; becomes v3-reference once you confirm the flags",
        createdUtc=now(), domainMm=ctx.settings["domains"][d["domain"]] + [ctx.settings["tissue"]["wallMm"]],
        spacingMm=spacing, timeStepUs=d["timeStepUs"], sampleStepMs=d["sampleStepMs"], beatCount=d["beatCount"],
        bathMm=bath, bloodSPerM=ctx.settings["tissue"]["bloodSPerM"], forwardModel=SOURCE_TEXT[source],
        conductivitiesSPerM=tuned(ctx, spacing)["conductivitiesSPerM"], stimulusStrength=tuned(ctx, spacing)["stimulusStrength"],
        cvTargetMPerS=ctx.settings["tissue"]["cvTargetMPerS"], scar="insulating (pilot) unless you choose otherwise (R10)",
        measurementWindow="derived per run (DL-2026-10-28-08)", groundTruth="exact contribution field (contribution_field.py)",
        flags=flags)
    return dict(meshComparison=mesh, productionModel=model)


def stageBidomainCheck(ctx):
    """OPTIONAL (--only bidomainCheck), only if the sigma_i source is needed: cost and bath loading of full bidomain."""
    bath = ctx.settings["runBathMm"]
    duration = ctx.settings["bidomainCheck"]["durationMs"]
    jobs, specs, timing = runPair(ctx, "dx0.50_bidomain60", 0.5, bath, bidomain=1, conditions=("intact",),
                                  extra={"durationMs": duration})
    job, spec = jobs["intact"], specs["intact"]
    pseudoJob = ctx.out / "sims" / "dx0.50_pseudo_intact"
    t = ctx.settings["tissue"]
    meshBase = pr.jobRecord(job)["meshBase"]
    model = cf.BathModel(meshBase, tuned(ctx, 0.5)["conductivitiesSPerM"], t["lesionConductivitySPerM"], t["bloodSPerM"], "sigmaI")
    interpolator, _ = pr.recoveryInterpolator(meshBase, ctx.out / "variants" / variantName(0.5, bath, "sigmaM"))
    frames = igb_io.header(job / "phie.igb")[2]
    chosen = np.arange(int(round(10 / spec["sampleStepMs"])), frames, 5)
    bido = np.stack([interpolator @ pr.readPhie(job, slice(f, f + 1)).T[:, 0] for f in chosen], axis=1)
    femI = np.stack([interpolator @ model.solver.solve(model.fullVm(igb_io.read(job / "vm.igb", slice(f, f + 1)).T))[:, 0]
                     for f in chosen], axis=1)
    rel = lambda x, y: float(np.linalg.norm(x - y) / np.linalg.norm(y))
    import common
    nodes = igb_io.header(job / "vm.igb")[1]
    firstB = common.firstActivationInWindow(*common.readActivationFile(job / "act-thresh.dat", nodes), nodes, 0, duration)
    firstP = common.firstActivationInWindow(*common.readActivationFile(pseudoJob / "act-thresh.dat", nodes), nodes, 0, duration)
    both = np.isfinite(firstB) & np.isfinite(firstP)
    pseudoMs = spec["pacing"]["cycleLengthMs"] * ctx.decisions()["beatCount"]
    return dict(timing=timing, sigmaISolveVsBidomainRelativeRms=rel(femI, bido),
                activationDifferenceMsMax=float(np.abs(firstB[both] - firstP[both]).max()),
                costSecondsPerSimulatedMs=dict(bidomain=timing["intact"]["wallSeconds"] / duration,
                                               pseudoBidomain=load(ctx.out / "sims" / "dx0.50_pseudo_intact.timing.json")["wallSeconds"] / pseudoMs))


STAGE_FUNCTIONS = dict(preflight=stagePreflight, recompute=stageRecompute, tune=stageTune, run050=stageRun050,
                       forwardModel=stageForwardModel, bathStudy=stageBathStudy, run025=stageRun025, decide=stageDecide,
                       bidomainCheck=stageBidomainCheck)


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
        lines += ["## CV tuning", "", "| Spacing (mm) | CV (m/s) | scale of Roberts-Scher | sigma_il | sigma_el |", "| --- | --- | --- | --- | --- |"]
        for key, v in tu["tuned"].items():
            g = v["conductivitiesSPerM"]
            lines.append(f"| {key} | {fmt(v['achievedMPerS'])} | {fmt(v['scaleOfRobertsScher'])} | {fmt(g['g_il'])} | {fmt(g['g_el'])} |")
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
    fm = ctx.stage("forwardModel")
    if fm:
        lines += table("Forward model", "openCARP pseudo-bidomain if it passes, else the sigma_i source", [fm["comparison"]])
        lines += [f"Chosen: {SOURCE_TEXT[fm['chosen']]}", ""]
    bs = ctx.stage("bathStudy")
    if bs:
        lines += table("Bath thickness", "the thinnest bath that passes against the next thicker", bs["comparisons"].values())
        lines += [f"Chosen bath: {bs['chosenBathMm']:g} mm" + (f" ({bs['flag']})" if bs.get("flag") else ""), ""]
        lines += table("Scar type (R10, your decision)", "nothing automatically; size of the effect", [bs["scarEffect"]])
    de = ctx.stage("decide")
    if de:
        lines += table("Mesh spacing (matched CV)", "0.5 mm if it passes, else 0.25 mm", [de["meshComparison"]])
        lines += ["## Proposed production model", "", "```json", json.dumps(de["productionModel"], indent=2), "```", ""]
    b = ctx.stage("bidomainCheck")
    if b:
        lines += ["## Full bidomain (optional check)", "",
                  f"- sigma_i solve vs openCARP bidomain: relative RMS {fmt(b['sigmaISolveVsBidomainRelativeRms'])}",
                  f"- activation shift vs pseudo-bidomain: max {fmt(b['activationDifferenceMsMax'])} ms",
                  f"- cost: bidomain {fmt(b['costSecondsPerSimulatedMs']['bidomain'])} s, pseudo-bidomain "
                  f"{fmt(b['costSecondsPerSimulatedMs']['pseudoBidomain'])} s per simulated ms", ""]
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
    def nodes(spacing, bath):
        return int(np.prod([round(domain[0] / spacing) + 1, round(domain[1] / spacing) + 1, round((wall + bath) / spacing) + 1]))
    print(f"Domain {domain[0]:g} x {domain[1]:g} x {wall:g} mm ({d['domain']}); beats {d['beatCount']}; output {d['sampleStepMs']} ms")
    print("openCARP runs (pseudo-bidomain), each intact + lesion:")
    print(f"  run050  0.5 mm, bath {ctx.settings['runBathMm']:g} mm: {nodes(0.5, ctx.settings['runBathMm']):,} nodes")
    for bath in ctx.settings["bathStudyMm"]:
        print(f"  run025  0.25 mm if the bath study chooses {bath:g} mm: {nodes(0.25, bath):,} nodes")
    print("Extracellular solves on the run050 Vm: bath " + ", ".join(f"{b:g} mm ({nodes(0.5, b):,} nodes)" for b in ctx.settings["bathStudyMm"]))
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
    parser.add_argument("--out", default=str(REPOSITORY / "3D_Code" / "runs" / "final"))
    parser.add_argument("--only", choices=STAGES + OPTIONAL_STAGES)
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
