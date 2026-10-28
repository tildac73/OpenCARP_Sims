"""Step 3: 3D convergence ladder on the tuned model (mesh, time step, output interval, beats, domain).

Every run is the unchanged pilot pipeline (run_boundary.py: mesher -> bench -> openCARP with
phi_e recovery -> physiology checks -> EGM measures -> contributions -> change points). Only the
config differs. Physical parameters are identical across levels: conductivities and stimulus
strength from the CV stage (tuned.json), so differences between levels are discretisation or
domain effects.

Measurement window: derived per run from that run's intact activation map before any EGM is
measured (rule in readiness_kit.json, PROPOSED DL-2026-10-28-08), because the 25-100 ms window of
DL-2026-10-27-01 was set from the pilot's 0.35 m/s timing.

Comparisons follow DL-2026-10-28-05: CV <= 2 %, PP <= 5 %, FOV80 <= 0.5 mm, change point <= 0.5 mm.
PP is compared before calibration (PP / gain), because each run calibrates its own gain and a
calibrated PP would hide amplitude changes. FOV80 is gain-independent. Change points use each run's
own calibrated analysis, as the clinical method would.
"""
import copy
import json
from pathlib import Path
import shutil
import subprocess
import sys

import numpy as np
import pandas as pd

import common
import boundary_setup
import igb_io

RUNNER = common.REPOSITORY / "3D_Code" / "run_boundary.py"
TOLERANCES = {"cvRelative": 0.02, "ppRelative": 0.05, "fov80Mm": 0.5, "changePointMm": 0.5}
FIT_KEYS = ["condition", "gapMm", "noiseRepeat", "kind", "label", "distanceConvention", "preprocessing"]
POINT_KEYS = ["condition", "noiseRepeat", "centreYmm", "gapMm", "kind", "label"]


# ----------------------------------------------------------------- configs
def makeConfig(base, run, settings, tuned):
    config = copy.deepcopy(base)
    domain = run["domain"]
    if domain == "reduced":
        config["slabSizeMm"] = list(settings["reducedSlabSizeMm"])
        config["electrodes"]["centreYmm"] = list(settings["reducedCentreYmm"])
        config["acceptance"]["viableRegionYmm"] = list(settings["reducedViableRegionYmm"])
        config["analysis"]["groundTruthCentreYmm"] = list(settings["reducedGroundTruthCentreYmm"])
    elif domain == "large":
        config["slabSizeMm"] = list(settings["largeSlabSizeMm"])
    elif domain != "pilot":
        raise ValueError(f"Unknown domain {domain}")
    spacing = float(run["spacingMm"])
    config["meshSpacingMm"] = spacing
    config["timeStepUs"] = float(run["timeStepUs"])
    config["sampleStepMs"] = float(run["sampleStepMs"])
    config["pacing"]["beatCount"] = int(run.get("beatCount", settings["beatCount"]))
    config["pacing"]["strength"] = float(tuned["stimulusStrength"])
    config["acceptance"]["responseWindowMs"] = list(settings["physiologyResponseWindowMs"])
    config["analysis"]["beatWindowMs"] = [25.0, 100.0]   # placeholder; replaced by deriveWindow before analysis
    config["analysis"]["leadFieldQuadrature"] = {"nearRadiusMm": settings["quadratureNearSpacings"] * spacing,
                                                 "subdivisionLevel": settings["quadratureLevel"]}
    config["tissue"] = {"conductivitiesSPerM": dict(tuned["conductivitiesSPerM"]),
                        "source": "readiness CV stage (tuned.json); Roberts and Scher ratios scaled to the CV target"}
    config["_purpose"] = f"Readiness ladder run {run['name']}; not a reference-model result."
    boundary_setup.validateConfig(config)
    return config


def deriveWindow(runFolder, config, settings):
    """Measurement window from intact last-beat activation under the grid (top surface, |x| <= extent)."""
    runFolder = Path(runFolder)
    positions = boundary_setup.readNodePositions(runFolder / "mesh" / "intact")
    nodes, times = common.readActivationFile(common.findActivationFile(runFolder / "intact"), len(positions))
    cycle = config["pacing"]["cycleLengthMs"]
    beatStart = (config["pacing"]["beatCount"] - 1) * cycle
    first = common.firstActivationInWindow(nodes, times, len(positions), beatStart, beatStart + cycle) - beatStart
    top = np.isclose(positions[:, 2], positions[:, 2].max())
    under = top & (np.abs(positions[:, 0]) <= settings["gridHalfExtentXmm"])
    if not np.isfinite(first[under]).all():
        raise ValueError("Not every surface node under the grid activated in the last beat")
    step = config["sampleStepMs"]
    earliest, latest = float(first[under].min()), float(first[under].max())
    start = max(earliest - settings["windowLeadMs"], config["pacing"]["durationMs"] + settings["windowMinimumAfterStimulusMs"])
    end = latest + settings["windowTrailMs"]
    start, end = np.floor(start / step + 1e-9) * step, np.ceil(end / step - 1e-9) * step   # outward to the sample grid
    if not 0 <= start < end < cycle:
        raise ValueError(f"Derived window {start}-{end} ms does not fit the cycle")
    return dict(windowMs=[round(float(start), 6), round(float(end), 6)], gridFirstActivationMs=earliest,
                gridLastActivationMs=latest, rule={k: settings[k] for k in
                ["windowLeadMs", "windowTrailMs", "windowMinimumAfterStimulusMs", "gridHalfExtentXmm"]},
                nodesUsed=int(under.sum()))


# ----------------------------------------------------------------- output interval
def decimateIgb(source, target, factor, block=1000):
    fields, nodes, frames, _ = igb_io.header(source)
    keep = np.arange(0, frames, factor)
    increment = float(fields["inc_t"]) * factor if "inc_t" in fields else float(fields["dim_t"]) / (frames - 1) * factor
    header = {**fields, "t": str(len(keep)), "inc_t": f"{increment:g}", "type": "float", "systeme": "little_endian"}
    header.pop("facteur", None); header.pop("zero", None)
    text = " ".join(f"{k}:{v}" for k, v in header.items())
    if len(text) > 1023:
        raise ValueError("IGB header too long")
    with open(target, "wb") as stream:
        stream.write(text.encode("latin-1").ljust(1023, b" ") + b"\n")
        for first in range(0, len(keep), block):
            selected = keep[first:first + block]
            data = igb_io.read(source, slice(selected[0], selected[-1] + 1, factor))
            stream.write(data.astype("<f4").tobytes())


def decimateRun(sourceFolder, targetFolder, config, newStepMs):
    """Analysis-only copy at a coarser output interval (output sampling does not change the solution)."""
    sourceFolder, targetFolder = Path(sourceFolder), Path(targetFolder)
    factor = round(newStepMs / config["sampleStepMs"])
    if not np.isclose(factor * config["sampleStepMs"], newStepMs) or factor < 2:
        raise ValueError("New output interval must be an integer multiple of the original")
    if (targetFolder / "lesion" / "meta.json").exists():
        return targetFolder
    targetFolder.mkdir(parents=True, exist_ok=True)
    decimated = copy.deepcopy(config)
    decimated["sampleStepMs"] = newStepMs
    (targetFolder / "mesh").exists() or (targetFolder / "mesh").symlink_to(sourceFolder / "mesh", target_is_directory=True)
    for name in ["electrodes", "conditioning", "sourceSnapshot"]:
        if not (targetFolder / name).exists():
            shutil.copytree(sourceFolder / name, targetFolder / name)
    for path in list(sourceFolder.glob("*.vtx")) + list(sourceFolder.glob("*.par")) + list(sourceFolder.glob("*.log")):
        shutil.copy2(path, targetFolder / path.name)
    common.copyIfExists(sourceFolder / "manifest.json", targetFolder / "manifest.json")
    boundary_setup.saveJson(targetFolder / "config.json", decimated)
    for condition in ["intact", "lesion"]:
        src, dst = sourceFolder / condition, targetFolder / condition
        dst.mkdir(exist_ok=True)
        for name in ["vm.igb", "phie_recovery.igb"]:
            decimateIgb(src / name, dst / name, factor)
        for name in ["act-thresh.dat", "meshname.txt"]:
            common.copyIfExists(src / name, dst / name)
        text = (src / "parameters.par").read_text()
        (dst / "parameters.par").write_text(text + f"\n# readiness kit: output decimated x{factor} to {newStepMs} ms (analysis only)\n")
        boundary_setup.buildEGMs(dst, targetFolder, decimated)
        meta = json.loads((src / "meta.json").read_text())
        meta.update(decimatedFrom=str(src), decimationFactor=factor)
        boundary_setup.saveJson(dst / "meta.json", meta)
    return targetFolder


# ----------------------------------------------------------------- running
def runPipeline(arguments, logPath, processCount):
    command = [sys.executable, RUNNER, *arguments, "--np", processCount]
    return common.runLogged(command, logPath, check=False)


def analyse(runFolder, configPath, processCount):
    """--process-only; on physiology failure rerun as --diagnostic so outputs exist (and are flagged)."""
    code = runPipeline(["--config", configPath, "--output", runFolder, "--process-only"], Path(runFolder) / "analysis.log", processCount)
    status = json.loads((Path(runFolder) / "physiologyStatus.json").read_text()) if (Path(runFolder) / "physiologyStatus.json").exists() else {}
    diagnosticCode = None
    if code == 2 and not status.get("passed", True):
        # Physiology failed: produce diagnostic outputs so the failure can be inspected; results stay flagged.
        diagnosticCode = runPipeline(["--config", configPath, "--output", runFolder, "--process-only", "--diagnostic"],
                                     Path(runFolder) / "analysis_diagnostic.log", processCount)
    elif code not in (0, 2):
        raise RuntimeError(f"{runFolder}: analysis failed (exit {code}); see analysis.log")
    pilot = Path(runFolder) / "results" / "pilotStatus.json"
    return dict(exitCode=code, diagnosticExitCode=diagnosticCode, physiologyPassed=status.get("passed"),
                physiologyCriteria=status.get("criteria"), pilotStatus=json.loads(pilot.read_text()) if pilot.exists() else None)


def runOne(stageFolder, run, base, settings, tuned, processCount):
    folder = Path(stageFolder) / run["name"]
    donePath = folder / "ladderRun.json"
    if donePath.exists():
        return json.loads(donePath.read_text())
    config = makeConfig(base, run, settings, tuned)
    folder.mkdir(parents=True, exist_ok=True)
    configPath = folder.parent / f"{run['name']}.config.json"
    if not configPath.exists():
        boundary_setup.saveJson(configPath, config)
    code = runPipeline(["--config", configPath, "--output", folder, "--simulate-only"], folder.parent / f"{run['name']}.simulate.log", processCount)
    if code != 0:
        raise RuntimeError(f"{run['name']}: simulation failed (exit {code}); see {run['name']}.simulate.log")
    window = deriveWindow(folder, config, settings)
    config["analysis"]["beatWindowMs"] = window["windowMs"]
    boundary_setup.saveJson(configPath, config)
    common.saveJson(folder / "windowDerivation.json", window)
    record = dict(name=run["name"], run=run, window=window, analysis=analyse(folder, configPath, processCount),
                  wallSeconds={c: common.lastProgressSeconds(folder / f"{c}.log") for c in ["intact", "lesion"]},
                  openCARP=common.openCarpBanner(folder / "intact.log"))
    if run.get("decimateTo"):
        decimatedFolder = decimateRun(folder, folder / f"decimated_out{run['decimateTo']:.2f}", config, run["decimateTo"])
        decimatedConfig = copy.deepcopy(config); decimatedConfig["sampleStepMs"] = run["decimateTo"]
        decimatedConfigPath = folder.parent / f"{run['name']}.decimated.config.json"
        boundary_setup.saveJson(decimatedConfigPath, decimatedConfig)
        record["decimatedAnalysis"] = analyse(decimatedFolder, decimatedConfigPath, processCount)
    common.saveJson(donePath, record)
    return record


# ----------------------------------------------------------------- comparison
def loadAnalysed(runFolder):
    results = Path(runFolder) / "results"
    gain = json.loads((results / "calibration.json").read_text())["gain"]
    points = pd.concat([pd.read_csv(results / f"{c}Points.csv") for c in ["intact", "lesion"]], ignore_index=True)
    points["rawCleanPp"] = points.cleanAtSelectedWeightsPpMv / gain
    contributions = pd.read_csv(results / "contributions.csv").merge(
        points[["recordingId"] + POINT_KEYS], on="recordingId", how="left", validate="one_to_one")
    fits = pd.read_csv(results / "clinical" / "changePointFits.csv")
    fits = fits[fits.selected.astype(str).str.lower() == "true"]
    physiology = pd.read_csv(Path(runFolder) / "physiologyChecks.csv")
    lastBeat = physiology[physiology.condition == "intact"].beatNumber.max()
    speed = float(physiology[(physiology.condition == "intact") & (physiology.beatNumber == lastBeat)].speedMmPerMs.iloc[0])
    statusPath = Path(runFolder) / "physiologyStatus.json"
    physiologyPassed = json.loads(statusPath.read_text()).get("passed") if statusPath.exists() else None
    return dict(gain=gain, points=points, contributions=contributions, fits=fits, speed=speed,
                placements=sorted(points.centreYmm.unique()), physiologyPassed=physiologyPassed)


def compare(coarseFolder, fineFolder, label):
    c, f = loadAnalysed(coarseFolder), loadAnalysed(fineFolder)
    samePlacements = c["placements"] == f["placements"]
    points = c["points"].merge(f["points"], on=POINT_KEYS, suffixes=("C", "F"))
    clean = points[points.noiseRepeat == 0]
    relative = (clean.rawCleanPpF - clean.rawCleanPpC).abs() / clean.rawCleanPpF
    near = clean.distanceInsideMmF.abs() <= 10
    contributions = c["contributions"].merge(f["contributions"], on=POINT_KEYS, suffixes=("C", "F"))
    fovDifference = (contributions.fov80ReconstructionMmF - contributions.fov80ReconstructionMmC).abs()
    fits = c["fits"].merge(f["fits"], on=FIT_KEYS, suffixes=("C", "F"))
    primary = fits[(fits.preprocessing == "clinicalTrimmed") & (fits.distanceConvention == "recordingAnchor")]
    cpDifference = (primary.changePointMmF - primary.changePointMmC).abs()
    supportedC = primary.fitStatusC == "supportedDescriptiveChangePoint"
    supportedF = primary.fitStatusF == "supportedDescriptiveChangePoint"
    supportedEither = primary[supportedC | supportedF]
    summary = dict(
        label=label, coarse=str(coarseFolder), fine=str(fineFolder), tolerances=TOLERANCES,
        cvCoarseMPerS=c["speed"], cvFineMPerS=f["speed"], cvRelative=abs(c["speed"] - f["speed"]) / f["speed"],
        gainCoarse=c["gain"], gainFine=f["gain"],
        physiologyPassedCoarse=c["physiologyPassed"], physiologyPassedFine=f["physiologyPassed"],
        matchedCleanSignals=int(len(clean)),
        ppRelativeMaxAll=float(relative.max()), ppRelativeMaxWithin10mm=float(relative[near].max()),
        ppRelativeMedian=float(relative.median()),
        fov80MaxAbsMm=float(fovDifference.max()) if len(fovDifference) else None,
        fov80Changed=int((fovDifference > 0).sum()), fov80Count=int(len(contributions)),
        changePointComparable=bool(samePlacements),
        changePointMaxAbsMmAllPrimary=float(cpDifference.max()) if len(cpDifference) else None,
        changePointMaxAbsMmSupported=float((supportedEither.changePointMmF - supportedEither.changePointMmC).abs().max()) if len(supportedEither) else None,
        primaryFits=int(len(primary)), fitStatusChanges=int((primary.fitStatusC != primary.fitStatusF).sum()),
        supportStatusChanges=int((supportedC != supportedF).sum()),
        selectedModelChanges=int((primary.modelC != primary.modelF).sum()),
    )
    basic = (summary["cvRelative"] <= TOLERANCES["cvRelative"] and summary["ppRelativeMaxAll"] <= TOLERANCES["ppRelative"]
             and (summary["fov80MaxAbsMm"] or 0) <= TOLERANCES["fov80Mm"])
    if samePlacements:
        strict = basic and (summary["changePointMaxAbsMmAllPrimary"] or 0) <= TOLERANCES["changePointMm"] and summary["fitStatusChanges"] == 0
        clarified = basic and (summary["changePointMaxAbsMmSupported"] or 0) <= TOLERANCES["changePointMm"] and summary["supportStatusChanges"] == 0
    else:
        strict = clarified = basic
    summary.update(passedStrict=bool(strict), passedClarified=bool(clarified),
                   note=None if samePlacements else "Placement sets differ; change points not compared.")
    return summary


def runStage(stageFolder, settings, tuned, processCount, tiers=("A",)):
    stageFolder = Path(stageFolder)
    base = json.loads((common.REPOSITORY / settings["baseConfig"]).read_text())
    records, failures = {}, {}
    for run in settings["runs"]:
        if run["tier"] not in tiers:
            continue
        try:
            records[run["name"]] = runOne(stageFolder, run, base, settings, tuned, processCount)
        except (RuntimeError, ValueError, OSError, subprocess.CalledProcessError) as error:
            failures[run["name"]] = f"{type(error).__name__}: {error}"
            print(f"Ladder run {run['name']} failed: {error}", flush=True)
    comparisons = []
    for item in settings["comparisons"]:
        coarse, fine = stageFolder / item["coarse"], stageFolder / item["fine"]
        if not ((coarse / "results" / "contributions.csv").exists() and (fine / "results" / "contributions.csv").exists()):
            comparisons.append(dict(label=item["label"], skipped="one or both runs not analysed"))
            continue
        try:
            comparisons.append(compare(coarse, fine, item["label"]))
        except (KeyError, ValueError, FileNotFoundError) as error:
            comparisons.append(dict(label=item["label"], error=f"{type(error).__name__}: {error}"))
    summary = dict(runs=records, failures=failures, comparisons=comparisons, tiers=list(tiers))
    common.saveJson(stageFolder / "ladderSummary.json", summary)
    pd.DataFrame([{k: v for k, v in row.items() if not isinstance(v, dict)} for row in comparisons]).to_csv(
        stageFolder / "convergence.csv", index=False)
    return summary


def smoke(stageFolder, settings, processCount, provisionalTuned):
    """Prepare and condition one ladder config, then a 1 ms openCARP run of each condition."""
    stageFolder = Path(stageFolder)
    base = json.loads((common.REPOSITORY / settings["baseConfig"]).read_text())
    stageFolder.mkdir(parents=True, exist_ok=True)
    run = dict(settings["runs"][0]); run["name"] = "smoke_" + run["name"]
    config = makeConfig(base, run, settings, provisionalTuned)
    folder = stageFolder / run["name"]
    configPath = stageFolder / f"{run['name']}.config.json"
    boundary_setup.saveJson(configPath, config)
    code = runPipeline(["--config", configPath, "--output", folder, "--condition-only"], stageFolder / f"{run['name']}.prepare.log", processCount)
    if code != 0:
        raise RuntimeError(f"Ladder smoke preparation failed (exit {code})")
    nodes = len(boundary_setup.readNodePositions(folder / "mesh" / "intact"))
    prefix, _ = common.simulatorPrefix(processCount, nodes)
    for condition in ["intact", "lesion"]:
        smokePar = common.smokeCopy(folder / f"{condition}.par", stageFolder / f"smoke_{condition}")
        common.runLogged(prefix + ["+F", smokePar], stageFolder / f"smoke_{condition}.log")
        common.checkSingleLaunch(stageFolder / f"smoke_{condition}.log")
        if not (stageFolder / f"smoke_{condition}" / "phie_recovery.igb").exists():
            raise FileNotFoundError(f"Ladder smoke ({condition}): openCARP wrote no phie_recovery.igb")
