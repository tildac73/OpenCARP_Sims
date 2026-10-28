"""Step 3: conduction-velocity tuning and CV resolution study on thin tetrahedral strips.

Method (openCARP example 02_EP_tissue/03A_study_prep_tuneCV): measure CV for a planar wave,
multiply all intracellular and extracellular conductivities by (CV_target / CV)^2, repeat until
|CV - target| / target <= tolerance. All four Roberts and Scher ratios stay fixed (DL-2026-10-28-04).

Why not the tuneCV tool itself: it uses a one-element hexahedral cable, whereas the slab is
tetrahedral from openCARP's mesher, and CV depends on element type at a given spacing. Here the
strip is built by the same mesher call, with the same solver settings (3d_pilot.par),
Courtemanche limit-cycle state, pacing protocol and LAT threshold as the slab.

Strip: 20 mm along the propagation axis, 1 x 1 mm cross-section, fibres along x. A planar wave in
a uniform strip with no-flux sides has the same CV as a planar wave in the 4 mm slab, so the
cross-section only has to hold a few elements. 'x' strips measure longitudinal CV, 'y' strips
transverse CV.
"""
from pathlib import Path

import numpy as np

import common
import boundary_setup

AXES = {"x": 0, "y": 1}


def scaledConductivities(base, scale):
    return {name: float(value) * scale for name, value in base.items()}


def monodomainLongitudinal(conductivities):
    gi, ge = conductivities["g_il"], conductivities["g_el"]
    return gi * ge / (gi + ge)


def stripSize(direction, lengthMm, crossMm):
    size = [crossMm, crossMm, crossMm]
    size[AXES[direction]] = lengthMm
    return size


def conditioningFolder(stageFolder, timeStepUs, strength, settings):
    """bench limit cycle (as boundary_setup.conditionCells), one per time step and strength."""
    folder = Path(stageFolder) / f"conditioning_dt{timeStepUs:g}_s{strength:g}"
    config = dict(timeStepUs=timeStepUs, conditioning=dict(enabled=True, beatCount=settings["conditioningBeats"]),
                  pacing=dict(cycleLengthMs=settings["cycleLengthMs"], durationMs=settings["stimulus"]["durationMs"],
                              strength=strength))
    folder.mkdir(parents=True, exist_ok=True)
    boundary_setup.conditionCells(folder, config)
    return folder / "conditioning" / "Courtemanche.sv"


def prepareStrip(jobFolder, direction, spacingMm, conductivities, settings, strength, timeStepUs, stateFile,
                 durationMs=None):
    """Mesh, stimulus and parameter file for one strip run; returns (parameter file, positions)."""
    jobFolder = Path(jobFolder)
    jobFolder.mkdir(parents=True, exist_ok=True)
    size = stripSize(direction, settings["stripLengthMm"], settings["stripCrossSectionMm"])
    positions, elements = common.generateBox(jobFolder / "mesh", "strip", size, spacingMm)
    # Re-write with explicit x fibres (the slab's convention) rather than relying on mesher defaults.
    common.writeMesh(jobFolder / "mesh" / "strip", positions, elements, np.ones(len(elements), int))
    axis = AXES[direction]
    stimulated = np.flatnonzero(positions[:, axis] <= positions[:, axis].min() + settings["stimulus"]["stripWidthMm"] + 1e-9)
    common.writeVertices(jobFolder / "stimulus.vtx", stimulated)
    beats = settings["beatCount"]
    cycle = settings["cycleLengthMs"]
    tend = durationMs if durationMs is not None else (beats - 1) * cycle + settings["lastBeatDurationMs"]
    values = {
        "meshname": jobFolder / "mesh" / "strip", "simID": jobFolder / "sim",
        "dt": timeStepUs, "spacedt": settings["outputStepMs"], "tend": tend,
        "stim[0].elec.vtx_file": jobFolder / "stimulus.vtx",
        "stim[0].pulse.strength": strength, "stim[0].ptcl.duration": settings["stimulus"]["durationMs"],
        "stim[0].ptcl.bcl": cycle, "stim[0].ptcl.npls": beats,
        "imp_region[0].im_sv_init": stateFile,
    }
    parameterFile = common.writeParameterFile(jobFolder / "strip.par", common.templateForTissue(conductivities), values)
    return parameterFile, positions


def measureStrip(jobFolder, positions, direction, settings):
    """Last-beat CV from openCARP LATs; also checks every beat activated the measurement range."""
    axis = AXES[direction]
    nodes, times = common.readActivationFile(common.findActivationFile(Path(jobFolder) / "sim"), len(positions))
    lo, hi = settings["measureRangeMm"]
    inRange = (positions[:, axis] >= lo) & (positions[:, axis] <= hi)
    cycle = settings["cycleLengthMs"]
    captured = []
    for beat in range(settings["beatCount"]):
        first = common.firstActivationInWindow(nodes, times, len(positions), beat * cycle, (beat + 1) * cycle)
        captured.append(bool(np.isfinite(first[inRange]).all()))
    last = common.firstActivationInWindow(nodes, times, len(positions), (settings["beatCount"] - 1) * cycle,
                                          settings["beatCount"] * cycle)
    result = dict(capturedEveryBeat=all(captured), capturedByBeat=captured)
    if all(captured):
        result.update(common.fitPlanarSpeed(positions[inRange], last[inRange], axis))
        firstBeat = common.firstActivationInWindow(nodes, times, len(positions), 0, cycle)
        result["firstBeatSpeedMPerS"] = common.fitPlanarSpeed(positions[inRange], firstBeat[inRange], axis)["speedMPerS"]
    return result


def runStrip(jobFolder, direction, spacingMm, conductivities, settings, strength, timeStepUs, stageFolder,
             processCount, smoke=False):
    """One strip simulation, reused if already complete. Returns the measurement dict."""
    jobFolder = Path(jobFolder)
    resultPath = jobFolder / "result.json"
    if resultPath.exists() and not smoke:
        return common.loadSettings(resultPath)
    if jobFolder.exists() and any(jobFolder.iterdir()) and not smoke:
        raise FileExistsError(f"{jobFolder} is incomplete; move it aside before rerunning")
    stateFile = conditioningFolder(stageFolder, timeStepUs, strength, settings)
    parameterFile, positions = prepareStrip(jobFolder, direction, spacingMm, conductivities, settings, strength,
                                            timeStepUs, stateFile)
    prefix, ranks = common.simulatorPrefix(processCount, len(positions))
    if smoke:
        smokePar = common.smokeCopy(parameterFile, jobFolder / "smoke")
        common.runLogged(prefix + ["+F", smokePar], jobFolder / "smoke.log")
        common.checkSingleLaunch(jobFolder / "smoke.log")
        return dict(smokePassed=True)
    common.runLogged(prefix + ["+F", parameterFile], jobFolder / "openCARP.log")
    common.checkSingleLaunch(jobFolder / "openCARP.log")
    result = measureStrip(jobFolder, positions, direction, settings)
    result.update(direction=direction, spacingMm=spacingMm, timeStepUs=timeStepUs, strength=strength,
                  conductivitiesSPerM=conductivities, monodomainLongitudinalSPerM=monodomainLongitudinal(conductivities),
                  nodeCount=len(positions), ranks=ranks, wallSeconds=common.lastProgressSeconds(jobFolder / "openCARP.log"))
    common.saveJson(resultPath, result)
    return result


def tune(stageFolder, label, targetMPerS, spacingMm, settings, strength, processCount, allowCaptureSearch=False):
    """Iterate the documented update rule. Returns (scale, strength, history, converged)."""
    base = settings["baseConductivitiesSPerM"]
    scale = float(settings["initialScale"])
    history = []
    doublings = 0
    for iteration in range(settings["maximumIterations"]):
        conductivities = scaledConductivities(base, scale)
        job = Path(stageFolder) / f"tune_{label}" / f"iter{iteration:02d}_s{strength:g}"
        result = runStrip(job, "x", spacingMm, conductivities, settings, strength, settings["timeStepUs"],
                          stageFolder, processCount)
        if not result["capturedEveryBeat"]:
            if allowCaptureSearch and doublings < settings["maximumStrengthDoublings"]:
                doublings += 1
                strength *= 2
                history.append(dict(iteration=iteration, scale=scale, captured=False, action=f"strength doubled to {strength:g}"))
                continue
            history.append(dict(iteration=iteration, scale=scale, captured=False, action="stopped: no capture"))
            return scale, strength, history, False
        cv = result["speedMPerS"]
        error = (cv - targetMPerS) / targetMPerS
        history.append(dict(iteration=iteration, scale=scale, speedMPerS=cv, relativeError=error,
                            firstBeatSpeedMPerS=result["firstBeatSpeedMPerS"], strength=strength))
        if abs(error) <= settings["tuningToleranceRelative"]:
            return scale, strength, history, True
        scale *= (targetMPerS / cv) ** 2
    return scale, strength, history, False


def runStage(stageFolder, settings, processCount):
    """Full CV programme. Blocking result: the reference tuning at the target CV."""
    stageFolder = Path(stageFolder)
    strength = float(settings["stimulus"]["strength"])
    reference = settings["referenceSpacingMm"]
    scale, strength, history, converged = tune(stageFolder, f"target{settings['targetMPerS']:g}_dx{reference:g}",
                                               settings["targetMPerS"], reference, settings, strength, processCount,
                                               allowCaptureSearch=True)
    summary = dict(targetMPerS=settings["targetMPerS"], referenceSpacingMm=reference, converged=converged,
                   history=history, strength=strength, scale=scale)
    if not converged:
        common.saveJson(stageFolder / "cvSummary.json", summary)
        return summary
    tuned = scaledConductivities(settings["baseConductivitiesSPerM"], scale)
    summary["conductivitiesSPerM"] = tuned
    evaluations = []
    for spacing in settings["evaluationSpacingsMm"]:
        result = runStrip(stageFolder / f"eval_x_dx{spacing:g}", "x", spacing, tuned, settings, strength,
                          settings["timeStepUs"], stageFolder, processCount)
        evaluations.append(dict(kind="spacing", direction="x", spacingMm=spacing, timeStepUs=settings["timeStepUs"], **_short(result)))
    for timeStep in settings["evaluationTimeStepsUs"]:
        result = runStrip(stageFolder / f"eval_x_dx{reference:g}_dt{timeStep:g}", "x", reference, tuned, settings,
                          strength, timeStep, stageFolder, processCount)
        evaluations.append(dict(kind="timeStep", direction="x", spacingMm=reference, timeStepUs=timeStep, **_short(result)))
    transverse = runStrip(stageFolder / f"eval_y_dx{reference:g}", "y", reference, tuned, settings, strength,
                          settings["timeStepUs"], stageFolder, processCount)
    evaluations.append(dict(kind="transverse", direction="y", spacingMm=reference, timeStepUs=settings["timeStepUs"], **_short(transverse)))
    for spacing in settings["pilotConductivitySpacingsMm"]:
        result = runStrip(stageFolder / f"pilotConductivity_x_dx{spacing:g}", "x", spacing,
                          settings["pilotConductivitiesSPerM"], settings, strength, settings["timeStepUs"],
                          stageFolder, processCount)
        evaluations.append(dict(kind="pilotConductivity", direction="x", spacingMm=spacing, timeStepUs=settings["timeStepUs"], **_short(result)))
    summary["evaluations"] = evaluations
    summary["transverseSpeedMPerS"] = transverse.get("speedMPerS")
    if transverse.get("speedMPerS"):
        summary["speedAnisotropyRatio"] = settings["targetMPerS"] / transverse["speedMPerS"]
    sensitivity = {}
    for target in settings["sensitivityTargetsMPerS"]:
        s, sensitivityStrength, h, ok = tune(stageFolder, f"target{target:g}_dx{reference:g}", target, reference,
                                             settings, strength, processCount, allowCaptureSearch=True)
        sensitivity[f"{target:g}"] = dict(converged=ok, scale=s, history=h, stimulusStrength=sensitivityStrength,
                                          conductivitiesSPerM=scaledConductivities(settings["baseConductivitiesSPerM"], s) if ok else None)
    summary["sensitivity"] = sensitivity
    common.saveJson(stageFolder / "cvSummary.json", summary)
    common.saveJson(stageFolder / "tuned.json", dict(
        conductivitiesSPerM=tuned, stimulusStrength=strength, scaleOfRobertsScher=scale,
        targetMPerS=settings["targetMPerS"], achievedMPerS=history[-1]["speedMPerS"], referenceSpacingMm=reference,
        interpretation="Calibration: Roberts and Scher 1982 ratios, magnitudes scaled to the CV target (DL-2026-10-28-04). Not a measurement."))
    return summary


def _short(result):
    keys = ["capturedEveryBeat", "speedMPerS", "firstBeatSpeedMPerS", "directionErrorDeg", "planeResidual95Ms", "nodeCount", "wallSeconds"]
    return {key: result.get(key) for key in keys}


def smoke(stageFolder, settings, processCount):
    """1 ms runs of both strip orientations at the coarsest spacing (parameter-name and file checks)."""
    conductivities = scaledConductivities(settings["baseConductivitiesSPerM"], settings["initialScale"])
    for direction in ("x", "y"):
        runStrip(Path(stageFolder) / f"strip_{direction}", direction, 0.5, conductivities, settings,
                 settings["stimulus"]["strength"], settings["timeStepUs"], stageFolder, processCount, smoke=True)
