"""Physiology checks of each tissue run, and the measurement window derived from activation.

Checks on the last three beats, in a viable region away from stimulus, edges and boundary
(acceptance.physiology): one activation per beat, complete repolarisation inside the response
window, quiet lesion core, planar wave in +x, beat-to-beat stability of latency, amplitude and
APD90, and conduction velocity within tolerance of the target. These are project engineering
criteria, not validated biological limits.

Activation time = first linearly interpolated upward crossing of the threshold in Vm sampled at
the output step. APD90 = time from activation to the first downward crossing of 10 % of the
amplitude above the pre-beat voltage.
"""
from pathlib import Path

import numpy as np
import pandas as pd

from .geometry import signedDistance
from .io import firstActivation, igbTimes, readActivations, readIgb, saveJson


def activationTimes(timeMs, voltageMv, thresholdMv):
    """First interpolated upward crossing and the number of crossings per node."""
    crossings = (voltageMv[:-1] < thresholdMv) & (voltageMv[1:] >= thresholdMv)
    count = crossings.sum(axis=0)
    first = crossings.argmax(axis=0)
    columns = np.arange(voltageMv.shape[1])
    lower, upper = voltageMv[first, columns], voltageMv[first + 1, columns]
    fraction = np.divide(thresholdMv - lower, upper - lower, out=np.zeros(len(columns)), where=upper != lower)
    times = timeMs[first] + fraction * (timeMs[first + 1] - timeMs[first])
    times[count == 0] = np.nan
    return times, count


def apd90(timeMs, voltageMv, activationMs):
    """APD90 per node relative to the pre-beat voltage; NaN where repolarisation is incomplete."""
    result = np.full(voltageMv.shape[1], np.nan)
    for column in np.flatnonzero(np.isfinite(activationMs)):
        trace = voltageMv[:, column]
        peak = int(np.argmax(trace))
        threshold = trace[0] + 0.1 * (trace[peak] - trace[0])
        down = np.flatnonzero((trace[peak:-1] > threshold) & (trace[peak + 1:] <= threshold))
        if len(down):
            i = peak + down[0]
            fraction = (trace[i] - threshold) / (trace[i] - trace[i + 1])
            result[column] = timeMs[i] + fraction * (timeMs[i + 1] - timeMs[i]) - activationMs[column]
    return result


def activationPlane(positions, times):
    """Least-squares plane t = a.x + c through all depths: speed = 1/|a|, direction error from +x."""
    valid = np.isfinite(times)
    if valid.sum() < 4:
        return dict(speedMPerS=np.nan, directionErrorDeg=np.nan, planeResidual95Ms=np.nan)
    design = np.column_stack([positions[valid], np.ones(valid.sum())])
    coefficients = np.linalg.lstsq(design, times[valid], rcond=None)[0]
    residual = times[valid] - design @ coefficients
    gradient = coefficients[:3]                                       # ms / mm
    magnitude = np.linalg.norm(gradient)
    return dict(speedMPerS=float(1 / magnitude) if magnitude > 0 else np.nan,    # mm/ms = m/s
                directionErrorDeg=float(np.degrees(np.arccos(np.clip(gradient[0] / magnitude, -1, 1)))) if magnitude > 0 else np.nan,
                planeResidual95Ms=float(np.percentile(np.abs(residual), 95)))


def checkCondition(config, condition, job, tissue):
    """Per-beat summary table and named pass/fail criteria for one run."""
    settings = config["acceptance"]["physiology"]
    pace, spacing = config["pacing"], config["tissue"]["spacingMm"]
    boundary = config["lesion"]["boundaryYmm"]
    positions = tissue["tissuePositions"]
    distance = signedDistance(positions, boundary)
    region = ((positions[:, 0] >= settings["regionXmm"][0]) & (positions[:, 0] <= settings["regionXmm"][1])
              & (positions[:, 1] >= settings["viableRegionYmm"][0]) & (positions[:, 1] <= settings["viableRegionYmm"][1])
              & (distance < -spacing))
    if region.sum() < 4:
        raise ValueError("Physiology region holds fewer than 4 viable nodes; check acceptance.physiology")
    lesionCore = distance > 2 * spacing
    times = igbTimes(Path(job) / "vm.igb")
    response = settings["responseWindowMs"]
    rows, previous = [], None
    last = pace["beatCount"] - 1
    for beat in range(last - 2, last + 1):
        start = beat * pace["cycleLengthMs"]
        frames = slice(int(np.searchsorted(times, start)), int(np.searchsorted(times, start + pace["cycleLengthMs"])))
        local = times[frames] - start
        voltage = readIgb(Path(job) / "vm.igb", frames)
        activation, count = activationTimes(local, voltage, settings["activationThresholdMv"])
        roiTimes, roiVoltage = activation[region], voltage[:, region]
        amplitude = np.ptp(roiVoltage, axis=0)
        apd = apd90(local, roiVoltage, roiTimes)
        row = dict(condition=condition, beatNumber=beat + 1,
                   capturePassed=bool(np.all(count[region] == 1)),
                   responseInsideWindow=bool(np.isfinite(roiTimes).all() and np.all(roiTimes >= response[0])
                                             and np.all(roiTimes + apd <= response[1])),
                   apdComplete=bool(np.isfinite(apd).all()),
                   lesionCoreCrossings=int(count[lesionCore].sum()) if condition == "lesion" else 0,
                   medianApd90Ms=float(np.nanmedian(apd)) if np.isfinite(apd).any() else np.nan,
                   medianAmplitudeMv=float(np.median(amplitude)), **activationPlane(positions[region], roiTimes))
        if previous is not None:
            row.update(latencyChangeMs=float(np.nanmax(np.abs(roiTimes - previous[0]))),
                       amplitudeRelativeChange=float(np.max(np.abs(amplitude - previous[1]) / np.maximum(previous[1], 1e-12))),
                       apdChangeMs=float(np.nanmax(np.abs(apd - previous[2]))) if np.isfinite(apd - previous[2]).any() else np.nan)
        rows.append(row)
        previous = (roiTimes, amplitude, apd)
    table = pd.DataFrame(rows)
    compared = table.iloc[1:]
    lastSpeed = float(table.speedMPerS.iloc[-1])
    target = config["tissue"]["cvTargetMPerS"]
    criteria = {
        "oneActivationPerBeat": bool(table.capturePassed.all()),
        "responseInsideWindow": bool(table.responseInsideWindow.all()),
        "completeRepolarisation": bool(table.apdComplete.all()),
        "quietLesionCore": bool((table.lesionCoreCrossings == 0).all()),
        "planarWave": bool((table.planeResidual95Ms <= settings["maximumPlaneResidualMs"]).all()),
        "intendedDirection": bool((table.directionErrorDeg <= settings["maximumDirectionErrorDeg"]).all()),
        "stableActivation": bool((compared.latencyChangeMs <= settings["maximumBeatLatencyChangeMs"]).all()),
        "stableAmplitude": bool((compared.amplitudeRelativeChange <= settings["maximumBeatAmplitudeChange"]).all()),
        "stableApd": bool((compared.apdChangeMs <= settings["maximumBeatApdChangeMs"]).all()),
        "conductionVelocity": bool(abs(lastSpeed - target) / target <= settings["cvRelativeTolerance"]),
    }
    return table, criteria, dict(regionNodes=int(region.sum()), lastBeatSpeedMPerS=lastSpeed, targetMPerS=target)


def runPhysiology(config, jobs, tissues, folder):
    folder = Path(folder)
    tables, result = [], dict(conditions={})
    for condition, job in jobs.items():
        table, criteria, info = checkCondition(config, condition, job, tissues[condition])
        tables.append(table)
        result["conditions"][condition] = dict(passed=all(criteria.values()), criteria=criteria, **info)
    pd.concat(tables, ignore_index=True).to_csv(folder / "physiologyChecks.csv", index=False)
    result["passed"] = all(c["passed"] for c in result["conditions"].values())
    result["note"] = "Project engineering criteria; not independently validated physiological limits."
    saveJson(folder / "physiologyStatus.json", result)
    return result


def lastBeatActivation(config, job, tissue):
    """Earliest activation of every tissue node in the last beat (ms after that beat's stimulus)."""
    nodes, times = readActivations(Path(job) / "act-thresh.dat")
    beatStart = (config["pacing"]["beatCount"] - 1) * config["pacing"]["cycleLengthMs"]
    count = len(tissue["tissueNodes"])
    return firstActivation(nodes, times, count, beatStart, beatStart + config["pacing"]["cycleLengthMs"]) - beatStart


def deriveWindow(config, job, tissue):
    """Measurement window (ms after the last beat's stimulus) from intact top-surface activation under the grid.

    start = max(first - lead, stimulus end + minimum), end = last + trail; rounded outward to the output
    step. Decided before any electrogram is read (DL-2026-10-28-08).
    """
    rule, pace = config["analysis"]["window"], config["pacing"]
    positions = tissue["tissuePositions"]
    first = lastBeatActivation(config, job, tissue)
    under = np.isclose(positions[:, 2], positions[:, 2].max()) & (np.abs(positions[:, 0]) <= rule["gridHalfExtentXmm"])
    if not np.isfinite(first[under]).all():
        raise ValueError("Not every surface node under the grid activated in the last beat")
    earliest, latest = float(first[under].min()), float(first[under].max())
    step = config["numerics"]["sampleStepMs"]
    start = max(earliest - rule["leadMs"], pace["stimulusDurationMs"] + rule["minimumAfterStimulusMs"])
    end = latest + rule["trailMs"]
    start, end = np.floor(start / step + 1e-9) * step, np.ceil(end / step - 1e-9) * step
    if not 0 <= start < end < pace["cycleLengthMs"]:
        raise ValueError(f"Derived window {start}-{end} ms does not fit in one cycle")
    atFloor = bool(np.isclose(start, np.floor((pace["stimulusDurationMs"] + rule["minimumAfterStimulusMs"]) / step + 1e-9) * step))
    return dict(windowMs=[round(float(start), 6), round(float(end), 6)], gridFirstActivationMs=earliest,
                gridLastActivationMs=latest, nodesUsed=int(under.sum()), startsAtStimulusFloor=atFloor,
                note=("Window starts at the stimulus floor: far-field from the pacing strip may enter it (DL-2026-10-28-08)."
                      if atFloor else ""))
