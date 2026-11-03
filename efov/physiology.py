"""Physiology checks of each tissue run, and the measurement window derived from activation.

Checks on the last three beats, on catheter-facing tissue nodes in a viable region away from the
stimulus, edges and boundary (acceptance.physiology): one activation per beat, complete
repolarisation inside the response window, quiet lesion core, a planar wave in the intended
direction (wave angle, FM-17), and beat-to-beat stability of latency, amplitude and APD90. The
surface CV is reported, not gated: the CV gate is the tune stage (FM-18), because fibre models other
than uniform change the slab CV, as they physically should. These are project engineering
criteria, not validated biological limits.

Activation time = first linearly interpolated upward crossing of the threshold in Vm sampled at
the output step. APD90 = time from activation to the first downward crossing of 10 % of the
amplitude above the pre-beat voltage.
"""
from pathlib import Path

import numpy as np
import pandas as pd

from .config import beatCount, cycleLengthMs, sampleStepMs, spacingMm, stimulusDurationMs
from .geometry import signedDistance
from .io import firstActivation, igbTimes, readActivations, readIgb, saveJson
from .meshbuild import waveDirection


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


def activationPlane(positions, times, direction):
    """Least-squares plane t = a.(x, y) + c through surface nodes: speed = 1/|a|, angle error from direction."""
    valid = np.isfinite(times)
    if valid.sum() < 4:
        return dict(speedMPerS=np.nan, directionErrorDeg=np.nan, planeResidual95Ms=np.nan)
    design = np.column_stack([positions[valid, :2], np.ones(valid.sum())])
    coefficients = np.linalg.lstsq(design, times[valid], rcond=None)[0]
    residual = times[valid] - design @ coefficients
    gradient = coefficients[:2]                                       # ms / mm
    magnitude = np.linalg.norm(gradient)
    cosine = float(gradient @ direction / magnitude) if magnitude > 0 else np.nan
    return dict(speedMPerS=float(1 / magnitude) if magnitude > 0 else np.nan,    # mm/ms = m/s
                directionErrorDeg=float(np.degrees(np.arccos(np.clip(cosine, -1, 1)))) if magnitude > 0 else np.nan,
                planeResidual95Ms=float(np.percentile(np.abs(residual), 95)))


def checkCondition(config, condition, job, tissue):
    """Per-beat summary table and named pass/fail criteria for one run."""
    settings = config["fixed"]["acceptance"]["physiology"]
    cycle, spacing = cycleLengthMs(config), spacingMm(config)[0]
    boundary = config["fixed"]["tissue"]["boundaryYmm"]
    positions = tissue["tissuePositions"]
    distance = signedDistance(positions, boundary)
    surface = np.isclose(positions[:, 2], positions[:, 2].max())
    region = (surface & (positions[:, 0] >= settings["regionXmm"][0]) & (positions[:, 0] <= settings["regionXmm"][1])
              & (positions[:, 1] >= settings["viableRegionYmm"][0]) & (positions[:, 1] <= settings["viableRegionYmm"][1])
              & (distance < -spacing))
    direction = waveDirection(config["values"]["waveAngleDeg"])
    if region.sum() < 4:
        raise ValueError("Physiology region holds fewer than 4 viable nodes; check acceptance.physiology")
    lesionCore = distance > 2 * spacing
    times = igbTimes(Path(job) / "vm.igb")
    response = settings["responseWindowMs"]
    rows, previous = [], None
    last = beatCount(config) - 1
    for beat in range(last - 2, last + 1):
        start = beat * cycle
        frames = slice(int(np.searchsorted(times, start)), int(np.searchsorted(times, start + cycle)))
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
                   medianAmplitudeMv=float(np.median(amplitude)), **activationPlane(positions[region], roiTimes, direction))
        if previous is not None:
            row.update(latencyChangeMs=float(np.nanmax(np.abs(roiTimes - previous[0]))),
                       amplitudeRelativeChange=float(np.max(np.abs(amplitude - previous[1]) / np.maximum(previous[1], 1e-12))),
                       apdChangeMs=float(np.nanmax(np.abs(apd - previous[2]))) if np.isfinite(apd - previous[2]).any() else np.nan)
        rows.append(row)
        previous = (roiTimes, amplitude, apd)
    table = pd.DataFrame(rows)
    compared = table.iloc[1:]
    lastSpeed = float(table.speedMPerS.iloc[-1])
    target = config["values"]["cvTargetMPerS"]
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
    }
    return table, criteria, dict(regionNodes=int(region.sum()), lastBeatSurfaceSpeedMPerS=lastSpeed, stripTargetMPerS=target,
                                 note="Surface CV reported, not gated; the CV gate is the tune stage (FM-18).")


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
    beatStart = (beatCount(config) - 1) * cycleLengthMs(config)
    count = len(tissue["tissueNodes"])
    return firstActivation(nodes, times, count, beatStart, beatStart + cycleLengthMs(config)) - beatStart


def commonStepMs(*steps):
    """Smallest time step that is a whole multiple of every given step (ms), e.g. 0.2 and 0.5 -> 1.0."""
    from fractions import Fraction
    from math import lcm
    fractions = [Fraction(step).limit_denominator(10000) for step in steps]
    denominator = lcm(*(f.denominator for f in fractions))
    return float(Fraction(lcm(*(int(f * denominator) for f in fractions)), denominator))


def deriveWindow(config, job, tissue):
    """Measurement window (ms after the last beat's stimulus) from intact top-surface activation under the grid.

    start = max(first - lead, stimulus end + minimum), end = last + trail; rounded outward to the output
    step. Decided before any electrogram is read (DL-2026-10-28-08).
    """
    rule = config["fixed"]["analysis"]["window"]
    v = config["values"]
    halfExtent = abs(config["fixed"]["electrodes"]["centreXmm"]) + 1.5 * v["gridSpacingMm"] + v["electrodeLengthMm"] / 2
    positions = tissue["tissuePositions"]
    first = lastBeatActivation(config, job, tissue)
    under = np.isclose(positions[:, 2], positions[:, 2].max()) & (np.abs(positions[:, 0]) <= halfExtent)
    if not np.isfinite(first[under]).all():
        raise ValueError("Not every surface node under the grid activated in the last beat")
    earliest, latest = float(first[under].min()), float(first[under].max())
    cycle, stimulus = cycleLengthMs(config), stimulusDurationMs(config)
    step = commonStepMs(sampleStepMs(config), 1000.0 / config["fixed"]["analysis"]["acquisitionRateHz"])
    start = max(earliest - rule["leadMs"], stimulus + rule["minimumAfterStimulusMs"])
    end = latest + rule["trailMs"]
    # Rounded outward to a step on both the output grid and the acquisition grid (FM-22), so every
    # window boundary is a sample of the openCARP output and of the resampled electrograms.
    start, end = np.floor(start / step + 1e-9) * step, np.ceil(end / step - 1e-9) * step
    if not 0 <= start < end < cycle:
        raise ValueError(f"Derived window {start}-{end} ms does not fit in one cycle")
    atFloor = bool(np.isclose(start, np.floor((stimulus + rule["minimumAfterStimulusMs"]) / step + 1e-9) * step))
    return dict(windowMs=[round(float(start), 6), round(float(end), 6)], gridHalfExtentXmm=halfExtent, gridFirstActivationMs=earliest,
                gridLastActivationMs=latest, nodesUsed=int(under.sum()), startsAtStimulusFloor=atFloor,
                note=("Window starts at the stimulus floor: far-field from the pacing strip may enter it (DL-2026-10-28-08)."
                      if atFloor else ""))
