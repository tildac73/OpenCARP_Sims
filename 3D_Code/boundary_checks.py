"""Physiological checks, independent of the electrogram noise model."""

from pathlib import Path
import numpy as np
import pandas as pd
from boundary_setup import readNodePositions, saveJson, igb_io
from boundary_geometry import signedDistance


def activationTimes(timeMs, voltageMv, thresholdMv):
    """First interpolated upward crossing and the crossing count for each node."""
    crossings = (voltageMv[:-1] < thresholdMv) & (voltageMv[1:] >= thresholdMv)
    count = crossings.sum(axis=0)
    first = crossings.argmax(axis=0)
    columns = np.arange(voltageMv.shape[1])
    lower = voltageMv[first, columns]
    upper = voltageMv[first + 1, columns]
    fraction = np.divide(
        thresholdMv - lower, upper - lower, out=np.zeros(len(columns)), where=upper != lower
    )
    times = timeMs[first] + fraction * (timeMs[first + 1] - timeMs[first])
    times[count == 0] = np.nan
    return times, count


def actionPotentialDurations(timeMs, voltageMv, times):
    """APD90 relative to the pre-beat voltage; unfinished repolarisation is NaN."""
    result = np.full(voltageMv.shape[1], np.nan)
    for column in range(voltageMv.shape[1]):
        if not np.isfinite(times[column]):
            continue
        trace = voltageMv[:, column]
        peakIndex = int(np.argmax(trace))
        threshold = trace[0] + 0.1 * (trace[peakIndex] - trace[0])
        down = np.flatnonzero(
            (trace[peakIndex:-1] > threshold) & (trace[peakIndex + 1 :] <= threshold)
        )
        if len(down):
            index = peakIndex + down[0]
            fraction = (trace[index] - threshold) / (trace[index] - trace[index + 1])
            result[column] = (
                timeMs[index] + fraction * (timeMs[index + 1] - timeMs[index]) - times[column]
            )
    return result


def fitActivationPlane(positions, times):
    """Fit activation time = ax*x + ay*y + az*z + offset.

    The time gradient points along propagation; its inverse magnitude is speed.
    Fit all depths, not just the surface, and compare direction with intended +x.
    """
    valid = np.isfinite(times)
    if valid.sum() < 4:
        raise ValueError("Too few activated nodes to fit an activation plane")
    design = np.column_stack([positions[valid], np.ones(valid.sum())])
    coefficients = np.linalg.lstsq(design, times[valid], rcond=None)[0]
    residuals = times[valid] - design @ coefficients
    gradient = coefficients[:3]
    gradientMagnitude = np.linalg.norm(gradient)
    speed = np.nan
    directionError = np.nan
    if gradientMagnitude > 0:
        speed = 1 / gradientMagnitude
        directionCosine = np.clip(gradient[0] / gradientMagnitude, -1, 1)
        directionError = np.degrees(np.arccos(directionCosine))
    return dict(
        speedMmPerMs=float(speed),
        directionErrorDeg=float(directionError),
        planeResidual95Ms=float(np.percentile(np.abs(residuals), 95)),
    )


def checkCondition(runFolder, condition, config):
    simulationFolder = Path(runFolder) / condition
    meshBase = Path(runFolder) / "mesh" / condition
    positions = readNodePositions(meshBase)
    timeMs = igb_io.timeAxis(simulationFolder / "vm.igb")
    settings = config["acceptance"]
    cycleLength = config["pacing"]["cycleLengthMs"]
    finalBeat = config["pacing"]["beatCount"] - 1
    roiMask = (
        (positions[:, 0] >= settings["regionXmm"][0])
        & (positions[:, 0] <= settings["regionXmm"][1])
        & (positions[:, 1] >= settings["viableRegionYmm"][0])
        & (positions[:, 1] <= settings["viableRegionYmm"][1])
    )
    if condition == "lesion":
        roiMask &= signedDistance(positions, config["boundary"]) < -config["meshSpacingMm"]
    if roiMask.sum() < 4:
        raise ValueError("No usable viable validation region")
    summaries, maps = [], []
    previousTimes, previousAmplitude, previousApd = None, None, None
    windowEnd = config["analysis"]["beatWindowMs"][1]
    for beatIndex in range(finalBeat - 2, finalBeat + 1):
        beatStart = beatIndex * cycleLength
        frameStart = int(np.searchsorted(timeMs, beatStart))
        frameEnd = int(np.searchsorted(timeMs, beatStart + cycleLength, side="right"))
        localTime = timeMs[frameStart:frameEnd] - beatStart
        voltage = igb_io.read(simulationFolder / "vm.igb", slice(frameStart, frameEnd))
        times, counts = activationTimes(localTime, voltage, settings["activationThresholdMv"])
        roiTimes = times[roiMask]
        roiVoltage = voltage[:, roiMask]
        amplitudes = np.ptp(roiVoltage, axis=0)
        apd = actionPotentialDurations(localTime, roiVoltage, roiTimes)
        capturePassed = bool(np.all(counts[roiMask] == 1))
        plane = dict(speedMmPerMs=np.nan, directionErrorDeg=np.nan, planeResidual95Ms=np.nan)
        if np.isfinite(roiTimes).sum() >= 4:
            plane = fitActivationPlane(positions[roiMask], roiTimes)

        # The first of three beats supplies the baseline; compare the next two.
        latencyChange = np.nan
        amplitudeChange = np.nan
        apdChange = np.nan
        if previousTimes is not None:
            if np.isfinite(roiTimes - previousTimes).any():
                latencyChange = float(np.nanmax(np.abs(roiTimes - previousTimes)))
            amplitudeChange = float(
                np.max(
                    np.abs(amplitudes - previousAmplitude) / np.maximum(previousAmplitude, 1e-12)
                )
            )
            if np.isfinite(apd - previousApd).any():
                apdChange = float(np.nanmax(np.abs(apd - previousApd)))
        lesionCore = signedDistance(positions, config["boundary"]) > 2 * config["meshSpacingMm"]
        lesionCrossings = int(counts[lesionCore].sum()) if condition == "lesion" else 0
        covered = bool(
            np.isfinite(roiTimes).all()
            and np.all(roiTimes >= config["analysis"]["beatWindowMs"][0])
            and np.all(roiTimes + apd <= windowEnd)
        )
        summaries.append(
            dict(
                condition=condition,
                beatNumber=beatIndex + 1,
                capturePassed=capturePassed,
                activatedFraction=float(np.mean(counts[roiMask] == 1)),
                measurementCoversResponse=covered,
                lesionCoreCrossings=lesionCrossings,
                latencyChangeMs=latencyChange,
                amplitudeRelativeChange=amplitudeChange,
                apdChangeMs=apdChange,
                apdComplete=bool(np.isfinite(apd).all()),
                **plane
            )
        )
        maps.append(times)
        previousTimes, previousAmplitude, previousApd = roiTimes, amplitudes, apd
    table = pd.DataFrame(summaries)
    criteria = physiologyCriteria(table, settings)
    passed = all(criteria.values())
    np.savez_compressed(
        simulationFolder / "activationChecks.npz",
        positionsMm=positions,
        activationMs=np.asarray(maps),
        regionMask=roiMask,
    )
    return table, passed


def checkExperiment(runFolder, config):
    tables, accepted, criteria = [], {}, {}
    for condition in ["intact", "lesion"]:
        table, passed = checkCondition(runFolder, condition, config)
        tables.append(table)
        accepted[condition] = passed
        criteria[condition] = physiologyCriteria(table, config["acceptance"])
    pd.concat(tables, ignore_index=True).to_csv(
        Path(runFolder) / "physiologyChecks.csv", index=False
    )
    saveJson(
        Path(runFolder) / "physiologyStatus.json",
        dict(
            passed=all(accepted.values()),
            conditions=accepted,
            criteria=criteria,
            toleranceStatus="Provisional pilot engineering criteria; not independently validated physiological limits",
        ),
    )
    return all(accepted.values())


def physiologyCriteria(table, settings):
    """Named criteria make a failed physiological check easier to understand."""
    comparedBeats = table.iloc[1:]
    return {
        "oneActivationPerBeat": bool(table.capturePassed.all()),
        "responseInsideWindow": bool(table.measurementCoversResponse.all()),
        "completeRepolarisation": bool(table.apdComplete.all()),
        "quietLesionCore": bool((table.lesionCoreCrossings == 0).all()),
        "planarWave": bool((table.planeResidual95Ms <= settings["maximumPlaneResidualMs"]).all()),
        "intendedDirection": bool(
            (table.directionErrorDeg <= settings["maximumDirectionErrorDeg"]).all()
        ),
        "stableActivation": bool(
            (comparedBeats.latencyChangeMs <= settings["maximumBeatLatencyChangeMs"]).all()
        ),
        "stableAmplitude": bool(
            (comparedBeats.amplitudeRelativeChange <= settings["maximumBeatAmplitudeChange"]).all()
        ),
        "stableApd": bool((comparedBeats.apdChangeMs <= settings["maximumBeatApdChangeMs"]).all()),
    }
