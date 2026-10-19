"""Analyse the prespecified final beat at matched catheter placements.

Read analyseExperiment for the order of work. The notebook exposes these same
steps: calibrate once -> measure both conditions -> reconstruct contributions ->
check eligibility -> save summaries. On-disk NPZ keys retain the original names
so existing raw simulation exports can still be read.
"""

import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd
from boundary_setup import saveJson, igb_io
from boundary_geometry import recordingGeometry, signedDistance
import egm
import ground_truth as groundTruth


def analyseExperiment(runFolder, config, physiologyPassed):
    outputFolder = Path(runFolder) / "results"
    outputFolder.mkdir(exist_ok=True)
    gain = calibrateGain(runFolder, config)
    measurementTables, contributionTables = [], []
    for condition in ["intact", "lesion"]:
        measurements = processCondition(runFolder, condition, config, gain)
        measurementTables.append(measurements)
        contributionTables.append(
            computeContributions(runFolder, condition, config, measurements, gain, physiologyPassed)
        )
    return saveExperimentResults(
        runFolder, config, physiologyPassed, measurementTables, contributionTables
    )


def measurementSlices(timeMs, config):
    analysis = config["analysis"]
    beatStart = (config["pacing"]["beatCount"] - 1) * config["pacing"]["cycleLengthMs"]
    startTime, endTime = beatStart + np.asarray(analysis["beatWindowMs"])
    start = int(np.searchsorted(timeMs, startTime))
    stop = int(
        np.searchsorted(timeMs, endTime)
    )  # half-open: equal sample counts without duplicate beat endpoints
    history = int(np.searchsorted(timeMs, startTime - analysis["filterHistoryMs"]))
    if (
        startTime - analysis["filterHistoryMs"] < timeMs[0]
        or endTime > timeMs[-1]
        or stop - start < 2
    ):
        raise ValueError("Recording does not contain the requested beat and filter history")
    return slice(history, stop), slice(start - history, stop - history)


def filterElectrodes(signals, sampleRate, config):
    analysis = config["analysis"]
    filterSections = egm.makeFilter(
        analysis["differentialBandHz"], sampleRate, analysis["filterOrder"]
    )
    return egm.applyFilter(filterSections, signals)


def stableSeed(baseSeed, condition, repeatIndex):
    # Stable text hashing keeps the original noise streams independent of call order.
    identity = f"{baseSeed}:{condition}:{repeatIndex}".encode()
    return int.from_bytes(hashlib.sha256(identity).digest()[:8], "little")


def calibrateGain(runFolder, config):
    """One declared intact placement, one standoff, one active beat; then freeze gain."""
    with np.load(Path(runFolder) / "intact/egm.npz") as source:
        rawSlice, measureSlice = measurementSlices(source["t_ms"], config)
        sampleRate = 1000 / np.diff(source["t_ms"])[0]
        signals = source["uni"][..., rawSlice]
        filtered = filterElectrodes(signals, sampleRate, config)[..., measureSlice]
        selected = np.isclose(
            source["centre_xy_mm"][:, 1], config["analysis"]["referenceCentreYmm"]
        ) & np.isclose(source["p_gap_mm"], config["analysis"]["referenceGapMm"])
        if selected.sum() != 1:
            raise ValueError("Calibration must identify exactly one intact placement")
        recordings = egm.makeBipolarRecordings(filtered)
        referenceValues = np.concatenate(
            [np.ptp(record.traces[selected], axis=-1) for record in recordings]
        )
        reference = float(np.median(referenceValues))
        if not np.isfinite(reference) or reference <= 1e-8:
            raise ValueError("The intact calibration signal is absent or numerically negligible")
        gain = config["analysis"]["targetBipolarMv"] / reference
        saveJson(
            Path(runFolder) / "results/calibration.json",
            dict(
                gain=gain,
                rawReferenceMedianMv=reference,
                targetBipolarMv=config["analysis"]["targetBipolarMv"],
                condition="intact",
                centreYmm=config["analysis"]["referenceCentreYmm"],
                gapMm=config["analysis"]["referenceGapMm"],
                interpretation="Imposed amplitude calibration, not an independent physiological validation",
            ),
        )
        return gain


def processCondition(runFolder, condition, config, gain):
    """Save clean/noisy traces and exactly the weights used to select each recording."""
    outputFolder = Path(runFolder) / "results"
    source = dict(np.load(Path(runFolder) / condition / "egm.npz"))
    rawSlice, measureSlice = measurementSlices(source["t_ms"], config)
    measureTime = source["t_ms"][rawSlice][measureSlice]
    sampleRate = 1000 / np.diff(source["t_ms"])[0]
    cleanElectrodes = gain * source["uni"][..., rawSlice]
    cleanDifferential = filterElectrodes(cleanElectrodes, sampleRate, config)[..., measureSlice]
    cleanUnipolar = cleanElectrodes[..., measureSlice]
    checkFilterHistory(
        source,
        config,
        gain,
        rawSlice,
        measureSlice,
        cleanDifferential,
        outputFolder / f"{condition}FilterCheck.json",
    )
    rows, traces, cleanAtSelectedWeights, noiseTraces = [], [], [], []
    for repeatIndex in range(config["analysis"]["noiseRepeats"] + 1):
        randomGenerator = np.random.default_rng(
            stableSeed(config["analysis"]["noiseSeed"], condition, repeatIndex)
        )
        noise = np.zeros_like(cleanElectrodes)
        if repeatIndex > 0:
            noise = randomGenerator.normal(
                0, config["analysis"]["noiseSdMv"], cleanElectrodes.shape
            )
        unipolar = (cleanElectrodes + noise)[..., measureSlice]
        differential = filterElectrodes(cleanElectrodes + noise, sampleRate, config)[
            ..., measureSlice
        ]
        recordings = egm.makeRecordings(
            unipolar, differential, config["analysis"]["omniAngleStepDeg"]
        )
        declaredLabels = config["analysis"].get("recordingLabels")
        if declaredLabels:
            recordings = [record for record in recordings
                          if record.label in declaredLabels[record.kind]]
        for recording in recordings:
            kind, label = recording.kind, recording.label
            trace, weights = recording.traces, recording.weights
            cleanSignals = cleanUnipolar if kind == "uni" else cleanDifferential
            # Apply the measured recording's weights to the noise-free signals.
            # This preserves noisy pair/direction choices in the comparison.
            cleanTrace = np.einsum("pe,pet->pt", weights, cleanSignals)
            for placementIndex in range(len(trace)):
                geometry = recordingGeometry(
                    source["uni_xy"][placementIndex],
                    source["uni_sample_xyz"][placementIndex],
                    kind,
                    label,
                    weights[placementIndex],
                    config["boundary"],
                    condition,
                )
                traceIndex = len(traces)
                row = dict(
                    recordingId=f"{condition}:{repeatIndex}:{placementIndex}:{kind}:{label}",
                    condition=condition,
                    noiseRepeat=repeatIndex,
                    placement=placementIndex,
                    kind=kind,
                    label=label,
                    gapMm=float(source["p_gap_mm"][placementIndex]),
                    centreYmm=float(source["centre_xy_mm"][placementIndex, 1]),
                    ppMv=float(np.ptp(trace[placementIndex])),
                    cleanAtSelectedWeightsPpMv=float(np.ptp(cleanTrace[placementIndex])),
                    windowStartMs=float(measureTime[0]),
                    windowEndExclusiveMs=float(measureTime[-1] + 1000 / sampleRate),
                    traceIndex=traceIndex,
                    noiseAtSelectedWeightsPpMv=float(
                        np.ptp(trace[placementIndex] - cleanTrace[placementIndex])
                    ),
                    cleanSignalAboveThreshold=bool(
                        np.ptp(cleanTrace[placementIndex]) >= config["analysis"]["minimumCleanPpMv"]
                    ),
                    **geometry,
                )
                row.update(
                    {
                        f"weightE{index}": float(weight)
                        for index, weight in enumerate(weights[placementIndex])
                    }
                )
                rows.append(row)
                traces.append(trace[placementIndex])
                cleanAtSelectedWeights.append(cleanTrace[placementIndex])
                noiseTraces.append(trace[placementIndex] - cleanTrace[placementIndex])
    table = pd.DataFrame(rows)
    table.to_csv(outputFolder / f"{condition}Points.csv", index=False)
    np.savez_compressed(
        outputFolder / f"{condition}Traces.npz",
        timeMs=measureTime,
        measuredMv=np.asarray(traces, dtype=np.float32),
        cleanAtSelectedWeightsMv=np.asarray(cleanAtSelectedWeights, dtype=np.float32),
        noiseAtSelectedWeightsMv=np.asarray(noiseTraces, dtype=np.float32),
    )
    saveElectrodePositions(source, config, condition, outputFolder)
    return table


def checkFilterHistory(source, config, gain, rawSlice, measureSlice, cleanDifferential, outputPath):
    """Compare the same beat with the declared history and twice that history."""
    measureTime = source["t_ms"][rawSlice][measureSlice]
    sampleRate = 1000 / np.diff(source["t_ms"])[0]
    # Double the history to check that causal filter initialisation no longer matters.
    measurementStart = source["t_ms"][rawSlice][measureSlice][0]
    doubleHistoryAvailable = (
        measurementStart - 2 * config["analysis"]["filterHistoryMs"] >= source["t_ms"][0]
    )
    longerStart = int(
        np.searchsorted(
            source["t_ms"], measurementStart - 2 * config["analysis"]["filterHistoryMs"]
        )
    )
    longerSignals = gain * source["uni"][..., longerStart : rawSlice.stop]
    longerFiltered = filterElectrodes(longerSignals, sampleRate, config)[..., -len(measureTime) :]
    referenceRange = np.max(np.ptp(cleanDifferential, axis=-1))
    historyError = float(
        np.max(np.abs(cleanDifferential - longerFiltered)) / max(referenceRange, 1e-12)
    )
    saveJson(
        outputPath,
        dict(
            relativeError=historyError,
            extraHistoryAvailable=bool(doubleHistoryAvailable and longerStart < rawSlice.start),
            passed=bool(
                doubleHistoryAvailable
                and longerStart < rawSlice.start
                and historyError <= config["analysis"]["maximumFilterHistoryError"]
            ),
        ),
    )


def saveElectrodePositions(source, config, condition, outputFolder):
    """Export every electrode centre separately from the derived recording anchors."""
    electrodeRows = []
    for placementIndex, positions in enumerate(source["uni_xy"]):
        distances = signedDistance(positions, config["boundary"])
        for electrodeIndex, (position, distance) in enumerate(zip(positions, distances)):
            tissue = (
                "viable"
                if condition == "intact" or distance < -1e-9
                else "lesion" if distance > 1e-9 else "boundary"
            )
            electrodeRows.append(
                dict(
                    condition=condition,
                    placement=placementIndex,
                    electrode=electrodeIndex,
                    xMm=position[0],
                    yMm=position[1],
                    zMm=position[2],
                    distanceInsideMm=distance,
                    tissueBelowCentre=tissue,
                )
            )
    pd.DataFrame(electrodeRows).to_csv(outputFolder / f"{condition}Electrodes.csv", index=False)


def computeContributions(runFolder, condition, config, measurements, gain, physiologyPassed):
    """Reconstruct clean primary recordings; eligibility remains separate from diagnostics."""
    outputFolder = Path(runFolder) / "results"
    source = dict(np.load(Path(runFolder) / condition / "egm.npz"))
    rawSlice, measureSlice = measurementSlices(source["t_ms"], config)
    sampleRate = 1000 / np.diff(source["t_ms"])[0]
    geometry = groundTruth.readElementGeometry(runFolder, condition)
    selectedRows = measurements[
        (measurements.noiseRepeat == 0)
        & measurements.centreYmm.isin(config["analysis"]["groundTruthCentreYmm"])
    ]
    rows, curveIds, radiusCurves, reconstructionCurves = [], [], [], []
    fullReconstructions, recoveredTraces = [], []
    with np.load(outputFolder / f"{condition}Traces.npz") as traceArchive:
        measuredTraces = traceArchive["cleanAtSelectedWeightsMv"]
    for mode in ["uni", "differential"]:
        modeRows = selectedRows[selectedRows.kind == "uni"] if mode == "uni" else selectedRows[selectedRows.kind != "uni"]
        if modeRows.empty:
            continue
        band = None if mode == "uni" else config["analysis"]["differentialBandHz"]
        voltage = groundTruth.readFilteredVoltage(
            Path(runFolder) / condition, rawSlice, band, sampleRate,
            config["analysis"]["filterOrder"],
        )
        selectedVoltage = voltage[:, measureSlice]
        if len(voltage) != len(geometry.nodePositions):
            raise ValueError("Vm node count differs from contribution mesh")
        for record in modeRows.to_dict("records"):
            print(f"Contributions: {record['recordingId']}", flush=True)
            weights = np.array([record[f"weightE{index}"] for index in range(16)])
            anchor = np.array([record["anchorXmm"], record["anchorYmm"], record["anchorZmm"]])
            fullTrace, radius, radii, curve = groundTruth.reconstructRecording(
                geometry, selectedVoltage, source["uni_sample_xyz"][record["placement"]],
                weights, anchor, config["analysis"].get("contributionRingStepMm", config["meshSpacingMm"]),
            )
            recovered = measuredTraces[record["traceIndex"]].astype(float) / gain
            agreement = waveformAgreement(fullTrace, recovered)
            cleanSignalUsable = bool(
                record["cleanAtSelectedWeightsPpMv"] >= config["analysis"]["minimumCleanPpMv"]
                and np.isfinite(fullTrace).all()
                and np.linalg.norm(fullTrace - fullTrace.mean()) > 1e-12
            )
            edgeDistance = float(np.min(np.asarray(config["slabSizeMm"][:2]) / 2 - np.abs(anchor[:2])))
            rows.append(dict(
                recordingId=record["recordingId"], condition=condition, kind=record["kind"],
                rawFov80ReconstructionMm=radius,
                physiologyPassed=bool(physiologyPassed), cleanSignalUsable=cleanSignalUsable,
                signalEligible=bool(physiologyPassed and cleanSignalUsable),
                outerEdgeDistanceMm=edgeDistance,
                radiusReachesOuterEdge=bool(np.isfinite(radius) and radius >= edgeDistance),
                **agreement,
            ))
            curveIds.append(record["recordingId"])
            radiusCurves.append(radii)
            reconstructionCurves.append(curve)
            fullReconstructions.append(fullTrace)
            recoveredTraces.append(recovered)
        del selectedVoltage, voltage
    maximumBins = max(map(len, radiusCurves), default=0)
    padded = []
    for curves in [radiusCurves, reconstructionCurves]:
        values = np.full((len(curves), maximumBins), np.nan)
        for index, curve in enumerate(curves):
            values[index, :len(curve)] = curve
        padded.append(values)
    np.savez_compressed(
        outputFolder / f"{condition}ContributionCurves.npz",
        recordingId=np.asarray(curveIds), radiusMm=padded[0], reconstructionFraction=padded[1],
        timeMs=source["t_ms"][rawSlice][measureSlice],
        reconstructedRaw=np.asarray(fullReconstructions), recoveredRaw=np.asarray(recoveredTraces),
    )
    return pd.DataFrame(rows)


def waveformAgreement(reconstructed, recovered):
    """Remove constant offsets only for waveform comparison; FOV retains the original definition."""
    reconstructed = reconstructed - reconstructed.mean()
    recovered = recovered - recovered.mean()
    denominator = float(reconstructed @ reconstructed)
    referenceEnergy = float(recovered @ recovered)
    if denominator <= 1e-24 or referenceEnergy <= 1e-24:
        return dict(signedCorrelation=np.nan, fittedScale=np.nan, shapeError=np.nan)
    scale = float(reconstructed @ recovered / denominator)
    return dict(
        signedCorrelation=float(np.corrcoef(reconstructed, recovered)[0, 1]),
        fittedScale=scale,
        shapeError=float(
            np.linalg.norm(scale * reconstructed - recovered) / np.sqrt(referenceEnergy)
        ),
    )


def qualifyContributions(contributions, config):
    referenceRows = contributions[
        (contributions.condition == "intact")
        & (contributions.kind == "uni")
        & contributions.cleanSignalUsable
        & (contributions.shapeError <= config["analysis"]["maximumShapeError"])
        & (contributions.signedCorrelation.abs() >= config["analysis"]["minimumAbsCorrelation"])
    ]
    scale = float(referenceRows.fittedScale.median()) if len(referenceRows) else np.nan
    # One signed scale from intact unipoles, never a separately fitted scale for each reported recording.
    contributions["globalSignedScale"] = scale
    contributions["globalScaleError"] = (
        np.abs(contributions.fittedScale / scale - 1)
        if np.isfinite(scale) and scale != 0
        else np.nan
    )
    contributions["fovEligible"] = (
        contributions.signalEligible
        & (contributions.signedCorrelation.abs() >= config["analysis"]["minimumAbsCorrelation"])
        & (contributions.shapeError <= config["analysis"]["maximumShapeError"])
        & (contributions.globalScaleError <= config["analysis"]["maximumGlobalScaleError"])
        & np.isfinite(contributions.rawFov80ReconstructionMm)
    )
    contributions["fov80ReconstructionMm"] = contributions.rawFov80ReconstructionMm.where(
        contributions.fovEligible
    )
    contributions["eligibilityReason"] = np.select(
        [~contributions.physiologyPassed, ~contributions.cleanSignalUsable,
         ~np.isfinite(contributions.globalSignedScale), ~contributions.fovEligible],
        ["physiologyFailed", "negligibleOrInvalidSignal", "noValidIntactScale", "recoveryMismatch"],
        default="eligiblePilotEstimate",
    )
    return contributions


def boundarySummary(measurements, config):
    """Descriptive plateau/contrast checks; overlapping rows are not independent replicates."""
    rows = []
    clean = measurements[measurements.noiseRepeat == 0]
    for (condition, kind, label, gap), group in clean.groupby(["condition", "kind", "label", "gapMm"]):
        far = group[group.distanceViableMm >= config["analysis"]["farRegionMm"]]
        # Collapse repeated recording positions before comparing spatial profiles.
        profile = far.groupby("distanceViableMm").ppMv.median()
        median = float(profile.median()) if len(profile) else np.nan
        relativeRange = float((profile.max() - profile.min()) / median) if median > 0 else np.nan
        lesionSide = group[
            group.distanceViableMm <= -config["analysis"]["farRegionMm"]
        ].ppMv.median()
        contrast = float((median - lesionSide) / median) if median > 0 else np.nan
        rows.append(
            dict(
                condition=condition,
                kind=kind,
                label=label,
                gapMm=gap,
                farPositions=len(profile),
                farMedianMv=median,
                farRelativeRange=relativeRange,
                farStable=bool(
                    len(profile) >= 3
                    and relativeRange <= config["analysis"]["maximumFarRelativeRange"]
                ),
                farStatus=("insufficientPositions" if len(profile) < 3 else
                           "passed" if relativeRange <= config["analysis"]["maximumFarRelativeRange"]
                           else "excessiveVariation"),
                contrastFraction=contrast,
                transitionObserved=bool(
                    condition == "lesion"
                    and contrast >= config["analysis"]["minimumTransitionFraction"]
                ),
            )
        )
    return pd.DataFrame(rows)


def saveExperimentResults(
    runFolder, config, physiologyPassed, measurementTables, contributionTables
):
    """Join measurements and contribution checks, then write the pilot decision."""
    outputFolder = Path(runFolder) / "results"
    measurements = pd.concat(measurementTables, ignore_index=True)
    contributions = qualifyContributions(pd.concat(contributionTables, ignore_index=True), config)
    contributions.to_csv(outputFolder / "contributions.csv", index=False)
    combined = measurements.merge(
        contributions.drop(columns=["condition", "kind"]),
        on="recordingId",
        how="left",
        validate="one_to_one",
    )
    combined["groundTruthComputed"] = combined.recordingId.isin(contributions.recordingId)
    combined.to_csv(outputFolder / "points.csv", index=False)
    noisyMeasurements = measurements[measurements.noiseRepeat > 0]
    noisyMeasurements.groupby(["condition", "placement", "kind", "label", "gapMm"]).ppMv.agg(
        noiseRepeatCount="count",
        meanPpMv="mean",
        sdPpMv="std",
        minimumPpMv="min",
        maximumPpMv="max",
    ).to_csv(outputFolder / "noiseSummary.csv")
    summary = boundarySummary(measurements, config)
    summary.to_csv(outputFolder / "boundaryChecks.csv", index=False)
    shapeGroups = (
        contributions.groupby(["condition", "kind"])
        .fovEligible.mean()
    )
    filterPassed = all(
        json.loads((outputFolder / f"{condition}FilterCheck.json").read_text())["passed"]
        for condition in ["intact", "lesion"]
    )
    criteria = {
        "physiology": bool(physiologyPassed),
        "filterHistory": bool(filterPassed),
        "stableFarRegion": bool(summary.farStable.all()),
        "lesionTransition": bool(summary[summary.condition == "lesion"].transitionObserved.all()),
        "validContributions": bool(len(shapeGroups) == 6 and (shapeGroups >= 0.9).all()),
    }
    passed = all(criteria.values())
    saveJson(
        outputFolder / "pilotStatus.json",
        dict(
            pipelineCompleted=True,
            pilotAcceptancePassed=passed,
            criteria=criteria,
            physiologyPassed=bool(physiologyPassed),
            filterHistoryPassed=filterPassed,
            eligibleContributionCount=int(contributions.fovEligible.sum()),
            contributionCount=len(contributions),
            noiseReplicatesAreBiologicalReplicates=False,
            noiseReplicateContributionsComputed=False,
            meshConvergenceVerified=False,
            domainConvergenceVerified=False,
            interpretation="Pilot estimates only. Inspect checks before interpreting spatial sensing; convergence is a separate experiment.",
        ),
    )
    from boundary_clinical import analyseChangePoints
    analyseChangePoints(runFolder, config, measurements, pilotPassed=passed)
    return combined, passed
