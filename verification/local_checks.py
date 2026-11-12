#!/usr/bin/env python3
"""Methodology verification, part 1: checks that need no openCARP (numpy and scipy only).

    python3 verification/local_checks.py --out runs/verification/local [--only NAME ...]

Each check backs one statement of the thesis Methodology with a number computed from the production
configuration (config/model.json, config/model.par) and the production analysis code (efov.measurements,
efov.peakfrequency, efov.geometry). Nothing is hard-coded that the configuration holds: if a setting
changes, rerunning this file updates every dependent number.

Checks
  conductivities   resistivity-to-conductivity conversion of Roberts and Scher (1982), anisotropy ratios,
                   monodomain (harmonic-mean) conductivities and the continuum prediction of the
                   transverse / longitudinal conduction-velocity ratio; bath-loading bound
  geometry         mesh size, layers through the wall, catheter travel, margins between the analysed
                   recordings and the slab edges, distinct electrodes, recordings per condition
  electrodes       how many mesh nodes represent one electrode at each gap and radius, and the
                   discrete electrode volume against the cylinder it stands for
  timing           expected activation and window timing of the baseline wave from the geometry
                   (predictions; the production run reports the measured values)
  resampling       pass-band error and alias rejection of the 5 kHz -> 2 kHz polyphase resampler
  bandpass         realised order, -3 dB corners, group delay, settling time and noise gain of the
                   30-500 Hz causal Butterworth filter; adequacy of the 200 ms filter history
  noise            band-passed bipolar noise level and noise-only peak-to-peak for each configured
                   noise SD, against the 0.01 mV exclusion limit and the 0.04 mV peak-frequency gate
  wavelet          peak-frequency recovery for tones and short bursts; window-end bias against the
                   length of trailing signal
The numbers are numerical properties of the analysis chain. None is a physiological validation.
"""
import argparse
import sys
from fractions import Fraction
from pathlib import Path

import numpy as np
from scipy import signal

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import REPOSITORY, cfg, productionConfig, runChecks  # noqa: E402

from efov import geometry, measurements, peakfrequency  # noqa: E402
from efov.recordings import allLabels  # noqa: E402

# Roberts and Scher 1982 (Circ Res 50:342-351), best-fit bidomain resistivities of canine left
# ventricle in situ, ohm cm, as recorded in docs/parameter_sources.md (read from the primary paper).
ROBERTS_SCHER_RESISTIVITY_OHM_CM = dict(g_il=291.0, g_it=1677.0, g_el=852.0, g_et=1247.0)
# Wang et al. region of interest relative to the scar boundary (Supplemental Methods 2): 10 mm into
# healthy myocardium, 15 mm into the scar. Positive = inside the lesion, as in efov.geometry.
WANG_ROI_MM = (-10.0, 15.0)
# Largest effective field of view reported by Wang et al. (unipolar, roof), mm.
WANG_LARGEST_EFOV_MM = 13.2


# ----------------------------------------------------------------- conductivities
def checkConductivities(config):
    g = {name: cfg.par(config, f"gregion[0].{name}") for name in cfg.CONDUCTIVITY_NAMES}
    bath = cfg.par(config, "gregion[2].g_bath")
    converted = {name: 100.0 / rho for name, rho in ROBERTS_SCHER_RESISTIVITY_OHM_CM.items()}     # S/m = 100 / (ohm cm)
    monodomain = {axis: g[f"g_i{axis}"] * g[f"g_e{axis}"] / (g[f"g_i{axis}"] + g[f"g_e{axis}"]) for axis in "lt"}
    bulk = {axis: g[f"g_i{axis}"] + g[f"g_e{axis}"] for axis in "lt"}
    beta = cfg.par(config, "imp_region[0].cellSurfVolRatio")
    return dict(
        supports="Tissue conductivities are the measured Roberts and Scher values; unequal anisotropy ratios; monodomain tensor; expected CV anisotropy.",
        kind="derivation",
        method="sigma [S/m] = 100 / rho [ohm cm]; sigma_m = sigma_i sigma_e / (sigma_i + sigma_e) per axis; in a continuum "
               "monodomain CV scales as sqrt(sigma_m), so CV_t / CV_l = sqrt(sigma_m,t / sigma_m,l). The bath-loading bound is the "
               "CV ratio if the bath short-circuited the extracellular space completely, sqrt((sigma_i + sigma_e) / sigma_e).",
        values=dict(
            configuredSPerM=g,
            convertedFromResistivitySPerM=converted,
            largestRoundingDifferenceSPerM=float(max(abs(g[k] - converted[k]) for k in converted)),
            anisotropyRatioIntracellular=g["g_il"] / g["g_it"],
            anisotropyRatioExtracellular=g["g_el"] / g["g_et"],
            monodomainSPerM=monodomain,
            monodomainAnisotropyRatio=monodomain["l"] / monodomain["t"],
            continuumCvRatioTransverseOverLongitudinal=float(np.sqrt(monodomain["t"] / monodomain["l"])),
            expectedTransverseCvMPerS=float(config["values"]["cvTargetMPerS"] * np.sqrt(monodomain["t"] / monodomain["l"])),
            bulkSPerM=bulk,
            sourceWeightRatioIntracellularOverMonodomain={axis: g[f"g_i{axis}"] / monodomain[axis] for axis in "lt"},
            fibreParallelUnderweightOfMonodomainSource=float((g["g_il"] / monodomain["l"]) / (g["g_it"] / monodomain["t"])),
            bathSPerM=bath,
            bathOverBulkLongitudinal=bath / bulk["l"], bathOverBulkTransverse=bath / bulk["t"],
            bathLoadingCvBound={axis: float(np.sqrt(bulk[axis] / g[f"g_e{axis}"])) for axis in "lt"},
            cellSurfVolRatioPerUm=beta, openCarpDefaultPerUm=0.14, ratioToDefault=beta / 0.14))


# ----------------------------------------------------------------- geometry
def checkGeometry(config):
    size, spacing = cfg.slabSizeMm(config), cfg.spacingMm(config)
    v, f = config["values"], config["fixed"]
    bath = f["bath"]["thicknessMm"]
    tissueNodes = int(np.prod(np.round(size / spacing).astype(int) + 1))
    # openCARP's mesher splits each hexahedral cell into 5 tetrahedra: the pilot mesh (40 x 60 x 4 mm at 0.5 mm)
    # had 384,000 tissue tetrahedra = 5 x 80 x 120 x 8 (Methodology, pilot run). The production run record
    # (runRecord.json nodeSets.elements) holds the actual count.
    tissueCells = int(np.prod(np.round(size / spacing).astype(int)))
    tissueElements = 5 * tissueCells
    layout = geometry.electrodeLayout(config)
    centres = layout["centres"]
    boundary = f["tissue"]["boundaryYmm"]
    distance = boundary - centres[..., 1]                                  # positive inside the lesion
    reachX = float(np.abs(centres[..., 0]).max() + v["electrodeLengthMm"] / 2)
    # Electrode positions that fall inside Wang's region of interest, and their distance from every slab edge.
    inRoi = (distance >= WANG_ROI_MM[0]) & (distance <= WANG_ROI_MM[1])
    yInRoi = centres[..., 1][inRoi]
    marginYLesionSide = float(yInRoi.min() + size[1] / 2)
    marginYViableSide = float(size[1] / 2 - yInRoi.max())
    marginX = float(size[0] / 2 - reachX)
    recordingsPerPlacement = {kind: len(allLabels(kind)) for kind in ("uni", "bi", "omni")}
    placements = len(layout["placements"])
    perRealisation = placements * sum(recordingsPerPlacement.values())
    lesionWidth = f["tissue"]["lesionWidthMm"]
    return dict(
        supports="Slab dimensions, mesh resolution, catheter travel and the number of recordings.",
        kind="derivation",
        method="Counts from the regular block mesh (5 tetrahedra per hexahedral cell, as in the pilot mesh); margins are the smallest distance between an electrode "
               "whose distance from the boundary lies in Wang's region of interest (-10 to +15 mm) and each slab edge.",
        values=dict(
            slabSizeMm=size.tolist(), spacingMm=spacing.tolist(), bathThicknessMm=bath,
            elementsThroughWall=int(round(size[2] / spacing[2])), nodeLayersThroughWall=int(round(size[2] / spacing[2])) + 1,
            tissueNodes=tissueNodes, tissueHexahedralCells=tissueCells, tissueTetrahedraExpected=tissueElements, meshNodesWithBath=cfg.meshNodeCount(config),
            catheterCentresYmm=[float(cfg.centreYmm(config).min()), float(cfg.centreYmm(config).max())],
            catheterStepMm=float(np.diff(cfg.centreYmm(config))[0]),
            electrodeDistanceFromBoundaryMm=[float(distance.min()), float(distance.max())],
            wangRegionOfInterestMm=list(WANG_ROI_MM), wangLargestEfovMm=WANG_LARGEST_EFOV_MM,
            marginRoiElectrodesToLesionSideEdgeMm=marginYLesionSide, marginRoiElectrodesToViableSideEdgeMm=marginYViableSide,
            marginElectrodesToXEdgesMm=marginX,
            smallestMarginOverLargestEfov=min(marginYLesionSide, marginYViableSide, marginX) / WANG_LARGEST_EFOV_MM,
            isolatedTissueBeyondLesionMm=float(size[1] / 2 + boundary - lesionWidth),
            lesionWidthMm=lesionWidth, lesionWidthInElements=lesionWidth / spacing[1],
            lesionWidthIsMeshMultiple=bool(np.isclose(lesionWidth / spacing[1], round(lesionWidth / spacing[1]))),
            placementsPerCondition=placements, gapsMm=list(v["gapsMm"]),
            distinctElectrodesPerCondition=int(len(layout["uniqueCentres"])),
            recordingsPerPlacement=recordingsPerPlacement, recordingsPerConditionAndRealisation=perRealisation,
            noiseRealisations=f["analysis"]["noise"]["repeats"],
            groundTruthPlacements=int(len(cfg.groundTruthCentresYmm(config)) * len(v["gapsMm"])),
            wangMappingPoints=12395))


# ----------------------------------------------------------------- electrodes on the mesh
def latticeAbove(config, centre, halfWidthMm=2.0):
    """Bath lattice nodes of the production mesh around one electrode (positions only; no mesh is built).

    The block mesh is a regular lattice centred on the origin with the configured spacings, and the
    catheter-facing tissue surface z = wall / 2 is a lattice plane, so the bath nodes above the tissue
    are surface + k x transmural spacing, k >= 1 (the surface nodes themselves belong to the tissue).
    """
    spacing, wall = cfg.spacingMm(config), cfg.slabSizeMm(config)[2]
    x = np.arange(-halfWidthMm, halfWidthMm + 1e-9, spacing[0]) + round(centre[0] / spacing[0]) * spacing[0]
    y = np.arange(-halfWidthMm, halfWidthMm + 1e-9, spacing[1]) + round(centre[1] / spacing[1]) * spacing[1]
    layers = int(round(config["fixed"]["bath"]["thicknessMm"] / spacing[2]))
    z = wall / 2 + spacing[2] * np.arange(1, layers + 1)
    return np.stack(np.meshgrid(x, y, z, indexing="ij"), axis=-1).reshape(-1, 3)


def checkElectrodes(config):
    v = config["values"]
    spacing, wall = cfg.spacingMm(config), cfg.slabSizeMm(config)[2]
    cell = float(np.prod(spacing))
    radii = sorted({v["electrodeRadiusMm"], *config["sweep"]["electrodeRadiusMm"]["planned"]})
    gaps = sorted({*v["gapsMm"], *(g for group in config["sweep"]["gapsMm"]["planned"] for g in group)})
    rows = []
    for radius in radii:
        for gap in gaps:
            centre = np.array([2.0, 2.0, wall / 2 + gap + radius])           # an HD Grid electrode position (on the lattice)
            lattice = latticeAbove(config, centre, halfWidthMm=max(2.0, radius + 1.0))
            try:
                nodes = geometry.electrodeNodeSets(lattice, np.arange(len(lattice)), centre[None], radius, v["electrodeLengthMm"])[0]
            except ValueError as error:
                rows.append(dict(radiusMm=radius, gapMm=gap, nodes=0, note=str(error)))
                continue
            heights = lattice[nodes, 2] - wall / 2
            cylinder = np.pi * radius ** 2 * v["electrodeLengthMm"]
            rows.append(dict(radiusMm=radius, gapMm=gap, nodes=int(len(nodes)),
                             lowestNodeAboveTissueMm=float(heights.min()), highestNodeAboveTissueMm=float(heights.max()),
                             nodeLayers=int(len(np.unique(np.round(heights, 6)))),
                             discreteVolumeMm3=float(len(nodes) * cell), cylinderVolumeMm3=float(cylinder),
                             discreteOverCylinderVolume=float(len(nodes) * cell / cylinder),
                             gapInTransmuralElements=gap / spacing[2], gapInInPlaneElements=gap / spacing[0]))
    return dict(
        supports="Electrodes are finite volumes resolved by the mesh at every gap and radius used.",
        kind="numerical",
        method="efov.geometry.electrodeNodeSets on the production lattice around one electrode (cylinder axis along x). "
               "Discrete volume = node count x lattice cell volume, an approximation of the lumped-mass volume openCARP uses.",
        values=dict(spacingMm=spacing.tolist(), electrodeLengthMm=v["electrodeLengthMm"], rows=rows))


# ----------------------------------------------------------------- activation and window timing (prediction)
def checkTiming(config, measuredBilayerCvMPerS=0.735):
    """Predicted timing of the reference wave. measuredBilayerCvMPerS: first-beat surface CV of the bilayer
    slab at the production mesh, wall 1.1 mm (docs/verification_log.md VL-2026-11-04-05)."""
    v, f = config["values"], config["fixed"]
    size = cfg.slabSizeMm(config)
    rule = f["analysis"]["window"]
    halfX = 1.5 * v["gridSpacingMm"] + v["electrodeLengthMm"] / 2
    stimulus = cfg.stimulusDurationMs(config)
    step = 1000.0 / f["analysis"]["acquisitionRateHz"]
    rows = []
    for label, speed in (("strip target (uniform fibres)", float(v["cvTargetMPerS"])), ("bilayer slab, measured on the small slab", measuredBilayerCvMPerS)):
        firstUnderGrid = (size[0] / 2 - halfX) / speed                      # stimulus band at x = -Lx/2, wave along +x
        lastUnderGrid = (size[0] / 2 + halfX) / speed
        farEdge = size[0] / speed
        floor = stimulus + rule["minimumAfterStimulusMs"]
        start = max(firstUnderGrid - rule["leadMs"], floor)
        end = lastUnderGrid + rule["trailMs"]
        rows.append(dict(case=label, speedMPerS=speed, firstActivationUnderGridMs=firstUnderGrid, lastActivationUnderGridMs=lastUnderGrid,
                         waveReachesFarEdgeMs=farEdge, windowStartMs=start, windowEndMs=end, windowLengthMs=end - start,
                         windowStartsAtStimulusFloor=bool(np.isclose(start, floor)),
                         stimulusEndToWindowStartMs=start - stimulus,
                         farEdgeTerminationInsideWindow=bool(start <= farEdge <= end),
                         activationSpanUnderGridMs=lastUnderGrid - firstUnderGrid))
    cycle, beats = cfg.cycleLengthMs(config), cfg.beatCount(config)
    needAfterWindow = f["analysis"]["trailingContextMs"] + f["analysis"]["resampleMarginMs"]
    return dict(
        supports="Measurement window rule (15 ms lead, 30 ms trail, 5 ms after the stimulus); 4 beats of 500 ms hold the window and its context.",
        kind="derivation",
        method="Plane wave along +x from the stimulus band at the slab edge at a constant speed; window rule of fixed.analysis.window. "
               "Predictions only: the production run derives each window from the simulated activation times.",
        values=dict(rows=rows, footprintHalfExtentXmm=halfX, acquisitionStepMs=step, cycleLengthMs=cycle, beats=beats,
                    signalNeededAfterWindowMs=needAfterWindow, filterHistoryMs=f["analysis"]["filterHistoryMs"],
                    lastBeatSlackAfterLongestWindowMs=float(cycle - max(r["windowEndMs"] for r in rows) - needAfterWindow),
                    solverStepsPerCondition=int(round(cfg.par(config, "tend") * 1000 / cfg.timeStepUs(config))),
                    outputFramesPerCondition=int(round(cfg.par(config, "tend") / cfg.sampleStepMs(config))) + 1))


# ----------------------------------------------------------------- resampling
def checkResampling(config):
    a = config["fixed"]["analysis"]
    inputRate = 1000.0 / cfg.sampleStepMs(config)
    outputRate = a["acquisitionRateHz"]
    ratio = Fraction(outputRate / inputRate).limit_denominator(1000)
    up, down = ratio.numerator, ratio.denominator
    # Empirical gain of the production call (efov.measurements.acquire -> scipy.signal.resample_poly defaults), tone by tone.
    duration = 0.4                                                           # s
    t = np.arange(int(round(duration * inputRate))) / inputRate

    def gain(frequency):
        tone = np.sin(2 * np.pi * frequency * t)
        resampled, times = measurements.acquire(config, tone[None], t * 1000.0)
        centre = slice(len(times) // 4, 3 * len(times) // 4)               # away from both ends of the FIR
        return float(np.sqrt(2 * np.mean(resampled[0, centre] ** 2)))      # amplitude of the output sinusoid
    band = a["differentialBandHz"]
    passFrequencies = np.arange(band[0], band[1] + 1e-9, 10.0)
    passGains = np.array([gain(f) for f in passFrequencies])
    # Input frequencies that fold into the analysed band after sampling at the acquisition rate.
    aliasFrequencies = np.arange(outputRate - band[1], min(outputRate + band[1], inputRate / 2) + 1e-9, 25.0)
    aliasGains = np.array([gain(f) for f in aliasFrequencies])
    halfLengthMs = 10 * max(up, down) / (inputRate * up) * 1000.0           # scipy default: half_len = 10 * max(up, down) taps at rate up x input
    return dict(
        supports="5 kHz simulation output is resampled to the 2 kHz acquisition rate without distorting the 30-500 Hz band or aliasing.",
        kind="numerical",
        method="Sinusoids through efov.measurements.acquire (scipy.signal.resample_poly, default Kaiser-window FIR); output amplitude "
               "measured on the central half of the record. Alias band: input frequencies that fold into 30-500 Hz at 2 kHz.",
        values=dict(inputRateHz=inputRate, outputRateHz=outputRate, up=up, down=down, outputNyquistHz=outputRate / 2,
                    inputNyquistHz=inputRate / 2, solverStepsPerOutputFrame=cfg.sampleStepMs(config) * 1000 / cfg.timeStepUs(config),
                    passBandHz=band, passBandGainMin=float(passGains.min()), passBandGainMax=float(passGains.max()),
                    passBandLargestErrorPercent=float(100 * np.max(np.abs(passGains - 1))),
                    passBandWorstFrequencyHz=float(passFrequencies[np.argmax(np.abs(passGains - 1))]),
                    aliasBandHz=[float(aliasFrequencies[0]), float(aliasFrequencies[-1])],
                    aliasLargestGain=float(aliasGains.max()), aliasSmallestRejectionDb=float(-20 * np.log10(aliasGains.max())),
                    aliasWorstFrequencyHz=float(aliasFrequencies[np.argmax(aliasGains)]),
                    firHalfLengthMs=halfLengthMs, configuredResampleMarginMs=a["resampleMarginMs"]))


# ----------------------------------------------------------------- band-pass filter
def checkBandpass(config):
    a = config["fixed"]["analysis"]
    rate = a["acquisitionRateHz"]
    sections = measurements.differentialFilter(config)
    frequencies, response = signal.sosfreqz(sections, worN=2 ** 15, fs=rate)
    magnitude = np.abs(response)
    passband = (frequencies >= a["differentialBandHz"][0]) & (frequencies <= a["differentialBandHz"][1])

    def corner(lowSide):
        level = 1 / np.sqrt(2)
        index = np.flatnonzero(magnitude >= level)
        return float(frequencies[index[0]] if lowSide else frequencies[index[-1]])
    # Noise power gain: output variance / input variance for white noise at the acquisition rate.
    impulse = np.zeros(int(rate)); impulse[0] = 1.0                           # 1 s is far longer than the response
    h = signal.sosfilt(sections, impulse)
    powerGain = float(np.sum(h ** 2))
    cumulative = np.cumsum(h ** 2) / powerGain
    history = int(round(a["filterHistoryMs"] * rate / 1000.0))
    # Group delay by phase differentiation of the cascade.
    phase = np.unwrap(np.angle(response))
    groupDelayMs = -np.gradient(phase, 2 * np.pi * frequencies) * 1000.0
    delays = {f"{f:g}Hz": float(np.interp(f, frequencies, groupDelayMs)) for f in (50, 100, 200, 300, 400)}
    # Step-like input: a worst case for the high-pass section. Time for the response to fall below 0.1 % of its peak.
    step = signal.sosfilt(sections, np.ones(int(rate)))
    settled = np.flatnonzero(np.abs(step) > 1e-3 * np.abs(step).max())
    return dict(
        supports="30-500 Hz causal Butterworth band-pass (order parameter 2); 200 ms of filter history before each window.",
        kind="numerical",
        method="Frequency, impulse and step responses of the second-order sections that efov.measurements.differentialFilter returns.",
        values=dict(orderParameter=a["filterOrder"], polesInCascade=int(2 * len(sections)), bandHz=a["differentialBandHz"],
                    minus3dBHz=[corner(True), corner(False)],
                    passBandGainMin=float(magnitude[passband].min()), passBandGainMax=float(magnitude[passband].max()),
                    gainAt100Hz=float(np.interp(100, frequencies, magnitude)), gainAt250Hz=float(np.interp(250, frequencies, magnitude)),
                    groupDelayMs=delays,
                    noisePowerGain=powerGain, noiseAmplitudeGain=float(np.sqrt(powerGain)),
                    bipolarNoiseSdPerElectrodeSd=float(np.sqrt(2 * powerGain)),
                    impulseEnergyOutsideHistory=float(1 - cumulative[history - 1]),
                    stepResponseBelow0p1PercentAfterMs=float((settled[-1] + 1) * 1000.0 / rate),
                    filterHistoryMs=a["filterHistoryMs"]))


# ----------------------------------------------------------------- noise against the absolute thresholds
def checkNoise(config):
    a = config["fixed"]["analysis"]
    rate = a["acquisitionRateHz"]
    sections = measurements.differentialFilter(config)
    impulse = np.zeros(int(rate)); impulse[0] = 1.0
    powerGain = float(np.sum(signal.sosfilt(sections, impulse) ** 2))
    levels = sorted({config["values"]["noiseSdMv"], *config["sweep"]["noiseSdMv"]["planned"]})
    detection, gate = a["wangReference"]["detectionLimitMv"], a["peakFrequency"]["minimumPpMv"]
    history = int(round(a["filterHistoryMs"] * rate / 1000.0))
    rows = []
    for windowMs in (60.0, 2 * a["ppWindowAroundPfMs"] + 1000.0 / rate):
        samples = int(round(windowMs * rate / 1000.0))
        generator = np.random.default_rng(measurements.NOISE_FLOOR_SEED)
        unit = generator.normal(0.0, 1.0, (4000, 2, history + samples))       # unit-SD noise; every level is a scaling of it
        filtered = signal.sosfilt(sections, unit, axis=-1)[..., history:]
        unitPp = np.ptp(filtered[:, 1] - filtered[:, 0], axis=-1)
        for sd in levels:
            pp = unitPp * sd
            rows.append(dict(windowMs=windowMs, noiseSdMv=sd, bipolarNoiseSdMv=float(sd * np.sqrt(2 * powerGain)),
                             noiseOnlyPpMedianMv=float(np.median(pp)), noiseOnlyPp5thMv=float(np.percentile(pp, 5)),
                             noiseOnlyPp95thMv=float(np.percentile(pp, 95)),
                             fractionAboveExclusionLimit=float(np.mean(pp > detection)),
                             fractionAbovePeakFrequencyGate=float(np.mean(pp >= gate))))
    unitMedian = float(np.median(unitPp))                                       # last window computed; ratio barely depends on it
    return dict(
        supports="Baseline noise 5 uV per electrode and the doubling series, in relation to Wang's 0.01 mV exclusion limit and the 0.04 mV peak-frequency gate.",
        kind="numerical",
        method="Two electrodes with independent Gaussian white noise at the acquisition rate through the production band-pass with its "
               "history; peak-to-peak of their difference over the window; 4,000 realisations, fixed seed. Riccio et al. 2021 range is "
               "quoted from docs/decision_log.md (FM-36), not re-derived here.",
        values=dict(exclusionLimitMv=detection, peakFrequencyGateMv=gate, noisePowerGain=powerGain, rows=rows,
                    noiseSdAtWhichMedianNoisePpEqualsExclusionLimitMv=detection / unitMedian,
                    noiseSdAtWhichMedianNoisePpEqualsPeakFrequencyGateMv=gate / unitMedian,
                    clinicalBipolarNoiseRangeUvRiccio2021=[1.2, 11.0]))


# ----------------------------------------------------------------- wavelet peak frequency
def checkWavelet(config):
    a = config["fixed"]["analysis"]
    rate, settings = a["acquisitionRateHz"], dict(a["peakFrequency"])
    settings["minimumPpMv"] = 0.0                                              # synthetic amplitudes are arbitrary
    omega0 = settings["morletOmega0"]
    windowMs, historyMs = 60.0, a["filterHistoryMs"]
    length = int(round(windowMs * rate / 1000.0))
    start = int(round(historyMs * rate / 1000.0))

    def segment(trailingMs):
        return np.zeros(start + length + int(round(trailingMs * rate / 1000.0)))

    def times(samples):
        return (np.arange(samples) - start) * 1000.0 / rate                    # ms from the window start

    def measure(trace):
        pf, index, pfGlobal = peakfrequency.peakFrequency(trace[None], rate, start, length, settings)
        return float(pf[0]), float(pfGlobal[0])

    tones = []
    for frequency in (50.0, 100.0, 250.0, 450.0):
        trace = segment(a["trailingContextMs"])
        trace[:] = np.sin(2 * np.pi * frequency * times(len(trace)) / 1000.0)
        pf, pfGlobal = measure(trace)
        tones.append(dict(trueHz=frequency, highestTraceHz=pf, globalHz=pfGlobal))
    bursts = []
    for frequency in (100.0, 250.0, 450.0):
        for cycles in (3, 6, 12):
            trace = segment(a["trailingContextMs"])
            t = times(len(trace))
            sigma = cycles / frequency * 1000.0 / 6.0                         # Gaussian envelope spanning about `cycles` periods (+-3 sigma)
            trace[:] = np.exp(-0.5 * ((t - windowMs / 2) / sigma) ** 2) * np.sin(2 * np.pi * frequency * (t - windowMs / 2) / 1000.0)
            pf, pfGlobal = measure(trace)
            bursts.append(dict(trueHz=frequency, cycles=cycles, highestTraceHz=pf, globalHz=pfGlobal,
                               highestTraceErrorHz=pf - frequency, globalErrorHz=pfGlobal - frequency))
    # Window-end bias: a band-passed biphasic deflection close to the window end, against the trailing signal kept.
    sections = measurements.differentialFilter(config)
    edge = []
    for beforeEndMs in (2.0, 5.0, 10.0, 30.0):
        row = dict(deflectionBeforeWindowEndMs=beforeEndMs)
        for trailingMs in (0.0, 25.0, 50.0, 100.0, 150.0, 300.0):
            trace = segment(trailingMs)
            t = times(len(trace))
            centre, width = windowMs - beforeEndMs, 1.5
            raw = -(t - centre) / width * np.exp(-0.5 * ((t - centre) / width) ** 2)        # biphasic (Gaussian derivative), 1.5 ms width
            pf, pfGlobal = measure(signal.sosfilt(sections, raw))
            row[f"highestTraceHz_trailing{trailingMs:g}ms"] = pf
            row[f"globalHz_trailing{trailingMs:g}ms"] = pfGlobal
        edge.append(row)
    lowest = settings["minimumHz"]
    return dict(
        supports="Morlet (omega0 = 6) peak frequency, 30-500 Hz in 1 Hz steps; 150 ms of trailing signal for the wavelet transform.",
        kind="numerical",
        method="efov.peakfrequency.peakFrequency on synthetic signals with a known frequency, read in a 60 ms window after 200 ms of "
               "history. Edge test: 1.5 ms Gaussian-derivative deflection through the production band-pass.",
        values=dict(omega0=omega0, frequencyStepHz=settings["stepHz"], gateFraction=settings["magnitudeGateFraction"],
                    temporalWidthAtLowestFrequencyMs=float(omega0 / (2 * np.pi * lowest) * 1000.0),
                    fourWidthsMs=float(4 * omega0 / (2 * np.pi * lowest) * 1000.0),
                    temporalWidthAt300HzMs=float(omega0 / (2 * np.pi * 300.0) * 1000.0),
                    relativeBandwidthSigmaFOverF=float(1.0 / omega0),
                    configuredTrailingContextMs=a["trailingContextMs"], tones=tones, bursts=bursts, windowEndBias=edge))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", default=str(REPOSITORY / "runs" / "verification" / "local"))
    parser.add_argument("--only", nargs="*")
    args = parser.parse_args()
    config = productionConfig()
    checks = [("conductivities", lambda: checkConductivities(config)), ("geometry", lambda: checkGeometry(config)),
              ("electrodes", lambda: checkElectrodes(config)), ("timing", lambda: checkTiming(config)),
              ("resampling", lambda: checkResampling(config)), ("bandpass", lambda: checkBandpass(config)),
              ("noise", lambda: checkNoise(config)), ("wavelet", lambda: checkWavelet(config))]
    status = runChecks(checks, args.out, args.only)
    return 0 if all(state == "completed" for state in status.values()) else 1


if __name__ == "__main__":
    sys.exit(main())
