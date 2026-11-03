"""Electrogram measures at every catheter placement: calibration, noise, recordings, peak-to-peak.

Order (runMeasurements):
  1. gain: one intact reference placement; median peak-to-peak of its nine selected bipoles is set
     to targetBipolarMv. The same gain scales every signal of both conditions (imposed normalisation,
     not validation).
  2. per condition and noise realisation (0 = clean; 1..repeats = independent Gaussian noise per
     electrode, sdMv, stable seeds): unipoles unfiltered, differential signals band-passed with the
     filter history; declared recordings built; peak-to-peak in the measurement window. The selected
     weights are also applied to the clean signals (clean PP at the selected weights).
  3. filter-history check: the same beat filtered with twice the history must agree.
  4. profile checks: a stable far-viable plateau and a lesion transition (descriptive prerequisites).
Noise realisations are measurement repeats, never biological replicates.
"""
import hashlib
from pathlib import Path

import numpy as np
import pandas as pd

from .geometry import recordingGeometry
from .io import saveJson
from .recordings import applyFilter, bandPass, bipole, makeRecordings


def measurementSlices(timeMs, config, windowMs):
    """(raw, measure): raw = filter history + window on timeMs; measure = the window inside raw."""
    beatStart = (config["pacing"]["beatCount"] - 1) * config["pacing"]["cycleLengthMs"]
    start, end = beatStart + windowMs[0], beatStart + windowMs[1]
    history = config["analysis"]["filterHistoryMs"]
    first, begin, stop = (int(np.searchsorted(timeMs, t - 1e-9)) for t in (start - history, start, end))
    if start - history < timeMs[0] - 1e-9 or end > timeMs[-1] + 1e-9 or stop - begin < 2:
        raise ValueError("Electrograms do not contain the window and its filter history")
    return slice(first, stop), slice(begin - first, stop - first)


def sampleRateHz(timeMs):
    return 1000.0 / float(np.diff(timeMs[:2])[0])


def differentialFilter(config, timeMs):
    a = config["analysis"]
    return bandPass(a["differentialBandHz"], sampleRateHz(timeMs), a["filterOrder"])


def stableSeed(baseSeed, condition, repeat):
    """Seed from text, so each noise stream is independent of call order."""
    return int.from_bytes(hashlib.sha256(f"{baseSeed}:{condition}:{repeat}".encode()).digest()[:8], "little")


def calibrateGain(config, egm, windowMs):
    c = config["analysis"]["calibration"]
    raw, measure = measurementSlices(egm["timeMs"], config, windowMs)
    placements = egm["placements"]
    reference = np.isclose(placements[:, 1], c["referenceCentreYmm"]) & np.isclose(placements[:, 2], c["referenceGapMm"])
    if reference.sum() != 1:
        raise ValueError("Calibration must identify exactly one intact placement")
    filtered = applyFilter(differentialFilter(config, egm["timeMs"]), egm["uni"][reference][..., raw].astype(float))[..., measure]
    corners = [4 * r + k for r in range(3) for k in range(3)]
    values = [np.ptp(bipole(filtered, f"E{c0}-E{c0 + 1}|E{c0 + 4}")[0], axis=-1)[0] for c0 in corners]
    median = float(np.median(values))
    if not np.isfinite(median) or median <= 1e-8:
        raise ValueError("Calibration signal is absent or negligible")
    return dict(gain=c["targetBipolarMv"] / median, rawReferenceMedianMv=median, targetBipolarMv=c["targetBipolarMv"],
                centreYmm=c["referenceCentreYmm"], gapMm=c["referenceGapMm"],
                interpretation="Imposed amplitude calibration, not an independent physiological validation")


def filterHistoryCheck(config, egm, gain, windowMs, cleanDifferential):
    """Max |difference| between filtering with one and with two histories, relative to the largest clean PP."""
    raw, measure = measurementSlices(egm["timeMs"], config, windowMs)
    sections = differentialFilter(config, egm["timeMs"])
    longer = applyFilter(sections, gain * egm["uni"][..., :raw.stop].astype(float))[..., -cleanDifferential.shape[-1]:]
    error = float(np.max(np.abs(cleanDifferential - longer)) / max(np.max(np.ptp(cleanDifferential, axis=-1)), 1e-12))
    available = raw.start >= int(round(config["analysis"]["filterHistoryMs"] * sampleRateHz(egm["timeMs"]) / 1000)) - 1
    return dict(relativeError=error, doubleHistoryAvailable=bool(available),
                passed=bool(available and error <= config["acceptance"]["profile"]["maximumFilterHistoryError"]))


def measureCondition(config, condition, egm, gain, windowMs):
    """Points table (one row per recording, placement and noise realisation) and the traces."""
    a = config["analysis"]
    raw, measure = measurementSlices(egm["timeMs"], config, windowMs)
    sections = differentialFilter(config, egm["timeMs"])
    clean = gain * egm["uni"][..., raw].astype(float)
    cleanUni, cleanDiff = clean[..., measure], applyFilter(sections, clean)[..., measure]
    measureTimes = egm["timeMs"][raw][measure]
    rows, measured, cleanTraces = [], [], []
    for repeat in range(a["noise"]["repeats"] + 1):
        noise = np.zeros_like(clean)
        if repeat:
            noise = np.random.default_rng(stableSeed(a["noise"]["seed"], condition, repeat)).normal(0, a["noise"]["sdMv"], clean.shape)
        uni = (clean + noise)[..., measure]
        diff = applyFilter(sections, clean + noise)[..., measure]
        for kind, label, traces, weights in makeRecordings(uni, diff, a["recordings"], a["omniAngleStepDeg"]):
            cleanAtWeights = np.einsum("pe,pet->pt", weights, cleanUni if kind == "uni" else cleanDiff)
            for p in range(len(traces)):
                rows.append(dict(
                    recordingId=f"{condition}:{repeat}:{p}:{kind}:{label}", condition=condition, noiseRepeat=repeat,
                    placement=p, kind=kind, label=label, gapMm=float(egm["placements"][p, 2]),
                    centreYmm=float(egm["placements"][p, 1]), ppMv=float(np.ptp(traces[p])),
                    cleanPpMv=float(np.ptp(cleanAtWeights[p])), noisePpMv=float(np.ptp(traces[p] - cleanAtWeights[p])),
                    windowStartMs=float(measureTimes[0]), windowEndMs=float(measureTimes[-1]), traceIndex=len(measured),
                    **recordingGeometry(egm["centres"][p], kind, label, weights[p], config["lesion"]["boundaryYmm"]),
                    **{f"weightE{k}": float(weights[p, k]) for k in range(16)}))
                measured.append(traces[p]); cleanTraces.append(cleanAtWeights[p])
    return pd.DataFrame(rows), np.asarray(measured, np.float32), np.asarray(cleanTraces, np.float32), measureTimes, cleanDiff


def profileChecks(points, config):
    """Per clean profile: stable plateau far into viable tissue, and (lesion) a drop across the boundary.

    farRelativeRange = (max - min) / median PP over positions >= farRegionMm into viable tissue;
    contrast = (far-viable median - far-lesion median) / far-viable median. Descriptive only:
    overlapping placements are not independent replicates.
    """
    p = config["acceptance"]["profile"]
    rows = []
    for (condition, kind, label, gap), group in points[points.noiseRepeat == 0].groupby(["condition", "kind", "label", "gapMm"]):
        viable = -group.distanceInsideMm
        profile = group[viable >= p["farRegionMm"]].groupby("distanceInsideMm").ppMv.median()
        median = float(profile.median()) if len(profile) else np.nan
        spread = float((profile.max() - profile.min()) / median) if median > 0 else np.nan
        lesionSide = group[group.distanceInsideMm >= p["farRegionMm"]].ppMv.median()
        contrast = float((median - lesionSide) / median) if median > 0 else np.nan
        rows.append(dict(condition=condition, kind=kind, label=label, gapMm=gap, farPositions=len(profile),
                         farMedianMv=median, farRelativeRange=spread,
                         farStable=bool(len(profile) >= 3 and spread <= p["maximumFarRelativeRange"]),
                         contrastFraction=contrast,
                         transitionObserved=bool(condition == "lesion" and contrast >= p["minimumTransitionFraction"])))
    return pd.DataFrame(rows)


def runMeasurements(config, egmFolder, windowMs, folder):
    folder = Path(folder)
    egms = {c: dict(np.load(Path(egmFolder) / f"{c}.npz")) for c in ("intact", "lesion")}
    calibration = calibrateGain(config, egms["intact"], windowMs)
    saveJson(folder / "calibration.json", calibration)
    tables, filterChecks = [], {}
    for condition, egm in egms.items():
        table, measured, clean, times, cleanDiff = measureCondition(config, condition, egm, calibration["gain"], windowMs)
        filterChecks[condition] = filterHistoryCheck(config, egm, calibration["gain"], windowMs, cleanDiff)
        table.to_csv(folder / f"{condition}Points.csv", index=False)
        np.savez_compressed(folder / f"{condition}Traces.npz", timeMs=times, measuredMv=measured, cleanMv=clean)
        tables.append(table)
    points = pd.concat(tables, ignore_index=True)
    checks = profileChecks(points, config)
    checks.to_csv(folder / "profileChecks.csv", index=False)
    saveJson(folder / "filterHistoryCheck.json", filterChecks)
    status = dict(filterHistory=all(c["passed"] for c in filterChecks.values()),
                  stableFarRegion=bool(checks.farStable.all()),
                  lesionTransition=bool(checks[checks.condition == "lesion"].transitionObserved.all()))
    return points, calibration, status
