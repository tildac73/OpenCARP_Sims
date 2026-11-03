"""Electrogram measures at every catheter placement: acquisition chain, calibration, noise, recordings,
peak-to-peak voltage and peak frequency.

Acquisition chain (FM-09), identical for every signal:
  1. unipolar phi_e at the simulation output rate (4 kHz), scaled by the calibration gain;
  2. rational resampling to the acquisition rate (2 kHz; EnSite X, Europace 2024;26:euae154) with
     scipy.signal.resample_poly, whose Kaiser-window FIR is the anti-alias filter (FM-22: dt 20 us
     and the 5 kHz output step of earlier verified runs are kept, so the ratio is 2/5);
  3. measurement noise: independent Gaussian white noise per electrode at the acquisition rate
     (values.noiseSdMv; repeat 0 is clean);
  4. differential recordings (bipoles, omnipoles) from the 30-500 Hz causal Butterworth band-pass
     of the unipoles (the filter is linear, so filtering then differencing equals the reverse).
Measures (FM-10, FM-11):
  * unipolar PP: max - min over the measurement window;
  * bipolar / omnipolar peak frequency (efov.peakfrequency) on the band-passed segment; PP within
    t_PF +- ppWindowAroundPfMs inside the window, or over the whole window when PF is undetectable
    (the row says which);
  * the selected weights are also applied to the clean signals, and clean PP uses the same PP window.
Calibration (FM-19): fixedGain if given, otherwise the gain that sets the median PP of the nine
selected bipoles at the intact reference placement to targetBipolarMv (imposed normalisation).
Noise realisations are measurement repeats, never biological replicates.
"""
import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
from fractions import Fraction

from scipy.signal import butter, resample_poly, sosfilt

from .config import beatCount, cycleLengthMs
from .geometry import recordingGeometry
from .io import saveJson
from .peakfrequency import peakFrequency
from .recordings import bipole, makeRecordings, recordingLabels


# ----------------------------------------------------------------- acquisition chain
def sampleRateHz(timeMs):
    return 1000.0 / float(np.diff(timeMs[:2])[0])


def differentialFilter(config):
    a = config["fixed"]["analysis"]
    return butter(a["filterOrder"], a["differentialBandHz"], btype="bandpass", fs=a["acquisitionRateHz"], output="sos")


def acquire(config, signals, timeMs):
    """Resample to the acquisition rate (polyphase FIR, zero phase): (signals, times from the first input time)."""
    rate = config["fixed"]["analysis"]["acquisitionRateHz"]
    ratio = Fraction(rate / sampleRateHz(timeMs)).limit_denominator(1000)
    resampled = resample_poly(np.asarray(signals, float), ratio.numerator, ratio.denominator, axis=-1)
    return resampled, timeMs[0] + np.arange(resampled.shape[-1]) * 1000.0 / rate


def measurementSlices(timeMs, config, windowMs):
    """(raw, measure): raw = filter history + window on timeMs; measure = the window inside raw."""
    beatStart = (beatCount(config) - 1) * cycleLengthMs(config)
    start, end = beatStart + windowMs[0], beatStart + windowMs[1]
    history = config["fixed"]["analysis"]["filterHistoryMs"]
    first, begin, stop = (int(np.searchsorted(timeMs, t - 1e-9)) for t in (start - history, start, end + 1e-9))
    if start - history < timeMs[0] - 1e-9 or end > timeMs[-1] + 1e-9 or stop - begin < 2:
        raise ValueError("Electrograms do not contain the window and its filter history")
    return slice(first, stop), slice(begin - first, stop - first)


def stableSeed(baseSeed, condition, repeat):
    """Seed from text, so each noise stream is independent of call order."""
    return int.from_bytes(hashlib.sha256(f"{baseSeed}:{condition}:{repeat}".encode()).digest()[:8], "little")


# ----------------------------------------------------------------- calibration
def calibrateGain(config, egm, windowMs):
    c = config["fixed"]["analysis"]["calibration"]
    if c["fixedGain"] is not None:
        return dict(gain=float(c["fixedGain"]), source="fixedGain (from the inferior reference run, FM-19)",
                    interpretation="Imposed amplitude calibration, not an independent physiological validation")
    placements = egm["placements"]
    reference = np.isclose(placements[:, 1], c["referenceCentreYmm"]) & np.isclose(placements[:, 2], c["referenceGapMm"])
    if reference.sum() != 1:
        raise ValueError("Calibration must identify exactly one intact placement")
    uni, times = acquire(config, egm["uni"][reference].astype(float), egm["timeMs"])
    raw, measure = measurementSlices(times, config, windowMs)
    filtered = sosfilt(differentialFilter(config), uni[..., raw], axis=-1)[..., measure]
    corners = [4 * r + k for r in range(3) for k in range(3)]
    values = [np.ptp(bipole(filtered, f"E{c0}-E{c0 + 1}|E{c0 + 4}")[0], axis=-1)[0] for c0 in corners]
    median = float(np.median(values))
    if not np.isfinite(median) or median <= 1e-12:
        raise ValueError("Calibration signal is absent or negligible")
    return dict(gain=c["targetBipolarMv"] / median, rawReferenceMedianMv=median, targetBipolarMv=c["targetBipolarMv"],
                centreYmm=c["referenceCentreYmm"], gapMm=c["referenceGapMm"], source="computed in this run",
                interpretation="Imposed amplitude calibration, not an independent physiological validation")


# ----------------------------------------------------------------- measures
def ppAroundPeak(trace, measure, pfIndex, halfSamples):
    """PP within pfIndex +- halfSamples inside the measurement window, or the whole window if PF is undetectable."""
    if pfIndex < 0:
        return float(np.ptp(trace[measure])), "window"
    lo, hi = max(measure.start, pfIndex - halfSamples), min(measure.stop, pfIndex + halfSamples + 1)
    return float(np.ptp(trace[lo:hi])), "peakFrequency"


def measureCondition(config, condition, egm, gain, windowMs):
    """Points table (one row per recording, placement and noise realisation) and the window traces."""
    a, v = config["fixed"]["analysis"], config["values"]
    uniAcq, times = acquire(config, gain * egm["uni"].astype(float), egm["timeMs"])
    raw, measure = measurementSlices(times, config, windowMs)
    clean = uniAcq[..., raw]
    segmentTimes = times[raw]
    sections, rate = differentialFilter(config), a["acquisitionRateHz"]
    cleanDiff = sosfilt(sections, clean, axis=-1)
    half = int(round(a["ppWindowAroundPfMs"] * rate / 1000.0))
    labels = recordingLabels(a["recordings"])
    rows, measured, cleanTraces = [], [], []
    for repeat in range(a["noise"]["repeats"] + 1):
        noise = np.zeros_like(clean)
        if repeat:
            noise = np.random.default_rng(stableSeed(a["noise"]["seed"], condition, repeat)).normal(0, v["noiseSdMv"], clean.shape)
        uni = clean + noise
        diff = sosfilt(sections, uni, axis=-1)
        for kind, label, _, weights in makeRecordings(uni[..., measure], diff[..., measure], labels, a["omniAngleStepDeg"]):
            source, cleanSource = (uni, clean) if kind == "uni" else (diff, cleanDiff)
            traces = np.einsum("pe,pet->pt", weights, source)
            cleanAtWeights = np.einsum("pe,pet->pt", weights, cleanSource)
            if kind == "uni":
                pf, pfIndex, pfGlobal = (np.full(len(traces), np.nan), np.full(len(traces), -1), np.full(len(traces), np.nan))
            else:
                pf, pfIndex, pfGlobal = peakFrequency(traces, rate, measure, a["peakFrequency"])
            for p in range(len(traces)):
                if kind == "uni":
                    pp, cleanPp, ppWindow = float(np.ptp(traces[p, measure])), float(np.ptp(cleanAtWeights[p, measure])), "window"
                else:
                    pp, ppWindow = ppAroundPeak(traces[p], measure, int(pfIndex[p]), half)
                    cleanPp, _ = ppAroundPeak(cleanAtWeights[p], measure, int(pfIndex[p]), half)
                rows.append(dict(
                    recordingId=f"{condition}:{repeat}:{p}:{kind}:{label}", condition=condition, noiseRepeat=repeat,
                    placement=p, kind=kind, label=label, gapMm=float(egm["placements"][p, 2]),
                    centreYmm=float(egm["placements"][p, 1]), ppMv=pp, cleanPpMv=cleanPp, ppWindow=ppWindow,
                    pfHz=float(pf[p]), pfGlobalHz=float(pfGlobal[p]),
                    pfTimeMs=float(segmentTimes[pfIndex[p]]) if pfIndex[p] >= 0 else np.nan,
                    windowStartMs=float(segmentTimes[measure][0]), windowEndMs=float(segmentTimes[measure][-1]),
                    traceIndex=len(measured),
                    **recordingGeometry(egm["centres"][p], kind, label, weights[p], config["fixed"]["tissue"]["boundaryYmm"]),
                    **{f"weightE{k}": float(weights[p, k]) for k in range(16)}))
                measured.append(traces[p, measure]); cleanTraces.append(cleanAtWeights[p, measure])
    return pd.DataFrame(rows), np.asarray(measured, np.float32), np.asarray(cleanTraces, np.float32), segmentTimes[measure], (uniAcq, times)


def filterHistoryCheck(config, acquired, windowMs):
    """Max |difference| between band-passing with one and with two filter histories, relative to the largest clean PP."""
    uniAcq, times = acquired
    raw, measure = measurementSlices(times, config, windowMs)
    sections = differentialFilter(config)
    short = sosfilt(sections, uniAcq[..., raw], axis=-1)[..., measure]
    longer = sosfilt(sections, uniAcq[..., :raw.stop], axis=-1)[..., -short.shape[-1]:]
    error = float(np.max(np.abs(short - longer)) / max(np.max(np.ptp(short, axis=-1)), 1e-12))
    rate = config["fixed"]["analysis"]["acquisitionRateHz"]
    available = raw.start >= int(round(config["fixed"]["analysis"]["filterHistoryMs"] * rate / 1000)) - 1
    return dict(relativeError=error, doubleHistoryAvailable=bool(available),
                passed=bool(available and error <= config["fixed"]["acceptance"]["profile"]["maximumFilterHistoryError"]))


def profileChecks(points, config):
    """Per clean profile: a stable plateau far into viable tissue and (lesion) a drop across the boundary.

    farRelativeRange = (max - min) / median PP over positions >= farRegionMm into viable tissue;
    contrast = (far-viable median - far-lesion median) / far-viable median. Descriptive prerequisites:
    overlapping placements are not independent replicates.
    """
    p = config["fixed"]["acceptance"]["profile"]
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
        table, measured, clean, times, acquired = measureCondition(config, condition, egm, calibration["gain"], windowMs)
        filterChecks[condition] = filterHistoryCheck(config, acquired, windowMs)
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
