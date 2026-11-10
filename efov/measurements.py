"""Electrogram measures at every catheter placement: acquisition chain, noise, recordings,
peak-to-peak voltage and peak frequency. Amplitudes stay in the model's physical units (mV); they
are never scaled to clinical values (FM-25).

Acquisition chain (FM-09), identical for every signal:
  1. unipolar phi_e at the simulation output rate (5 kHz);
  2. rational resampling to the acquisition rate (2 kHz; EnSite X, Europace 2024;26:euae154) with
     scipy.signal.resample_poly, whose Kaiser-window FIR is the anti-alias filter (FM-22: dt 20 us
     and the 5 kHz output step of earlier verified runs are kept, so the ratio is 2/5);
  3. measurement noise: independent Gaussian white noise per electrode at the acquisition rate
     (values.noiseSdMv; repeat 0 is clean);
  4. differential recordings (bipoles, omnipoles) from the 30-500 Hz causal Butterworth band-pass
     of the unipoles (the filter is linear, so filtering then differencing equals the reverse).
Processing segment (FM-36): filtering and the wavelet transform run from filterHistoryMs before the
earliest window to trailingContextMs after the latest. The band-pass is causal, so only the wavelet
transform uses the trailing context; every measure is still read inside the placement's window.
Measures (FM-10, FM-11, FM-27), each inside its placement's own window (FM-28):
  * unipolar PP: max - min over the measurement window;
  * bipolar / omnipolar peak frequency (efov.peakfrequency, both definitions) on the band-passed
    segment; PP within t_PF +- ppWindowAroundPfMs inside the window, or over the whole window when PF
    is undetectable (the row says which);
  * the selected weights are also applied to the clean signals, and clean PP uses the same PP window.
Wang's clinical voltages are reported beside the model's (amplitudes.json) and never applied.
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
from .peakfrequency import peakFrequency, windowed
from .recordings import makeRecordings, recordingLabels


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


def placementIndices(timeMs, config, windows, trailingMs=0.0):
    """raw: slice of timeMs from (earliest window start - filter history) to (latest window end + trailingMs);
    starts: first sample of each placement's window inside raw; length: samples per window.

    trailingMs = 0 where only the causal band-pass runs (nothing after a window changes its output);
    fixed.analysis.trailingContextMs where the wavelet transform runs (FM-36).
    """
    windows = np.atleast_2d(np.asarray(windows, float))
    beatStart = (beatCount(config) - 1) * cycleLengthMs(config)
    history = config["fixed"]["analysis"]["filterHistoryMs"]
    begin, end = beatStart + windows[:, 0], beatStart + windows[:, 1]
    if begin.min() - history < timeMs[0] - 1e-9 or end.max() + trailingMs > timeMs[-1] + 1e-9:
        raise ValueError(f"Electrograms ({timeMs[0]:.1f}-{timeMs[-1]:.1f} ms) do not contain every window with its filter history and "
                         f"{trailingMs:g} ms of trailing context ({begin.min() - history:.1f}-{end.max() + trailingMs:.1f} ms)")
    first = int(np.searchsorted(timeMs, begin.min() - history - 1e-9))
    stop = int(np.searchsorted(timeMs, end.max() + trailingMs + 1e-9))
    starts = np.searchsorted(timeMs, begin - 1e-9) - first
    lengths = np.searchsorted(timeMs, end + 1e-9) - first - starts
    if lengths.min() != lengths.max() or lengths[0] < 2:
        raise ValueError("Placement windows must have the same length on the acquisition grid")
    return slice(first, stop), starts.astype(int), int(lengths[0])


def stableSeed(baseSeed, condition, repeat):
    """Seed from text, so each noise stream is independent of call order."""
    return int.from_bytes(hashlib.sha256(f"{baseSeed}:{condition}:{repeat}".encode()).digest()[:8], "little")


# ----------------------------------------------------------------- measures
def ppAroundPeak(trace, measure, pfIndex, halfSamples):
    """PP within pfIndex +- halfSamples inside the measurement window, or the whole window if PF is undetectable."""
    if pfIndex < 0:
        return float(np.ptp(trace[measure])), "window"
    lo, hi = max(measure.start, pfIndex - halfSamples), min(measure.stop, pfIndex + halfSamples + 1)
    return float(np.ptp(trace[lo:hi])), "peakFrequency"


def measureCondition(config, condition, egm, windows):
    """Points table (one row per recording, placement and noise realisation) and the window traces."""
    a, v = config["fixed"]["analysis"], config["values"]
    uniAcq, times = acquire(config, egm["uni"].astype(float), egm["timeMs"])
    raw, starts, length = placementIndices(times, config, windows, a["trailingContextMs"])
    clean = uniAcq[..., raw]
    segmentTimes = times[raw]
    sections, rate = differentialFilter(config), a["acquisitionRateHz"]
    cleanDiff = sosfilt(sections, clean, axis=-1)
    half = int(round(a["ppWindowAroundPfMs"] * rate / 1000.0))
    labels = recordingLabels(a["recordings"])
    rows, measured, cleanTraces, traceStarts = [], [], [], []
    for repeat in range(a["noise"]["repeats"] + 1):
        noise = np.zeros_like(clean)
        if repeat:
            noise = np.random.default_rng(stableSeed(a["noise"]["seed"], condition, repeat)).normal(0, v["noiseSdMv"], clean.shape)
        uni = clean + noise
        diff = sosfilt(sections, uni, axis=-1)
        for kind, label, _, weights in makeRecordings(windowed(uni, starts, length), windowed(diff, starts, length),
                                                      labels, a["omniAngleStepDeg"]):
            source, cleanSource = (uni, clean) if kind == "uni" else (diff, cleanDiff)
            traces = np.einsum("pe,pet->pt", weights, source)
            cleanAtWeights = np.einsum("pe,pet->pt", weights, cleanSource)
            if kind == "uni":
                pf, pfIndex, pfGlobal = (np.full(len(traces), np.nan), np.full(len(traces), -1), np.full(len(traces), np.nan))
            else:
                pf, pfIndex, pfGlobal = peakFrequency(traces, rate, starts, length, a["peakFrequency"])
            for p in range(len(traces)):
                measure = slice(int(starts[p]), int(starts[p]) + length)
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
                traceStarts.append(float(segmentTimes[measure][0]))
    traceInfo = dict(windowStartMs=np.asarray(traceStarts), stepMs=1000.0 / rate)
    return pd.DataFrame(rows), np.asarray(measured, np.float32), np.asarray(cleanTraces, np.float32), traceInfo, (uniAcq, times)


def filterHistoryCheck(config, acquired, windows):
    """Max |difference| between band-passing with one and with two filter histories, relative to the largest clean PP."""
    uniAcq, times = acquired
    raw, starts, length = placementIndices(times, config, windows)
    sections = differentialFilter(config)
    short = sosfilt(sections, uniAcq[..., raw], axis=-1)[..., starts.min():]
    longer = sosfilt(sections, uniAcq[..., :raw.stop], axis=-1)[..., -short.shape[-1]:]
    error = float(np.max(np.abs(short - longer)) / max(np.max(np.ptp(short, axis=-1)), 1e-12))
    rate = config["fixed"]["analysis"]["acquisitionRateHz"]
    available = raw.start >= int(round(config["fixed"]["analysis"]["filterHistoryMs"] * rate / 1000)) - 1
    return dict(relativeError=error, doubleHistoryAvailable=bool(available),
                passed=bool(available and error <= config["fixed"]["acceptance"]["profile"]["maximumFilterHistoryError"]))


def amplitudeSummary(points, config):
    """Model amplitudes (physical mV) far into viable tissue beside Wang's clinical medians; nothing is scaled (FM-25)."""
    far = config["fixed"]["acceptance"]["profile"]["farRegionMm"]
    region = config["values"]["region"]
    rows = []
    for (kind, gap), group in points[(points.noiseRepeat == 0) & (points.condition == "intact") & (points.distanceInsideMm <= -far)].groupby(["kind", "gapMm"]):
        rows.append(dict(kind=kind, gapMm=float(gap), medianCleanPpMv=float(group.cleanPpMv.median()), recordings=int(len(group))))
    return dict(modelFarViable=rows, wangBipolarMedianMv=config["fixed"]["analysis"]["wangReference"]["bipolarMedianMv"].get(region),
                region=region, note="Reported for comparison only; the model's amplitudes are not calibrated to these values (FM-25).")


def pfNoiseRobustness(points):
    """Median clean and noisy PF (both definitions) by recording kind, gap, condition and 1 mm distance bin (descriptive, FM-27)."""
    differential = points[points.kind != "uni"].copy()
    differential["distanceBinMm"] = np.round(differential.distanceInsideMm)
    differential["noisy"] = differential.noiseRepeat > 0
    table = differential.groupby(["condition", "kind", "gapMm", "distanceBinMm", "noisy"])[["pfHz", "pfGlobalHz"]].median().unstack("noisy")
    table.columns = [f"{column}_{'noisy' if noisy else 'clean'}" for column, noisy in table.columns]
    return table.reset_index()


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


def runMeasurements(config, egmFolder, windows, folder):
    folder = Path(folder)
    egms = {c: dict(np.load(Path(egmFolder) / f"{c}.npz")) for c in ("intact", "lesion")}
    tables, filterChecks = [], {}
    for condition, egm in egms.items():
        table, measured, clean, traceInfo, acquired = measureCondition(config, condition, egm, windows)
        filterChecks[condition] = filterHistoryCheck(config, acquired, windows)
        table.to_csv(folder / f"{condition}Points.csv", index=False)
        np.savez_compressed(folder / f"{condition}Traces.npz", measuredMv=measured, cleanMv=clean, **traceInfo)
        tables.append(table)
    points = pd.concat(tables, ignore_index=True)
    checks = profileChecks(points, config)
    checks.to_csv(folder / "profileChecks.csv", index=False)
    pfNoiseRobustness(points).to_csv(folder / "pfNoiseRobustness.csv", index=False)
    amplitudes = amplitudeSummary(points, config)
    saveJson(folder / "amplitudes.json", amplitudes)
    saveJson(folder / "filterHistoryCheck.json", filterChecks)
    status = dict(filterHistory=all(c["passed"] for c in filterChecks.values()),
                  stableFarRegion=bool(checks.farStable.all()),
                  lesionTransition=bool(checks[checks.condition == "lesion"].transitionObserved.all()))
    return points, amplitudes, status
