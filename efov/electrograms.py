"""Unipolar electrograms at every electrode, from openCARP's phi_e, and the forward-model check.

The unipolar signal of an electrode is openCARP's phi_e (referenced to the grounded bath corner)
interpolated (P1) at its surface samples and averaged over them: a passive sensing approximation,
not an equipotential conducting electrode. Only the frames the analysis reads are extracted:
the last beat's measurement window plus twice the filter history (the second history is for the
filter-history check).
"""
from pathlib import Path

import numpy as np

from .forward import pointInterpolator
from .geometry import electrodeLayout
from .io import igbTimes, loadJson, readIgb, saveJson


def analysisTimes(config, windowMs):
    """(start, end) in absolute ms of the frames to extract: window - 2 x history .. window end."""
    beatStart = (config["pacing"]["beatCount"] - 1) * config["pacing"]["cycleLengthMs"]
    return beatStart + windowMs[0] - 2 * config["analysis"]["filterHistoryMs"], beatStart + windowMs[1]


def frameSlice(times, startMs, endMs):
    start, stop = int(np.searchsorted(times, startMs - 1e-9)), int(np.searchsorted(times, endMs + 1e-9))
    if startMs < times[0] - 1e-9 or endMs > times[-1] + 1e-9:
        raise ValueError(f"Frames {startMs}-{endMs} ms lie outside the recording ({times[0]}-{times[-1]} ms)")
    return slice(start, stop)


def readPhie(job, frames):
    """phie.igb frames addressed on the vm.igb time axis, on whole-mesh numbering (NaN where not written)."""
    job = Path(job)
    offset = 0
    if (job / "restart.json").exists():
        step = float(np.diff(igbTimes(job / "vm.igb")[:2])[0])
        offset = int(round(loadJson(job / "restart.json")["phieStartMs"] / step))
    if frames.start < offset:
        raise ValueError(f"{job}: frame {frames.start} precedes phie.igb (starts at frame {offset}); raise prepacingMarginMs")
    values = readIgb(job / "phie.igb", slice(frames.start - offset, frames.stop - offset))
    nodes = np.load(job / "phieNodes.npy")
    if values.shape[1] != len(nodes):
        raise ValueError(f"{job}: phie.igb has {values.shape[1]} nodes, phieNodes.npy lists {len(nodes)}")
    full = np.full((values.shape[0], loadJson(job / "runRecord.json")["nodeSets"]["nodes"]), np.nan, values.dtype)
    full[:, nodes] = values
    return full


def electrodeInterpolator(tissue, layout):
    return pointInterpolator(tissue["positions"], tissue["elements"], layout["samplePoints"])


def extractElectrograms(config, job, tissue, windowMs, outputPath, interpolator=None, block=200):
    """Unipolar phi at every electrode for the analysis frames -> npz (uni: placements x 16 x frames, mV)."""
    layout = electrodeLayout(config)
    interpolator = interpolator if interpolator is not None else electrodeInterpolator(tissue, layout)
    times = igbTimes(Path(job) / "vm.igb")
    frames = frameSlice(times, *analysisTimes(config, windowMs))
    chunks = []
    for first in range(frames.start, frames.stop, block):
        part = readPhie(job, slice(first, min(first + block, frames.stop)))
        samples = interpolator @ part.T.astype(float)                       # (sample points, frames)
        if not np.isfinite(samples).all():
            raise ValueError("Electrode sample outside the phi_e output layer (raise numerics.phieOutputHeightMm)")
        chunks.append(samples[layout["sampleIndex"]].mean(axis=2))          # (placements, 16, frames)
    np.savez_compressed(outputPath, uni=np.concatenate(chunks, axis=-1).astype(np.float32), timeMs=times[frames],
                        centres=layout["centres"], samples=layout["samples"], placements=layout["placements"])
    return outputPath


def forwardModelCheck(config, job, tissue, model, windowMs, interpolator=None):
    """openCARP phi_e vs the independent sigma_m solve at all electrode samples, on frames spread over the window.

    Tests the configuration (tags, conductivities, ground, node order), not the physiology. Gate:
    acceptance.implementation (relative RMS and correlation per sample).
    """
    gate = config["acceptance"]["implementation"]
    interpolator = interpolator if interpolator is not None else electrodeInterpolator(tissue, electrodeLayout(config))
    times = igbTimes(Path(job) / "vm.igb")
    beatStart = (config["pacing"]["beatCount"] - 1) * config["pacing"]["cycleLengthMs"]
    window = frameSlice(times, beatStart + windowMs[0], beatStart + windowMs[1])
    chosen = np.unique(np.linspace(window.start, window.stop - 1, gate["forwardModelFrames"]).astype(int))
    carp, fem = [], []
    for frame in chosen:
        carp.append(interpolator @ readPhie(job, slice(frame, frame + 1))[0].astype(float))
        vm = readIgb(Path(job) / "vm.igb", slice(frame, frame + 1))[0]
        fem.append(interpolator @ model.solver.solve(model.fullVm(vm)))
    carp, fem = np.array(carp).T, np.array(fem).T                            # (samples, frames)
    rms = lambda x: np.sqrt(np.mean(x ** 2, axis=1))
    relative = rms(carp - fem) / np.maximum(rms(fem), 1e-30)
    a, b = carp - carp.mean(axis=1, keepdims=True), fem - fem.mean(axis=1, keepdims=True)
    correlation = np.sum(a * b, axis=1) / np.maximum(np.sqrt(np.sum(a * a, axis=1) * np.sum(b * b, axis=1)), 1e-30)
    result = dict(frames=len(chosen), samples=int(len(carp)), relativeRmsMax=float(relative.max()),
                  relativeRmsMedian=float(np.median(relative)), correlationMin=float(correlation.min()),
                  solver=model.solver.method)
    result["passed"] = bool(result["relativeRmsMax"] <= gate["forwardModelRelativeRms"]
                            and result["correlationMin"] >= gate["forwardModelCorrelation"])
    return result


def runElectrograms(config, jobs, tissues, models, windowMs, folder):
    folder = Path(folder)
    checks, layout = {}, electrodeLayout(config)
    for condition, job in jobs.items():
        interpolator = electrodeInterpolator(tissues[condition], layout)
        checks[condition] = forwardModelCheck(config, job, tissues[condition], models[condition], windowMs, interpolator)
        extractElectrograms(config, job, tissues[condition], windowMs, folder / f"{condition}.npz", interpolator)
    saveJson(folder / "forwardModelCheck.json", checks)
    return checks
