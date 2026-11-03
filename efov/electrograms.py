"""Unipolar electrograms at every electrode, from openCARP's pseudo-bidomain phi_e (DL-2026-11-02-01).

An electrode is the set of mesh nodes inside its cylinder (efov.geometry, FM-14); its unipolar
signal is the mean phi_e over that set, referenced to the grounded bath corner. This is a passive
sensing approximation, not an equipotential conducting electrode. Only the frames the analysis
reads are extracted: the last beat's measurement window plus twice the filter history (the second
history serves the filter-history check). The independent forward-model check was removed (FM-13).
"""
from pathlib import Path

import numpy as np

from .config import beatCount, cycleLengthMs
from .geometry import electrodeAveraging, electrodeLayout, electrodeNodeSets
from .io import igbTimes, loadJson, readIgb, saveJson


def analysisTimes(config, windowMs):
    """(start, end) in absolute ms of the frames to extract: window - 2 x history .. window end."""
    beatStart = (beatCount(config) - 1) * cycleLengthMs(config)
    return beatStart + windowMs[0] - 2 * config["fixed"]["analysis"]["filterHistoryMs"], beatStart + windowMs[1]


def frameSlice(times, startMs, endMs):
    if startMs < times[0] - 1e-9 or endMs > times[-1] + 1e-9:
        raise ValueError(f"Frames {startMs}-{endMs} ms lie outside the recording ({times[0]}-{times[-1]} ms)")
    return slice(int(np.searchsorted(times, startMs - 1e-9)), int(np.searchsorted(times, endMs + 1e-9)))


def phieTimes(job):
    """Absolute times (ms) of the phie.igb frames."""
    job = Path(job)
    times = igbTimes(job / "phie.igb")
    if (job / "restart.json").exists():
        start = loadJson(job / "restart.json")["phieStartMs"]
        times = times - times[0] + start
    return times


def electrodeSets(config, tissue, job):
    """Electrode layout, node sets and the averaging matrix onto phie.igb columns."""
    v = config["values"]
    layout = electrodeLayout(config)
    phieNodes = np.load(Path(job) / "phieNodes.npy")
    sets = electrodeNodeSets(tissue["positions"], phieNodes, layout["uniqueCentres"], v["electrodeRadiusMm"], v["electrodeLengthMm"])
    columnOfNode = np.full(len(tissue["positions"]), -1, np.int64)
    columnOfNode[phieNodes] = np.arange(len(phieNodes))
    return layout, sets, electrodeAveraging(sets, columnOfNode)


def extractElectrograms(config, job, tissue, windowMs, outputPath, block=400):
    """Unipolar phi at every electrode for the analysis frames -> npz (uni: placements x 16 x frames, mV)."""
    layout, sets, averaging = electrodeSets(config, tissue, job)
    times = phieTimes(job)
    frames = frameSlice(times, *analysisTimes(config, windowMs))
    chunks = []
    for first in range(frames.start, frames.stop, block):
        part = readIgb(Path(job) / "phie.igb", slice(first, min(first + block, frames.stop)))   # (frames, output nodes)
        chunks.append(averaging @ part.T.astype(float))                                      # (unique electrodes, frames)
    unique = np.concatenate(chunks, axis=1)
    sizes = np.array([len(s) for s in sets])
    np.savez_compressed(outputPath, uni=unique[layout["electrodeIndex"]].astype(np.float32), timeMs=times[frames],
                        centres=layout["centres"], placements=layout["placements"], uniqueCentres=layout["uniqueCentres"],
                        electrodeIndex=layout["electrodeIndex"], nodeSetNodes=np.concatenate(sets), nodeSetSizes=sizes)
    return dict(uniqueElectrodes=len(sets), nodesPerElectrodeMin=int(sizes.min()), nodesPerElectrodeMax=int(sizes.max()))


def nodeSetsFromArchive(archive):
    """The electrode node sets saved by extractElectrograms, as a list of arrays."""
    return np.split(archive["nodeSetNodes"], np.cumsum(archive["nodeSetSizes"])[:-1])


def runElectrograms(config, jobs, tissues, windowMs, folder):
    folder = Path(folder)
    result = {c: extractElectrograms(config, job, tissues[c], windowMs, folder / f"{c}.npz") for c, job in jobs.items()}
    saveJson(folder / "electrodes.json", result)
    return result
