"""Unipolar electrograms at every electrode, by reciprocity from the lead fields (FM-14, FM-23).

An electrode is the set of bath mesh nodes inside its cylinder (efov.geometry); its unipolar signal
is the mean phi_e over that set, referenced to the grounded bath corner. With the electrode's lead
field z (efov.leadfield), phi(t) = w . Vm(t) with w = -S_i z on the tissue nodes: exact for the
discrete bidomain extracellular problem with the sigma_i source and openCARP's monodomain Vm.

Frames read: from (earliest placement window - 2 x filter history) to (latest placement window end +
resampleMarginMs). The second filter history serves the filter-history check; the margin keeps the
resampling filter's end effect outside every window (FM-29).
"""
from pathlib import Path

import numpy as np

from .config import beatCount, cycleLengthMs, fingerprint, slabSizeMm
from .geometry import electrodeLayout, electrodeNodeSets
from .io import igbTimes, readIgb, readVertices, saveJson
from .leadfield import ContributionModel, electrodeLeadFields
from .meshbuild import catheterSideNodes


def analysisTimes(config, windows):
    """(start, end) in absolute ms of the Vm frames to read for placement windows (P, 2) (ms after the last stimulus)."""
    windows = np.atleast_2d(np.asarray(windows, float))
    a = config["fixed"]["analysis"]
    beatStart = (beatCount(config) - 1) * cycleLengthMs(config)
    return beatStart + windows[:, 0].min() - 2 * a["filterHistoryMs"], beatStart + windows[:, 1].max() + a["resampleMarginMs"]


def readVmFrames(job, startMs, endMs):
    """Vm (frames, tissue nodes) and times for [startMs, endMs] from vm.igb, or from vm_analysis.npz after compaction."""
    job = Path(job)
    if (job / "vm.igb").exists():
        times = igbTimes(job / "vm.igb")
        if startMs < times[0] - 1e-9 or endMs > times[-1] + 1e-9:
            raise ValueError(f"Frames {startMs}-{endMs} ms lie outside the recording ({times[0]}-{times[-1]} ms)")
        frames = slice(int(np.searchsorted(times, startMs - 1e-9)), int(np.searchsorted(times, endMs + 1e-9)))
        return readIgb(job / "vm.igb", frames), times[frames]
    archive = np.load(job / "vm_analysis.npz")
    times = archive["timeMs"]
    if times[0] > startMs + 1e-9 or times[-1] < endMs - 1e-9:
        raise ValueError(f"{job}: compacted Vm covers {times[0]}-{times[-1]} ms, not {startMs}-{endMs} ms")
    keep = (times >= startMs - 1e-9) & (times <= endMs + 1e-9)
    return archive["vm"][keep], times[keep]


def electrodeSets(config, tissue):
    """Electrode layout and the node set of every distinct electrode position."""
    v = config["values"]
    layout = electrodeLayout(config)
    candidates = catheterSideNodes(tissue["positions"], tissue["elements"], tissue["tags"], slabSizeMm(config)[2])
    sets = electrodeNodeSets(tissue["positions"], candidates, layout["uniqueCentres"], v["electrodeRadiusMm"], v["electrodeLengthMm"])
    return layout, sets


def extractElectrograms(config, job, tissue, windows, leadFieldPaths, layout, sets, outputPath, block=64, frameBlock=200):
    """Unipolar phi at every electrode for the analysis frames -> npz (uni: placements x 16 x frames, mV).

    Vm stays float32 in memory (frames x tissue nodes); products are formed in float64 in blocks of
    electrodes and frames. The weights of an electrode sum to zero (S_i annihilates constants), so the
    resting offset of Vm cancels exactly; float64 keeps that cancellation accurate.
    """
    model = ContributionModel(config, tissue)
    vm, times = readVmFrames(job, *analysisTimes(config, windows))
    if vm.shape[1] != len(model.tissueNodes):
        raise ValueError(f"{job}: Vm has {vm.shape[1]} nodes, the model {len(model.tissueNodes)}")
    unique = np.empty((len(sets), len(times)))
    for first in range(0, len(sets), block):
        rows = range(first, min(first + block, len(sets)))
        weights = np.stack([model.nodalWeights(np.load(leadFieldPaths[k])) for k in rows])   # (block, tissue nodes)
        for start in range(0, len(times), frameBlock):
            frames = slice(start, min(start + frameBlock, len(times)))
            unique[first:first + len(rows), frames] = weights @ vm[frames].astype(np.float64).T
    sizes = np.array([len(s) for s in sets])
    np.savez_compressed(outputPath, uni=unique[layout["electrodeIndex"]].astype(np.float32), timeMs=times,
                        centres=layout["centres"], placements=layout["placements"], uniqueCentres=layout["uniqueCentres"],
                        electrodeIndex=layout["electrodeIndex"], nodeSetNodes=np.concatenate(sets), nodeSetSizes=sizes,
                        leadFieldFiles=np.array([str(p) for p in leadFieldPaths]))
    return dict(uniqueElectrodes=len(sets), nodesPerElectrodeMin=int(sizes.min()), nodesPerElectrodeMax=int(sizes.max()),
                frames=int(len(times)))


def nodeSetsFromArchive(archive):
    """The electrode node sets saved by extractElectrograms, as a list of arrays."""
    return np.split(archive["nodeSetNodes"], np.cumsum(archive["nodeSetSizes"])[:-1])


def runLeadFields(config, jobs, tissues, cacheRoot, folder, processCount=1, log=print):
    """Lead field of every distinct electrode of this run, per condition (FM-23).

    Cached in <cacheRoot>/<conductor fingerprint>/<condition>/; <folder>/<condition>.npz lists the
    node sets and the cached files this run uses.
    """
    folder, key = Path(folder), fingerprint(config, "conductor")
    result = dict(conductorFingerprint=key, cache=str(Path(cacheRoot) / key), conditions={})
    for condition, job in jobs.items():
        layout, sets = electrodeSets(config, tissues[condition])
        paths, computed = electrodeLeadFields(config, tissues[condition], readVertices(Path(job) / "ground.vtx"), sets,
                                              Path(cacheRoot) / key / condition, processCount, log)
        np.savez(folder / f"{condition}.npz", nodeSetNodes=np.concatenate(sets), nodeSetSizes=[len(s) for s in sets],
                 files=np.array([str(p) for p in paths]))
        result["conditions"][condition] = dict(electrodes=len(sets), computedNow=computed, reused=len(sets) - computed)
    saveJson(folder / "leadFields.json", result)
    return result


def runElectrograms(config, jobs, tissues, windows, leadFieldFolder, folder):
    folder = Path(folder)
    result = {}
    for condition, job in jobs.items():
        layout, sets = electrodeSets(config, tissues[condition])
        stored = np.load(Path(leadFieldFolder) / f"{condition}.npz")
        if not np.array_equal(stored["nodeSetNodes"], np.concatenate(sets)):
            raise ValueError(f"{condition}: electrode node sets differ from those of the leadFields stage")
        result[condition] = extractElectrograms(config, job, tissues[condition], windows, list(stored["files"]), layout, sets,
                                                folder / f"{condition}.npz")
    saveJson(folder / "electrodes.json", result)
    return result
