"""Unipolar electrograms at every electrode from openCARP's lead-field module (FM-14, FM-23, FM-35).

An electrode is the set of bath mesh nodes inside its cylinder (efov.geometry); openCARP spreads its
unit lead-field current over that set by volume, so its unipolar signal is the volume-weighted mean
phi_e over the set, referenced to the grounded bath corner. openCARP writes the signal of every
electrode, V(t) = Z . (K_i Vm(t)), from the monodomain vm.igb (efov.leadfield). This module takes the
analysis frames from those traces and checks them against -Z^T S_i Vm formed from the same lead fields
with the sigma_i tensors the ground truth uses (CONSISTENCY_TOLERANCE).

Frames read: from (earliest placement window - 2 x filter history) to (latest placement window end +
trailingContextMs + resampleMarginMs). The second filter history serves the filter-history check; the
trailing context gives the wavelet transform signal after every window (FM-36); the margin keeps the
resampling filter's end effect outside the analysed segment (FM-29).
"""
from pathlib import Path

import numpy as np

from .config import beatCount, cycleLengthMs, fingerprint, slabSizeMm
from .geometry import electrodeLayout, electrodeNodeSets
from .io import igbTimes, readIgb, readVertices, saveJson
from .leadfield import ContributionModel, leadFieldsAndTraces, loadLeadField
from .meshbuild import catheterSideNodes


def analysisTimes(config, windows, trailingContext=True):
    """(start, end) in absolute ms of the frames to read for placement windows (P, 2) (ms after the last stimulus).

    trailingContext: include trailingContextMs after the latest window (electrograms, for the wavelet
    transform; FM-36). The ground truth passes False: it applies only the causal band-pass, which
    nothing after a window changes, so it does not read those Vm frames.
    """
    windows = np.atleast_2d(np.asarray(windows, float))
    a = config["fixed"]["analysis"]
    beatStart = (beatCount(config) - 1) * cycleLengthMs(config)
    trailing = a["trailingContextMs"] if trailingContext else 0.0
    end = beatStart + windows[:, 1].max() + trailing + a["resampleMarginMs"]
    if end > beatCount(config) * cycleLengthMs(config) + 1e-9:
        raise ValueError(f"The latest window ends {windows[:, 1].max():g} ms after the last stimulus; with {trailing:g} ms of trailing context "
                         f"and the {a['resampleMarginMs']:g} ms resampling margin it runs past the end of the simulation")
    return beatStart + windows[:, 0].min() - 2 * a["filterHistoryMs"], end


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


CONSISTENCY_TOLERANCE = 1e-3     # relative L2 difference, openCARP trace vs -Z^T S_i Vm (float32 files)
CONSISTENCY_ELECTRODES = 8


def extractElectrograms(config, job, tissue, windows, leadFieldArchive, layout, sets, outputPath):
    """Unipolar phi at every electrode for the analysis frames (openCARP traces) -> npz (uni: placements x 16 x frames, mV).

    Checks CONSISTENCY_ELECTRODES electrodes, spread over the node sets, against -Z^T S_i Vm from the same
    lead fields; stops if any differs by more than CONSISTENCY_TOLERANCE (relative L2 over the analysis frames).
    """
    stored = np.load(leadFieldArchive)
    startMs, endMs = analysisTimes(config, windows)
    times = stored["timeMs"]
    keep = (times >= startMs - 1e-9) & (times <= endMs + 1e-9)
    unique = stored["traces"][:, keep].astype(np.float64)
    times = times[keep]
    files = [str(f) for f in stored["files"]]
    model = ContributionModel(config, tissue)
    vm, vmTimes = readVmFrames(job, startMs, endMs)
    if len(vmTimes) != len(times) or np.abs(vmTimes - times).max() > 1e-6:
        raise ValueError(f"{job}: Vm frames and openCARP electrogram samples differ")
    checks = []
    for k in np.unique(np.linspace(0, len(sets) - 1, min(CONSISTENCY_ELECTRODES, len(sets))).round().astype(int)):
        reference = model.nodalWeights(loadLeadField(files[k])) @ vm.astype(np.float64).T
        scale = float(unique[k] @ reference / (reference @ reference))
        checks.append(dict(electrode=int(k), relativeDifference=float(np.linalg.norm(unique[k] - reference) / np.linalg.norm(reference)),
                           fittedScale=scale))
    worst = max(c["relativeDifference"] for c in checks)
    if worst > CONSISTENCY_TOLERANCE:
        raise RuntimeError(f"openCARP electrograms differ from -Z^T S_i Vm by up to {worst:.3g} (fitted scales "
                           f"{[round(c['fittedScale'], 6) for c in checks]}): openCARP's K_i or sign convention is not the "
                           "sigma_i source the ground truth uses; see docs/decision_log.md FM-35")
    sizes = np.array([len(s) for s in sets])
    np.savez_compressed(outputPath, uni=unique[layout["electrodeIndex"]].astype(np.float32), timeMs=times,
                        centres=layout["centres"], placements=layout["placements"], uniqueCentres=layout["uniqueCentres"],
                        electrodeIndex=layout["electrodeIndex"], nodeSetNodes=np.concatenate(sets), nodeSetSizes=sizes,
                        leadFieldFiles=np.array(files))
    return dict(uniqueElectrodes=len(sets), nodesPerElectrodeMin=int(sizes.min()), nodesPerElectrodeMax=int(sizes.max()),
                frames=int(len(times)), consistency=checks, consistencyMax=worst)


def nodeSetsFromArchive(archive):
    """The electrode node sets saved by extractElectrograms, as a list of arrays."""
    return np.split(archive["nodeSetNodes"], np.cumsum(archive["nodeSetSizes"])[:-1])


def runLeadFields(config, jobs, tissues, cacheRoot, folder, processCount=1, log=print):
    """Lead field and openCARP electrogram of every distinct electrode of this run, per condition (FM-35).

    Lead fields are cached in <cacheRoot>/<conductor fingerprint>/<condition>/; <folder>/<condition>.npz holds
    the node sets, the cached files this run uses and the electrogram traces (all Vm frames).
    """
    folder, key = Path(folder), fingerprint(config, "conductor")
    result = dict(conductorFingerprint=key, cache=str(Path(cacheRoot) / key), conditions={})
    for condition, job in jobs.items():
        layout, sets = electrodeSets(config, tissues[condition])
        paths, constructed, times, traces = leadFieldsAndTraces(
            config, tissues[condition], readVertices(Path(job) / "ground.vtx"), sets, Path(job) / "vm.igb",
            Path(cacheRoot) / key / condition, folder / condition, processCount, log)
        np.savez(folder / f"{condition}.npz", nodeSetNodes=np.concatenate(sets), nodeSetSizes=[len(s) for s in sets],
                 files=np.array([str(p) for p in paths]), timeMs=times, traces=traces.astype(np.float32))
        result["conditions"][condition] = dict(electrodes=len(sets), constructedNow=constructed,
                                               loadedFromCache=int(constructed == 0))
    saveJson(folder / "leadFields.json", result)
    return result


def runElectrograms(config, jobs, tissues, windows, leadFieldFolder, folder):
    folder = Path(folder)
    result = {}
    for condition, job in jobs.items():
        layout, sets = electrodeSets(config, tissues[condition])
        archive = Path(leadFieldFolder) / f"{condition}.npz"
        if not np.array_equal(np.load(archive)["nodeSetNodes"], np.concatenate(sets)):
            raise ValueError(f"{condition}: electrode node sets differ from those of the leadFields stage")
        result[condition] = extractElectrograms(config, job, tissues[condition], windows, archive, layout, sets,
                                                folder / f"{condition}.npz")
    saveJson(folder / "electrodes.json", result)
    return result
