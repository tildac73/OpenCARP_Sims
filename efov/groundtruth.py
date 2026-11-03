"""Ground truth: the exact contributing-tissue field of each recording (DL-2026-11-01-02, FM-14 to FM-16, FM-21).

For a recording with electrode weights w, the recorded potential is a sum of element contributions
(reciprocity, exact for the discrete problem):

    phi(t) = sum_e c_e(t),   c_e(t) = coeff_e . Vm_e(t),   coeff from the recording's lead field
    z = sum_k w_k z_k (efov.leadfield), with the sigma_m source of openCARP's pseudo-bidomain phi_e.

The share of element e over the measurement window is its signed part of the signal energy

    s_e = sum_t c_e(t) phi(t) / sum_t phi(t)^2,     sum_e s_e = 1 exactly.

Nothing is cut off and cancellation is kept (s_e < 0 where an element opposes the signal). Vm goes
through the same acquisition chain as the electrograms (anti-alias, decimation; band-pass for
bipoles and omnipoles), because the chain is linear and commutes with the field.

Per recording (all recordings at the ground-truth placements, clean signals):
  * classes (FM-15): near-field / far-field (lateral distance from the anchor <= nearFieldRadiusMm)
    x endocardial / epicardial half (depth < depthSplitFraction x wall); lesion elements reported
    separately. Signed share and |c|-mass fraction (sum_t |c_e(t)|, no cancellation) per class;
  * depth profile: one value per node layer of the mesh (signed share and |c|-mass);
  * cumulative signed share against lateral distance;
  * shape descriptors (secondary, descriptive): for each level p, the smallest set of elements
    holding p of sum |s_e| (elements ranked by |s_e| / volume), its volume, x and y extent, maximum
    depth, |s|-weighted centroid offset, in-plane eccentricity and major-axis angle.
Full element fields (s_e on every tissue element) for fixed.groundTruth.fieldRecordings, as npz and
legacy VTK files for ParaView.
Checks (stop the run if they fail): shares sum to 1; summed contributions reproduce the clean
electrogram the pipeline measured (relative residual), with scale = calibration gain.
"""
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.signal import sosfilt

from .config import beatCount, cycleLengthMs, groundTruthCentresYmm, slabSizeMm, spacingMm
from .electrograms import nodeSetsFromArchive
from .io import igbTimes, readIgb, readVertices, saveJson, writeVtkCells
from .leadfield import ContributionModel, PythonLeadFieldSolver, openCarpLeadField
from .measurements import acquire, differentialFilter, measurementSlices

CLASSES = ("nearEndocardial", "nearEpicardial", "farEndocardial", "farEpicardial")


# ----------------------------------------------------------------- Vm through the acquisition chain
def readVmFrames(job, startMs, endMs):
    """Vm (frames, tissue nodes) and times for [startMs, endMs] from vm.igb, or from vm_analysis.npz after compaction."""
    job = Path(job)
    if (job / "vm.igb").exists():
        times = igbTimes(job / "vm.igb")
        frames = slice(int(np.searchsorted(times, startMs - 1e-9)), int(np.searchsorted(times, endMs + 1e-9)))
        return readIgb(job / "vm.igb", frames), times[frames]
    archive = np.load(job / "vm_analysis.npz")
    times = archive["timeMs"]
    if times[0] > startMs + 1e-9 or times[-1] < endMs - 1e-9:
        raise ValueError(f"{job}: compacted Vm covers {times[0]}-{times[-1]} ms, not {startMs}-{endMs} ms")
    keep = (times >= startMs - 1e-9) & (times <= endMs + 1e-9)
    return archive["vm"][keep], times[keep]


def processedVm(config, job, windowMs, nodeCount, block=40000):
    """Vm of every tissue node through the acquisition chain, inside the window: {'uni', 'differential'} (nodes, samples).

    Vm is read once (frames x nodes, float32; ~4 GB on the production mesh) and filtered in node blocks.
    """
    beatStart = (beatCount(config) - 1) * cycleLengthMs(config)
    history = config["fixed"]["analysis"]["filterHistoryMs"]
    vm, times = readVmFrames(job, beatStart + windowMs[0] - 2 * history, beatStart + windowMs[1])
    if vm.shape[1] != nodeCount:
        raise ValueError(f"{job}: Vm has {vm.shape[1]} nodes, the model {nodeCount}")
    sections = differentialFilter(config)
    uni, diff = [], []
    for first in range(0, nodeCount, block):
        acquired, acquisitionTimes = acquire(config, vm[:, first:first + block].T.astype(float), times)    # (nodes, samples)
        raw, measure = measurementSlices(acquisitionTimes, config, windowMs)
        uni.append(acquired[:, raw][:, measure].astype(np.float32))
        diff.append(sosfilt(sections, acquired[:, raw], axis=-1)[:, measure].astype(np.float32))
    return dict(uni=np.concatenate(uni), differential=np.concatenate(diff))


# ----------------------------------------------------------------- shares and descriptors
def shares(model, coefficients, vm):
    """Element shares s_e, |c|-mass per element, the reconstructed signal phi(t) and sum(s_e)."""
    nodal = np.zeros(len(model.tissueNodes))
    np.add.at(nodal, model.localElements, coefficients)
    phi = nodal @ vm                                                           # (samples,)
    energy = float(phi @ phi)
    if energy <= 0:
        raise ValueError("Recording has zero signal energy in the window")
    u = vm @ phi
    elementShares = np.einsum("ej,ej->e", coefficients, u[model.localElements]) / energy
    mass = np.zeros(len(model.elements))
    for first in range(0, len(model.elements), 200000):
        rows = slice(first, first + 200000)
        mass[rows] = np.abs(np.einsum("ej,ejt->et", coefficients[rows], vm[model.localElements[rows]])).sum(axis=1)
    return elementShares, mass, phi, float(elementShares.sum())


def elementGeometry(model, config, anchor):
    """Lateral distance from the anchor, depth below the catheter-facing surface, layer index (per element)."""
    wall, dz = slabSizeMm(config)[2], spacingMm(config)[2]
    lateral = np.linalg.norm(model.centres[:, :2] - np.asarray(anchor)[:2], axis=1)
    depth = np.clip(wall / 2 - model.centres[:, 2], 0, wall)
    layer = np.clip(np.floor(depth / dz).astype(int), 0, int(round(wall / dz)) - 1)
    return lateral, depth, layer


def classSummary(model, config, elementShares, mass, lateral, depth):
    g, wall = config["fixed"]["groundTruth"], slabSizeMm(config)[2]
    near = lateral <= g["nearFieldRadiusMm"]
    endo = depth < g["depthSplitFraction"] * wall
    masks = dict(nearEndocardial=near & endo, nearEpicardial=near & ~endo, farEndocardial=~near & endo, farEpicardial=~near & ~endo)
    total = mass.sum()
    out = {}
    for name, mask in masks.items():
        out[f"share_{name}"] = float(elementShares[mask].sum())
        out[f"mass_{name}"] = float(mass[mask].sum() / total)
    lesion = model.tags == 5
    out.update(share_lesionElements=float(elementShares[lesion].sum()), mass_lesionElements=float(mass[lesion].sum() / total))
    return out


def shapeDescriptors(model, elementShares, lateral, depth, anchor, levels):
    """Smallest element sets holding each level of sum |s_e| (ranked by |s_e| per volume) and their shape."""
    magnitude = np.abs(elementShares)
    order = np.argsort(-(magnitude / model.volumes))
    cumulative = np.cumsum(magnitude[order]) / magnitude.sum()
    out = {}
    for level in levels:
        chosen = order[: int(np.searchsorted(cumulative, level)) + 1]
        weights = magnitude[chosen] / magnitude[chosen].sum()
        xy = model.centres[chosen, :2]
        centroid = weights @ xy
        covariance = np.cov((xy - centroid).T, aweights=weights) if len(chosen) > 2 else np.eye(2)
        eigenvalues, vectors = np.linalg.eigh(covariance)
        major = vectors[:, 1]
        tag = f"p{int(round(level * 100))}"
        out.update({f"{tag}_volumeMm3": float(model.volumes[chosen].sum()),
                    f"{tag}_extentXmm": float(np.ptp(xy[:, 0])), f"{tag}_extentYmm": float(np.ptp(xy[:, 1])),
                    f"{tag}_maximumDepthMm": float(depth[chosen].max()), f"{tag}_maximumLateralMm": float(lateral[chosen].max()),
                    f"{tag}_centroidOffsetXmm": float(centroid[0] - anchor[0]), f"{tag}_centroidOffsetYmm": float(centroid[1] - anchor[1]),
                    f"{tag}_eccentricity": float(np.sqrt(max(0.0, 1 - eigenvalues[0] / eigenvalues[1]))) if eigenvalues[1] > 0 else 0.0,
                    f"{tag}_majorAxisAngleDeg": float(np.degrees(np.arctan2(major[1], major[0])) % 180)})
    return out


def cumulativeLateral(elementShares, lateral, stepMm):
    edges = np.arange(0, lateral.max() + stepMm, stepMm)
    order = np.argsort(lateral)
    cumulative = np.cumsum(elementShares[order])
    counts = np.searchsorted(lateral[order], edges, side="right")
    return np.where(counts > 0, cumulative[np.maximum(counts - 1, 0)], 0.0)


# ----------------------------------------------------------------- lead fields
def leadFields(config, condition, tissue, job, electrodes, folder, processCount):
    """Compute (once) and store the lead field (all nodes, mV/uA) of each needed electrode; returns the npy paths."""
    method = config["fixed"]["groundTruth"]["leadFieldMethod"]
    folder = Path(folder) / condition
    folder.mkdir(parents=True, exist_ok=True)
    ground = readVertices(Path(job) / "ground.vtx")
    solver = PythonLeadFieldSolver(config, tissue, ground) if method == "python" else None
    result = {}
    for index, nodes in electrodes.items():
        path = folder / f"electrode{index:04d}.npy"
        if not path.exists():
            if method == "openCARP":
                z = openCarpLeadField(config, tissue["meshBase"], ground, nodes, len(tissue["positions"]),
                                      folder / f"electrode{index:04d}", processCount)
            else:
                z = solver.leadField(nodes)
            np.save(path, z.astype(np.float64))
        result[index] = path
    return result


# ----------------------------------------------------------------- stage
def runGroundTruth(config, jobs, tissues, electrogramsFolder, measurementsFolder, gain, windowMs, folder, processCount=1):
    """Contribution fields of every recording at the ground-truth placements (clean signals, both conditions)."""
    folder, g = Path(folder), config["fixed"]["groundTruth"]
    gate = config["fixed"]["acceptance"]["groundTruth"]
    centres = groundTruthCentresYmm(config)
    rows, depthRows, lateralRows, fieldValues = [], [], [], {}
    for condition, job in jobs.items():
        tissue = tissues[condition]
        model = ContributionModel(config, tissue)
        archive = np.load(Path(electrogramsFolder) / f"{condition}.npz")
        nodeSets = nodeSetsFromArchive(archive)
        points = pd.read_csv(Path(measurementsFolder) / f"{condition}Points.csv")
        points = points[(points.noiseRepeat == 0) & np.isclose(points.centreYmm.to_numpy()[:, None], centres).any(axis=1)]
        placements = sorted(points.placement.unique())
        needed = sorted({int(archive["electrodeIndex"][p, k]) for p in placements for k in range(16)})
        fields = leadFields(config, condition, tissue, job, {k: nodeSets[k] for k in needed}, folder / "leadFields", processCount)
        vm = processedVm(config, job, windowMs, len(model.tissueNodes))
        traces = np.load(Path(measurementsFolder) / f"{condition}Traces.npz")["cleanMv"]
        fieldLabels = {(kind, label) for kind, labels in g["fieldRecordings"].items() for label in labels}
        for p in placements:
            coefficientCache = {}                       # per placement only: ~80 MB per electrode on the full mesh
            for k in range(16):
                index = int(archive["electrodeIndex"][p, k])
                if index not in coefficientCache:
                    coefficientCache[index] = model.coefficients(np.load(fields[index]))
            for record in points[points.placement == p].to_dict("records"):
                weights = np.array([record[f"weightE{k}"] for k in range(16)])
                coefficients = sum(w * coefficientCache[int(archive["electrodeIndex"][p, k])]
                                   for k, w in enumerate(weights) if abs(w) > 1e-12)
                signal = vm["uni" if record["kind"] == "uni" else "differential"]
                elementShares, mass, phi, total = shares(model, coefficients, signal)
                measured = traces[record["traceIndex"]].astype(float)
                scale = float(measured @ phi / (phi @ phi))
                residual = float(np.linalg.norm(measured - scale * phi) / np.linalg.norm(measured))
                anchor = [record["anchorXmm"], record["anchorYmm"]]
                lateral, depth, layer = elementGeometry(model, config, anchor)
                layers = int(round(slabSizeMm(config)[2] / spacingMm(config)[2]))
                depthRows.append(np.concatenate([np.bincount(layer, elementShares, layers), np.bincount(layer, mass, layers) / mass.sum()]))
                lateralRows.append(cumulativeLateral(elementShares, lateral, g["curveStepMm"]))
                rows.append(dict(recordingId=record["recordingId"], condition=condition, kind=record["kind"], label=record["label"],
                                 gapMm=record["gapMm"], centreYmm=record["centreYmm"], anchorXmm=anchor[0], anchorYmm=anchor[1],
                                 distanceInsideMm=record["distanceInsideMm"], shareSum=total,
                                 shareAbsSum=float(np.abs(elementShares).sum()), reconstructionResidual=residual,
                                 scaleOverGain=scale / gain, **classSummary(model, config, elementShares, mass, lateral, depth),
                                 **shapeDescriptors(model, elementShares, lateral, depth, anchor, g["shapeLevels"])))
                if (record["kind"], record["label"]) in fieldLabels:
                    fieldValues.setdefault(condition, {})[record["recordingId"]] = elementShares.astype(np.float32)
        writeFields(model, condition, fieldValues.get(condition, {}), points, folder)
    table = pd.DataFrame(rows)
    table.to_csv(folder / "summary.csv", index=False)
    pad = lambda values: np.array([np.pad(v, (0, max(map(len, values)) - len(v)), constant_values=np.nan) for v in values])
    np.savez_compressed(folder / "profiles.npz", recordingId=table.recordingId.to_numpy().astype(str), depthShareAndMass=np.asarray(depthRows),
                        lateralCumulativeShare=pad(lateralRows), curveStepMm=g["curveStepMm"], layerThicknessMm=spacingMm(config)[2])
    checks = dict(recordings=len(table), shareSumMaxDeviation=float((table.shareSum - 1).abs().max()),
                  reconstructionResidualMax=float(table.reconstructionResidual.max()),
                  scaleOverGainMaxDeviation=float((table.scaleOverGain - 1).abs().max()),
                  leadFieldMethod=g["leadFieldMethod"])
    checks["passed"] = bool(checks["recordings"] > 0 and checks["shareSumMaxDeviation"] <= gate["shareSumDeviation"]
                            and checks["reconstructionResidualMax"] <= gate["reconstructionResidual"]
                            and checks["scaleOverGainMaxDeviation"] <= gate["scaleOverGainDeviation"])
    saveJson(folder / "checks.json", checks)
    return checks


def writeFields(model, condition, values, points, folder):
    """Full element fields of the field recordings: npz (all) and one VTK file per gap (ParaView)."""
    if not values:
        return
    ids = sorted(values)
    np.savez_compressed(Path(folder) / f"fields_{condition}.npz", recordingId=np.asarray(ids),
                        shares=np.stack([values[i] for i in ids]), elements=model.elements, tags=model.tags)
    meta = points.set_index("recordingId")
    for gap in sorted(meta.loc[ids].gapMm.unique()):
        arrays = {"tag": model.tags.astype(float)}
        for i in ids:
            if meta.loc[i].gapMm == gap:
                arrays[f"{meta.loc[i].kind}_{meta.loc[i].label}_y{meta.loc[i].centreYmm:+.1f}".replace("|", "or")] = values[i]
        writeVtkCells(Path(folder) / f"shares_{condition}_gap{gap:g}mm.vtk", model.positions, model.elements, arrays)
