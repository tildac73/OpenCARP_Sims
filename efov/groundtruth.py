"""Ground truth: the exact contributing-tissue field of a recording (DL-2026-11-01-02).

For a recording with electrode weights w and finite electrodes sampled at surface points, the
recorded potential is a sum of element contributions (reciprocity, exact for the discrete problem):

    phi(t) = sum_e c_e(t),     c_e(t) = coeff_e . Vm_e(t)

with coeff from the lead field of the recording (forward.ExtracellularSolver.elementCoefficients),
using the same sigma_m source, conductivities and ground as openCARP. The share of element e over
the measurement window is its signed part of the signal energy

    s_e = sum_t c_e(t) phi(t) / sum_t phi(t)^2,     sum_e s_e = 1 exactly.

Nothing is cut off and cancellation is kept (s_e < 0 where an element opposes the signal), so shares
are not probabilities. Differential recordings use band-passed Vm (the filter is linear and commutes
with the field). Outputs per recording: shares summed into fixed voxels over the tissue (comparable
between meshes) and descriptive cumulative-share curves against lateral distance and depth.

Checks (acceptance.implementation), enforced: shares sum to 1; the field reproduces the measured
clean electrogram (relative residual), with scale equal to the calibration gain.
"""
from pathlib import Path

import numpy as np
import pandas as pd

from .config import slabSizeMm
from .forward import BATH_TAG, pointInterpolator
from .geometry import electrodeLayout
from .io import igbTimes, readIgb, saveJson
from .measurements import measurementSlices
from .recordings import applyFilter, bandPass


def recordingRow(interpolator, sampleIndex, weights):
    """Interpolation row of a recording: sum over electrodes of weight x mean of the electrode's sample rows.

    interpolator: (sample points x nodes) P1 matrix; sampleIndex: (16, samples) rows of this placement.
    """
    used = np.flatnonzero(np.abs(weights) > 1e-12)
    if not len(used):
        raise ValueError("Recording has no non-zero electrode weight")
    row = sum(weights[e] * interpolator[sampleIndex[e]].mean(axis=0) for e in used)
    return np.asarray(row).ravel()


def shares(model, row, tissueVm):
    """Element shares s_e (E,), the reconstructed signal phi(t) and sum(s_e)."""
    z = model.solver.leadField(row[None, :])[:, 0]
    coefficients = model.solver.elementCoefficients(z)                # (E, 4), zero in the bath
    nodal = np.zeros(len(model.positions))
    np.add.at(nodal, model.elements, coefficients)
    vm = model.fullVm(tissueVm)                                        # (N, T)
    phi = nodal @ vm
    energy = float(phi @ phi)
    if energy <= 0:
        raise ValueError("Recording has zero signal energy in the window")
    u = vm @ phi
    elementShares = np.einsum("ej,ej->e", coefficients, u[model.elements]) / energy
    return elementShares, phi, float(elementShares.sum())


def voxelGrid(config):
    size, voxel = slabSizeMm(config), config["groundTruth"]["voxelMm"]
    return dict(lower=-size / 2, voxelMm=float(voxel), shape=tuple(int(n) for n in np.round(size / voxel)))


def voxelise(model, elementShares, grid):
    """Sum tissue-element shares into voxels by element centre (sum preserved)."""
    tissue = model.tags != BATH_TAG
    index = np.floor((model.centres[tissue] - grid["lower"]) / grid["voxelMm"] + 1e-9).astype(int)
    index = np.clip(index, 0, np.asarray(grid["shape"]) - 1)
    flat = np.ravel_multi_index(index.T, grid["shape"])
    return np.bincount(flat, weights=elementShares[tissue], minlength=int(np.prod(grid["shape"]))).reshape(grid["shape"])


def cumulativeCurves(model, elementShares, anchorMm, stepMm):
    """Cumulative share within lateral distance d of the anchor, and within depth d of the catheter-facing surface."""
    tissue = model.tags != BATH_TAG
    centres, values = model.centres[tissue], elementShares[tissue]
    keys = dict(lateral=np.linalg.norm(centres[:, :2] - np.asarray(anchorMm)[:2], axis=1),
                depth=np.clip(model.tissueTopMm - centres[:, 2], 0, None))
    out = {}
    for name, key in keys.items():
        edges = np.arange(0, key.max() + stepMm, stepMm)
        order = np.argsort(key)
        cumulative = np.cumsum(values[order])
        counts = np.searchsorted(key[order], edges, side="right")
        out[name] = np.where(counts > 0, cumulative[np.maximum(counts - 1, 0)], 0.0)
    return out


def fieldDistance(fieldA, fieldB):
    """L1 distance between two voxel share fields relative to the reference B: sum|A - B| / sum|B|."""
    return float(np.abs(np.asarray(fieldA) - np.asarray(fieldB)).sum() / np.abs(np.asarray(fieldB)).sum())


def runGroundTruth(config, jobs, models, measurementsFolder, gain, windowMs, folder):
    """Contribution fields of the ground-truth recordings (clean, centres in groundTruth.centresYmm, both conditions)."""
    folder, measurementsFolder = Path(folder), Path(measurementsFolder)
    a, gt, gate = config["analysis"], config["groundTruth"], config["acceptance"]["implementation"]
    grid = voxelGrid(config)
    layout = electrodeLayout(config)
    rows, fields, ids, lateral, depth = [], [], [], [], []
    for condition, model in models.items():
        points = pd.read_csv(measurementsFolder / f"{condition}Points.csv")
        points = points[(points.noiseRepeat == 0) & np.isclose(points.centreYmm.to_numpy()[:, None], gt["centresYmm"]).any(axis=1)]
        interpolator = pointInterpolator(model.positions, model.elements, layout["samplePoints"])
        traces = np.load(measurementsFolder / f"{condition}Traces.npz")["cleanMv"]
        times = igbTimes(Path(jobs[condition]) / "vm.igb")
        raw, measure = measurementSlices(times, config, windowMs)
        vm = readIgb(Path(jobs[condition]) / "vm.igb", raw).T.astype(float)            # (tissue nodes, frames)
        sections = bandPass(a["differentialBandHz"], 1000.0 / float(np.diff(times[:2])[0]), a["filterOrder"])
        voltages = dict(uni=vm[:, measure], differential=applyFilter(sections, vm)[:, measure])
        del vm
        for record in points.to_dict("records"):
            weights = np.array([record[f"weightE{k}"] for k in range(16)])
            row = recordingRow(interpolator, layout["sampleIndex"][record["placement"]], weights)
            elementShares, phi, total = shares(model, row, voltages["uni" if record["kind"] == "uni" else "differential"])
            measured = traces[record["traceIndex"]].astype(float)
            scale = float(measured @ phi / (phi @ phi))
            residual = float(np.linalg.norm(measured - scale * phi) / np.linalg.norm(measured))
            curves = cumulativeCurves(model, elementShares, [record["anchorXmm"], record["anchorYmm"]], gt["curveStepMm"])
            fields.append(voxelise(model, elementShares, grid).astype(np.float32))
            ids.append(record["recordingId"]); lateral.append(curves["lateral"]); depth.append(curves["depth"])
            rows.append(dict(recordingId=record["recordingId"], condition=condition, kind=record["kind"], label=record["label"],
                             gapMm=record["gapMm"], centreYmm=record["centreYmm"], anchorXmm=record["anchorXmm"],
                             anchorYmm=record["anchorYmm"], shareSum=total, shareAbsSum=float(np.abs(elementShares).sum()),
                             reconstructionResidual=residual, scaleOverGain=scale / gain))
    pad = lambda values: np.array([np.pad(v, (0, max(map(len, values)) - len(v)), constant_values=np.nan) for v in values])
    np.savez_compressed(folder / "contributionField.npz", recordingId=np.asarray(ids), fields=np.asarray(fields),
                        voxelMm=grid["voxelMm"], lowerMm=grid["lower"], lateralCumulativeShare=pad(lateral),
                        depthCumulativeShare=pad(depth), curveStepMm=gt["curveStepMm"], source="sigmaM")
    table = pd.DataFrame(rows)
    table.to_csv(folder / "summary.csv", index=False)
    checks = dict(recordings=len(table),
                  shareSumMaxDeviation=float((table.shareSum - 1).abs().max()),
                  reconstructionResidualMax=float(table.reconstructionResidual.max()),
                  scaleOverGainMaxDeviation=float((table.scaleOverGain - 1).abs().max()))
    checks["passed"] = bool(checks["recordings"] > 0
                            and checks["shareSumMaxDeviation"] <= gate["shareSumDeviation"]
                            and checks["reconstructionResidualMax"] <= gate["reconstructionResidual"]
                            and checks["scaleOverGainMaxDeviation"] <= gate["scaleOverGainDeviation"])
    saveJson(folder / "checks.json", checks)
    return checks
