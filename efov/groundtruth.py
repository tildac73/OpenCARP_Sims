"""Ground truth: the exact contributing-tissue field of each recording (DL-2026-11-01-02, FM-14 to FM-16, FM-21).

For a recording with electrode weights w, the recorded potential is a sum of element contributions
(reciprocity, exact for the discrete problem):

    phi(t) = sum_e c_e(t),   c_e(t) = coeff_e . Vm_e(t),   coeff from the recording's lead field
    z = sum_k w_k z_k (efov.leadfield), with the sigma_i source (FM-23). These are the same lead
    fields the electrograms are computed from, so the fields describe exactly the measured signals.

The share of element e over the measurement window is its signed part of the signal energy

    s_e = sum_t c_e(t) phi(t) / sum_t phi(t)^2,     sum_e s_e = 1 exactly.

Nothing is cut off and cancellation is kept (s_e < 0 where an element opposes the signal). Vm goes
through the same acquisition chain as the electrograms (anti-alias, decimation; band-pass for
bipoles and omnipoles), because the chain is linear and commutes with the field.

Per recording (all recordings at the ground-truth placements, clean signals):
  * retained share (FM-37): signed share and |c|-mass of tissue with y >= boundary, the tissue that
    the lesion run leaves active. In the intact run, its fall with the recording's distance inside the
    lesion is the ground truth matched to the clinical boundary sweep; boundaryCrossings.csv gives the
    distances where it first reaches fixed.groundTruth.retainedShareLevels. The |c|-mass is the
    primary measure (FM-43). The signed share divides by the signal energy, so it is ill-conditioned
    when the recorded signal is nearly null (e.g. a bipole parallel to the wavefront): recordings with
    sum |s_e| above fixed.groundTruth.maximumShareAbsSum are flagged (signedShareIllConditioned) and
    their signed-share crossings are not reported. Two further properties of these curves follow from
    the model and are not results: for a wave parallel to the boundary, mirror symmetry puts the
    retained share of a symmetric recording at 0.5 at the boundary, so the 0.5 level carries no
    information there; unipolar curves depend on where the reference (ground) sits, here on the
    retained side of the boundary;
  * classes (FM-15): near-field / far-field (lateral distance from the anchor <= nearFieldRadiusMm)
    x endocardial / epicardial half (depth < depthSplitFraction x wall); lesion elements reported
    separately. Signed share and |c|-mass fraction (sum_t |c_e(t)|, no cancellation) per class;
  * depth profile: one value per node layer of the mesh (signed share and |c|-mass);
  * cumulative signed share against lateral distance;
  * shape descriptors (secondary, descriptive): for each level p, the smallest set of elements
    holding p of sum |s_e| (elements ranked by |s_e| / volume), its volume, x and y extent, maximum
    depth, |s|-weighted centroid offset, in-plane eccentricity and major-axis angle.
Full element fields (s_e on every tissue element) for fixed.groundTruth.fieldRecordings, as npz and
legacy VTK files for ParaView. Each placement uses its own measurement window (FM-28).
Check (stops the run if it fails): shares sum to 1. This is an algebraic identity of the definition
above, so it detects a broken implementation (mismatched element indexing); it is not a validation
of the forward model or of the ground truth.
"""
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.signal import sosfilt

from .config import groundTruthCentresYmm, slabSizeMm, spacingMm
from .electrograms import analysisTimes, readVmFrames
from .io import saveJson, writeVtkCells
from .leadfield import ContributionModel, loadLeadField
from .measurements import acquire, differentialFilter, placementIndices

CLASSES = ("nearEndocardial", "nearEpicardial", "farEndocardial", "farEpicardial")


# ----------------------------------------------------------------- Vm through the acquisition chain
def processedVm(config, job, windows, nodeCount, block=40000):
    """Vm of every tissue node through the acquisition chain, from the earliest window start to the latest end.

    Returns {'uni', 'differential'} (nodes, samples) float32, 'offsets' (first sample of each
    placement's window) and 'length'. Vm is read once (frames x nodes, float32) and filtered in node blocks.
    """
    vm, times = readVmFrames(job, *analysisTimes(config, windows, trailingContext=False))
    if vm.shape[1] != nodeCount:
        raise ValueError(f"{job}: Vm has {vm.shape[1]} nodes, the model {nodeCount}")
    sections = differentialFilter(config)
    uni, diff = [], []
    for first in range(0, nodeCount, block):
        acquired, acquisitionTimes = acquire(config, vm[:, first:first + block].T.astype(float), times)    # (nodes, samples)
        raw, starts, length = placementIndices(acquisitionTimes, config, windows)
        uni.append(acquired[:, raw][:, starts.min():].astype(np.float32))
        diff.append(sosfilt(sections, acquired[:, raw], axis=-1)[:, starts.min():].astype(np.float32))
    return dict(uni=np.concatenate(uni), differential=np.concatenate(diff), offsets=starts - starts.min(), length=length)


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
    retained = model.centres[:, 1] >= config["fixed"]["tissue"]["boundaryYmm"]     # tissue still active after ablation (FM-37)
    out.update(share_retained=float(elementShares[retained].sum()), mass_retained=float(mass[retained].sum() / total))
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


def firstCrossingMm(distanceMm, value, level):
    """First distance (moving into the lesion) where value falls to level, by linear interpolation; NaN if it never does.

    distanceMm, value: one recording's curve over its placements (any order). Returns NaN too when the
    curve already lies at or below the level at its first distance (no crossing inside the sampled range).
    """
    order = np.argsort(distanceMm)
    d, v = np.asarray(distanceMm, float)[order], np.asarray(value, float)[order]
    below = np.flatnonzero(v <= level)
    if not len(below) or below[0] == 0:
        return np.nan
    k = below[0]
    return float(d[k - 1] + (v[k - 1] - level) * (d[k] - d[k - 1]) / (v[k - 1] - v[k]))


def boundaryCrossings(summary, levels, maximumShareAbsSum=np.inf):
    """Ground-truth distances matched to the clinical boundary sweep (FM-37), intact run.

    For each recording (kind, label, gap), retained share (signed share and |c|-mass of tissue with
    y >= boundary, i.e. the tissue the lesion run leaves active) against the recording's distance inside
    the lesion; the distance where each curve first falls to each level.

    The |c|-mass curve is the primary measure (FM-43). A signed-share curve is reported only when none
    of its placements is ill-conditioned (sum |s_e| > maximumShareAbsSum: the signal is nearly null and
    the shares are large terms of opposite sign); otherwise its distance is NaN. illConditionedPlacements
    counts those placements, so the reason for a missing value is recorded, not hidden.
    """
    intact = summary[summary.condition == "intact"]
    rows = []
    for (kind, label, gap), group in intact.groupby(["kind", "label", "gapMm"]):
        illConditioned = int((group.shareAbsSum > maximumShareAbsSum).sum()) if "shareAbsSum" in group else 0
        for measure, column in (("signedShare", "share_retained"), ("mass", "mass_retained")):
            for level in levels:
                usable = measure == "mass" or illConditioned == 0
                rows.append(dict(kind=kind, label=label, gapMm=gap, measure=measure, level=level,
                                 distanceMm=firstCrossingMm(group.distanceInsideMm, group[column], level) if usable else np.nan,
                                 placements=len(group), illConditionedPlacements=illConditioned))
    return pd.DataFrame(rows)


def cumulativeLateral(elementShares, lateral, stepMm):
    edges = np.arange(0, lateral.max() + stepMm, stepMm)
    order = np.argsort(lateral)
    cumulative = np.cumsum(elementShares[order])
    counts = np.searchsorted(lateral[order], edges, side="right")
    return np.where(counts > 0, cumulative[np.maximum(counts - 1, 0)], 0.0)


# ----------------------------------------------------------------- stage
def runGroundTruth(config, jobs, tissues, electrogramsFolder, measurementsFolder, windows, folder):
    """Contribution fields of every recording at the ground-truth placements (clean signals, both conditions)."""
    folder, g = Path(folder), config["fixed"]["groundTruth"]
    gate = config["fixed"]["acceptance"]["groundTruth"]
    centres = groundTruthCentresYmm(config)
    layers = int(round(slabSizeMm(config)[2] / spacingMm(config)[2]))
    rows, depthRows, lateralRows, fieldValues = [], [], [], {}
    for condition, job in jobs.items():
        tissue = tissues[condition]
        model = ContributionModel(config, tissue)
        archive = np.load(Path(electrogramsFolder) / f"{condition}.npz")
        files = archive["leadFieldFiles"]
        points = pd.read_csv(Path(measurementsFolder) / f"{condition}Points.csv")
        points = points[(points.noiseRepeat == 0) & np.isclose(points.centreYmm.to_numpy()[:, None], centres).any(axis=1)]
        vm = processedVm(config, job, windows, len(model.tissueNodes))
        fieldLabels = {(kind, label) for kind, labels in g["fieldRecordings"].items() for label in labels}
        for p in sorted(points.placement.unique()):
            coefficientCache = {}                       # per placement only: ~80 MB per electrode on the full mesh
            for k in range(16):
                index = int(archive["electrodeIndex"][p, k])
                if index not in coefficientCache:
                    coefficientCache[index] = model.coefficients(loadLeadField(files[index]))
            window = slice(int(vm["offsets"][p]), int(vm["offsets"][p]) + vm["length"])
            for record in points[points.placement == p].to_dict("records"):
                weights = np.array([record[f"weightE{k}"] for k in range(16)])
                coefficients = sum(w * coefficientCache[int(archive["electrodeIndex"][p, k])]
                                   for k, w in enumerate(weights) if abs(w) > 1e-12)
                signal = vm["uni" if record["kind"] == "uni" else "differential"][:, window]
                elementShares, mass, phi, total = shares(model, coefficients, signal)
                anchor = [record["anchorXmm"], record["anchorYmm"]]
                lateral, depth, layer = elementGeometry(model, config, anchor)
                depthRows.append(np.concatenate([np.bincount(layer, elementShares, layers), np.bincount(layer, mass, layers) / mass.sum()]))
                lateralRows.append(cumulativeLateral(elementShares, lateral, g["curveStepMm"]))
                rows.append(dict(recordingId=record["recordingId"], condition=condition, kind=record["kind"], label=record["label"],
                                 gapMm=record["gapMm"], centreYmm=record["centreYmm"], anchorXmm=anchor[0], anchorYmm=anchor[1],
                                 distanceInsideMm=record["distanceInsideMm"], shareSum=total,
                                 shareAbsSum=float(np.abs(elementShares).sum()),
                                 signedShareIllConditioned=bool(np.abs(elementShares).sum() > g["maximumShareAbsSum"]),
                                 signalPpMv=float(np.ptp(phi)),
                                 **classSummary(model, config, elementShares, mass, lateral, depth),
                                 **shapeDescriptors(model, elementShares, lateral, depth, anchor, g["shapeLevels"])))
                if (record["kind"], record["label"]) in fieldLabels:
                    fieldValues.setdefault(condition, {})[record["recordingId"]] = elementShares.astype(np.float32)
        writeFields(model, condition, fieldValues.get(condition, {}), points, folder)
    table = pd.DataFrame(rows)
    table.to_csv(folder / "summary.csv", index=False)
    pad = lambda values: np.array([np.pad(v, (0, max(map(len, values)) - len(v)), constant_values=np.nan) for v in values])
    np.savez_compressed(folder / "profiles.npz", recordingId=table.recordingId.to_numpy().astype(str), depthShareAndMass=np.asarray(depthRows),
                        lateralCumulativeShare=pad(lateralRows), curveStepMm=g["curveStepMm"], layerThicknessMm=spacingMm(config)[2])
    boundaryCrossings(table, g["retainedShareLevels"], g["maximumShareAbsSum"]).to_csv(folder / "boundaryCrossings.csv", index=False)
    checks = dict(recordings=len(table), shareSumMaxDeviation=float((table.shareSum - 1).abs().max()), leadFields="openCARP lead-field module (FM-35)",
                  signedShareIllConditioned=int(table.signedShareIllConditioned.sum()))
    checks["passed"] = bool(checks["recordings"] > 0 and checks["shareSumMaxDeviation"] <= gate["shareSumDeviation"])
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
