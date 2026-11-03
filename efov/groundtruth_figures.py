"""Figures of the ground-truth contributing tissue (FM-21). All from the element shares s_e, no cut-off.

  sectionsFigure     x-y sections at the endocardial surface, mid-wall and epicardial surface, and
                     the x-z and y-z planes through the recording anchor (signed, symmetric-log colours)
  boundarySequence   the same endocardial section for one recording at every ground-truth placement
  profilesFigure     transmural depth profile (one value per node layer), cumulative lateral share,
                     and contribution classes against distance from the boundary
Element shares are summed into voxels of the mesh spacing (dx x dx x dz), so nothing is resampled
below the mesh resolution.
"""
from pathlib import Path

import numpy as np
import pandas as pd

from .config import slabSizeMm, spacingMm
from .groundtruth import CLASSES


def voxelField(positions, elements, values, config):
    """Sum element values into mesh-spacing voxels by element centroid: (nx, ny, nz), lower corner, spacing."""
    size, spacing = slabSizeMm(config), spacingMm(config)
    lower = -size / 2
    shape = tuple(int(n) for n in np.round(size / spacing))
    index = np.floor((positions[elements].mean(axis=1) - lower) / spacing + 1e-9).astype(int)
    index = np.clip(index, 0, np.asarray(shape) - 1)
    flat = np.ravel_multi_index(index.T, shape)
    return np.bincount(flat, weights=values, minlength=int(np.prod(shape))).reshape(shape), lower, spacing


def lesionEdges(config, axis, vertical=False):
    """Dashed lines at the lesion boundary and the far edge of the lesion strip (FM-24)."""
    tissue = config["fixed"]["tissue"]
    for y in (tissue["boundaryYmm"], tissue["boundaryYmm"] - tissue["lesionWidthMm"]):
        (axis.axvline if vertical else axis.axhline)(y, color="k", lw=0.6, ls="--")


def _norm(field):
    import matplotlib.colors as colors
    limit = float(np.abs(field).max()) or 1.0
    return colors.SymLogNorm(linthresh=limit * 1e-3, vmin=-limit, vmax=limit)


def sectionsFigure(run, condition, recordingId, path):
    import matplotlib.pyplot as plt
    archive = np.load(run.out / "groundTruth" / f"fields_{condition}.npz")
    ids = list(archive["recordingId"])
    tissue = run.tissue(condition)
    summary = pd.read_csv(run.out / "groundTruth" / "summary.csv").set_index("recordingId")
    record = summary.loc[recordingId]
    field, lower, spacing = voxelField(tissue["positions"], archive["elements"], archive["shares"][ids.index(recordingId)], run.config)
    norm = _norm(field)
    nz = field.shape[2]
    i = int(np.clip((record.anchorXmm - lower[0]) / spacing[0], 0, field.shape[0] - 1))
    j = int(np.clip((record.anchorYmm - lower[1]) / spacing[1], 0, field.shape[1] - 1))
    upper = lower + np.asarray(field.shape) * spacing
    fig, axes = plt.subplots(1, 5, figsize=(22, 4.2), gridspec_kw=dict(width_ratios=[1, 1, 1, 1.4, 1.4]))
    for axis, (layer, name) in zip(axes[:3], [(nz - 1, "endocardial surface"), (nz // 2, "mid-wall"), (0, "epicardial surface")]):
        image = axis.imshow(field[:, :, layer].T, origin="lower", extent=[lower[0], upper[0], lower[1], upper[1]], cmap="RdBu_r", norm=norm)
        lesionEdges(run.config, axis)
        axis.plot(record.anchorXmm, record.anchorYmm, "k+")
        axis.set(title=f"x-y, {name}", xlabel="x (mm)", ylabel="y (mm)")
    axes[3].imshow(field[:, j, :].T, origin="lower", extent=[lower[0], upper[0], lower[2], upper[2]], cmap="RdBu_r", norm=norm, aspect="auto")
    axes[3].set(title=f"x-z at y = {record.anchorYmm:.1f} mm", xlabel="x (mm)", ylabel="z (mm)")
    axes[4].imshow(field[i, :, :].T, origin="lower", extent=[lower[1], upper[1], lower[2], upper[2]], cmap="RdBu_r", norm=norm, aspect="auto")
    lesionEdges(run.config, axes[4], vertical=True)
    axes[4].set(title=f"y-z at x = {record.anchorXmm:.1f} mm", xlabel="y (mm)", ylabel="z (mm)")
    fig.colorbar(image, ax=axes, label="signed share per voxel (sum = 1)")
    fig.suptitle(f"{condition}: {record.kind} {record.label}, gap {record.gapMm:g} mm, distance inside lesion {record.distanceInsideMm:+.1f} mm")
    fig.savefig(path, dpi=150, bbox_inches="tight"); plt.close(fig)
    return path


def boundarySequence(run, condition, kind, label, gapMm, path):
    import matplotlib.pyplot as plt
    archive = np.load(run.out / "groundTruth" / f"fields_{condition}.npz")
    ids = list(archive["recordingId"])
    tissue = run.tissue(condition)
    summary = pd.read_csv(run.out / "groundTruth" / "summary.csv")
    chosen = summary[(summary.kind == kind) & (summary.label == label) & np.isclose(summary.gapMm, gapMm)
                     & (summary.condition == condition) & summary.recordingId.isin(ids)].sort_values("centreYmm")
    if chosen.empty:
        return None
    fig, axes = plt.subplots(1, len(chosen), figsize=(2.2 * len(chosen), 3.6), squeeze=False)
    for axis, record in zip(axes[0], chosen.itertuples()):
        field, lower, spacing = voxelField(tissue["positions"], archive["elements"], archive["shares"][ids.index(record.recordingId)], run.config)
        upper = lower + np.asarray(field.shape) * spacing
        axis.imshow(field.sum(axis=2).T, origin="lower", extent=[lower[0], upper[0], lower[1], upper[1]], cmap="RdBu_r", norm=_norm(field.sum(axis=2)))
        lesionEdges(run.config, axis)
        axis.set(title=f"d = {record.distanceInsideMm:+.1f}", xticks=[], yticks=[])
        axis.set_ylim(record.anchorYmm - 12, record.anchorYmm + 12); axis.set_xlim(record.anchorXmm - 12, record.anchorXmm + 12)
    fig.suptitle(f"{condition}: {kind} {label}, gap {gapMm:g} mm, share summed through the wall, by distance inside the lesion (mm)")
    fig.savefig(path, dpi=130, bbox_inches="tight"); plt.close(fig)
    return path


def profilesFigure(run, path):
    import matplotlib.pyplot as plt
    summary = pd.read_csv(run.out / "groundTruth" / "summary.csv")
    profiles = np.load(run.out / "groundTruth" / "profiles.npz", allow_pickle=False)
    ids = list(profiles["recordingId"])
    layers = profiles["depthShareAndMass"].shape[1] // 2
    dz = float(profiles["layerThicknessMm"])
    gap = summary.gapMm.min()
    fig, axes = plt.subplots(1, 3, figsize=(17, 4.5))
    for kind, colour in (("uni", "tab:green"), ("bi", "tab:blue"), ("omni", "tab:red")):
        centre = summary.centreYmm[summary.centreYmm > 0].min() if (summary.centreYmm > 0).any() else 0.0   # nearest viable-side centre
        label = run.config["fixed"]["groundTruth"]["fieldRecordings"][kind][0]
        rows = summary[(summary.kind == kind) & (summary.label == label) & (summary.condition == "intact")
                       & np.isclose(summary.gapMm, gap) & np.isclose(summary.centreYmm, centre)]
        if rows.empty:
            continue
        k = ids.index(rows.iloc[0].recordingId)
        axes[0].plot(profiles["depthShareAndMass"][k, layers:], (np.arange(layers) + 0.5) * dz, color=colour, label=f"{kind} {rows.iloc[0].label}")
        curve = profiles["lateralCumulativeShare"][k]
        axes[1].plot(np.arange(len(curve)) * float(profiles["curveStepMm"]), curve, color=colour, label=kind)
    axes[0].invert_yaxis(); axes[0].set(xlabel="|c|-mass fraction per layer", ylabel="depth below catheter-facing surface (mm)",
                                         title=f"Transmural profile (intact, gap {gap:g} mm)")
    axes[1].axhline(1, color="k", lw=0.5); axes[1].set(xlabel="lateral distance from anchor (mm)", ylabel="cumulative signed share", title="Lateral reach")
    lesion = summary[(summary.condition == "lesion") & np.isclose(summary.gapMm, gap) & (summary.kind == "bi")]
    grouped = lesion.groupby("distanceInsideMm")[[f"mass_{c}" for c in CLASSES]].median()
    for name in CLASSES:
        axes[2].plot(grouped.index, grouped[f"mass_{name}"], label=name)
    axes[2].axvline(0, color="k", lw=0.6)
    axes[2].axvline(run.config["fixed"]["tissue"]["lesionWidthMm"], color="k", lw=0.6, ls="--")   # far edge of the lesion strip
    axes[2].set(xlabel="distance inside lesion (mm)", ylabel="|c|-mass fraction (median over bipoles)", title=f"Contribution classes, lesion, gap {gap:g} mm")
    for axis in axes:
        axis.legend(fontsize=7)
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)
    return path


def makeGroundTruthFigures(run, folder):
    import matplotlib
    matplotlib.use("Agg")
    folder = Path(folder)
    made = [profilesFigure(run, folder / "05_ground_truth_profiles.png")]
    g = run.config["fixed"]["groundTruth"]
    gap = min(run.config["values"]["gapsMm"])
    summary = pd.read_csv(run.out / "groundTruth" / "summary.csv")
    for condition in ("intact", "lesion"):
        if not (run.out / "groundTruth" / f"fields_{condition}.npz").exists():
            continue
        for kind, labels in g["fieldRecordings"].items():
            for label in labels:
                chosen = summary[(summary.condition == condition) & (summary.kind == kind) & (summary.label == label)
                                 & np.isclose(summary.gapMm, gap) & np.isclose(summary.centreYmm, 0.0)]
                if len(chosen):
                    made.append(sectionsFigure(run, condition, chosen.iloc[0].recordingId,
                                               folder / f"06_sections_{condition}_{kind}_{label.replace('|', 'or')}.png"))
        for kind, labels in g["fieldRecordings"].items():
            made.append(boundarySequence(run, condition, kind, labels[0], gap, folder / f"07_sequence_{condition}_{kind}.png"))
    return [m for m in made if m is not None]
