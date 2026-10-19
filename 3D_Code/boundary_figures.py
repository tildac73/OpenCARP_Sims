"""A short diagnostic figure set for the controlled boundary experiment."""

import json
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib

# Use a file-only backend in terminal runs; keep Jupyter's existing display backend.
if "matplotlib.pyplot" not in sys.modules:
    matplotlib.use("Agg")
import matplotlib.pyplot as plt


def saveFigure(figure, outputFolder, name):
    figure.savefig(outputFolder / f"{name}.png", dpi=180, bbox_inches="tight")
    figure.savefig(outputFolder / f"{name}.pdf", bbox_inches="tight")
    plt.close(figure)


def makeFigures(runFolder, config):
    """Write activation diagnostics first, then the four signal/contribution figures."""
    runFolder = Path(runFolder)
    outputFolder = runFolder / "figures"
    outputFolder.mkdir(exist_ok=True)
    plotActivation(runFolder, config, outputFolder)
    if not (runFolder / "results/points.csv").exists():
        return
    points = pd.read_csv(runFolder / "results/points.csv")
    clean = points[points.noiseRepeat == 0]
    plotVoltageDistance(clean, config, outputFolder)
    plotContributionRadii(clean, outputFolder)
    plotExampleTraces(runFolder, points, clean, outputFolder)
    plotReconstructionChecks(clean, outputFolder)
    status = json.loads((runFolder / "results/pilotStatus.json").read_text())
    (outputFolder / "README.txt").write_text(
        "These figures describe a pilot, not a mesh- or domain-converged EFOV result.\n"
        f"Pilot acceptance passed: {status['pilotAcceptancePassed']}\n"
        "Read physiologyChecks.csv, results/boundaryChecks.csv and results/contributions.csv.\n"
        "Clean and noisy orientations are selected independently; exact weights are saved.\n"
    )


def plotActivation(runFolder, config, outputFolder):
    figure, axes = plt.subplots(1, 2, figsize=(11, 4), constrained_layout=True)
    for axis, condition in zip(axes, ["intact", "lesion"]):
        cache = np.load(runFolder / condition / "activationChecks.npz")
        positions = cache["positionsMm"]
        surface = np.isclose(positions[:, 2], positions[:, 2].max())
        colours = cache["activationMs"][-1, surface]
        points = axis.scatter(
            positions[surface, 0], positions[surface, 1], c=colours, s=5, cmap="viridis"
        )
        figure.colorbar(points, ax=axis, label="Activation after stimulus (ms)")
        if config["boundary"]["shape"] == "straight":
            axis.axhline(config["boundary"]["positionMm"], color="black", linestyle="--")
        else:
            axis.add_patch(
                plt.Circle(
                    config["boundary"]["centreMm"],
                    config["boundary"]["radiusMm"],
                    fill=False,
                    color="black",
                )
            )
        axis.set(title=condition, xlabel="x (mm)", ylabel="y (mm)", aspect="equal")
    saveFigure(figure, outputFolder, "01_activation")


def plotVoltageDistance(clean, config, outputFolder):
    gaps = config["electrodes"]["gapsMm"]
    figure, axes = plt.subplots(
        len(gaps), 3, figsize=(13, 4 * len(gaps)), squeeze=False, constrained_layout=True
    )
    for gapIndex, gap in enumerate(gaps):
        for kindIndex, kind in enumerate(["uni", "bi", "omni"]):
            axis = axes[gapIndex, kindIndex]
            for condition, colour in [("intact", "tab:blue"), ("lesion", "tab:orange")]:
                subset = clean[
                    (clean.kind == kind)
                    & np.isclose(clean.gapMm, gap)
                    & (clean.condition == condition)
                ]
                profile = subset.groupby("distanceViableMm").ppMv.median().sort_index()
                axis.plot(profile.index, profile.values, ".-", color=colour, label=condition)
            axis.axvline(0, color="black", linestyle="--", linewidth=0.7)
            axis.set(
                title=f"{kind}; gap {gap:g} mm",
                xlabel="Distance into viable side (mm)",
                ylabel="Clean peak-to-peak (mV)",
            )
            axis.legend()
    saveFigure(figure, outputFolder, "02_clean_voltage_distance")


def plotContributionRadii(clean, outputFolder):
    figure, axes = plt.subplots(1, 3, figsize=(13, 4), constrained_layout=True)
    for axis, kind in zip(axes, ["uni", "bi", "omni"]):
        for condition, colour in [("intact", "tab:blue"), ("lesion", "tab:orange")]:
            subset = clean[
                (clean.kind == kind) & (clean.condition == condition) & (clean.fovEligible == True)
            ]
            axis.scatter(
                subset.distanceViableMm,
                subset.fov80ReconstructionMm,
                s=12,
                alpha=0.5,
                color=colour,
                label=condition,
            )
        axis.set(
            title=kind,
            xlabel="Distance into viable side (mm)",
            ylabel="Eligible reconstruction radius (mm)",
        )
        axis.legend()
    figure.suptitle("Pilot contribution estimates; gaps shown together; convergence unverified")
    saveFigure(figure, outputFolder, "03_contribution_radii")


def plotExampleTraces(runFolder, points, clean, outputFolder):
    figure, axes = plt.subplots(2, 3, figsize=(13, 7), constrained_layout=True)
    for conditionIndex, condition in enumerate(["intact", "lesion"]):
        archive = np.load(runFolder / "results" / f"{condition}Traces.npz")
        for kindIndex, kind in enumerate(["uni", "bi", "omni"]):
            subset = clean[(clean.condition == condition) & (clean.kind == kind)]
            selected = subset.loc[subset.distanceViableMm.abs().idxmin()]
            axis = axes[conditionIndex, kindIndex]
            axis.plot(
                archive["timeMs"], archive["measuredMv"][int(selected.traceIndex)], label="clean"
            )
            noisy = points[
                (points.condition == condition)
                & (points.kind == kind)
                & (points.label == selected.label)
                & (points.placement == selected.placement)
                & (points.noiseRepeat == 1)
            ]
            if len(noisy):
                axis.plot(
                    archive["timeMs"],
                    archive["measuredMv"][int(noisy.iloc[0].traceIndex)],
                    alpha=0.6,
                    label="noise repeat 1",
                )
            axis.set(
                title=f"{condition}, {kind}, d={selected.distanceViableMm:.2f} mm",
                xlabel="Time (ms)",
                ylabel="mV",
            )
            axis.legend()
        archive.close()
    saveFigure(figure, outputFolder, "04_example_traces")


def plotReconstructionChecks(clean, outputFolder):
    figure, axes = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)
    computed = clean[clean.groundTruthComputed == True]
    for kind, group in computed.groupby("kind"):
        axes[0].scatter(
            group.distanceViableMm, group.signedCorrelation, s=10, alpha=0.5, label=kind
        )
        axes[1].scatter(group.distanceViableMm, group.shapeError, s=10, alpha=0.5, label=kind)
    axes[0].set(xlabel="Distance into viable side (mm)", ylabel="Signed reconstruction correlation")
    axes[1].set(
        xlabel="Distance into viable side (mm)", ylabel="Shape error after fitted signed scale"
    )
    for axis in axes:
        axis.legend()
    saveFigure(figure, outputFolder, "05_reconstruction_checks")
