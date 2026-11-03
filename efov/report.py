"""acceptance.json, report.md and the figures of one run."""
from pathlib import Path

import numpy as np
import pandas as pd

from .config import unconfirmed
from .io import loadJson, saveJson


def acceptance(run):
    physiology = run.result("physiology")
    forward = run.result("electrograms")["forwardModelCheck"]
    measures = run.result("measurements")["checks"]
    groundTruth = run.result("groundTruth")
    checks = {
        "implementation.forwardModel": all(c["passed"] for c in forward.values()),
        "implementation.groundTruth": bool(groundTruth["passed"]),
        **{f"physiology.{condition}.{name}": value for condition, c in physiology["conditions"].items()
           for name, value in c["criteria"].items()},
        **{f"profile.{name}": value for name, value in measures.items()},
    }
    pending = unconfirmed(run.config)
    return dict(checks=checks, modelAccepted=all(checks.values()), unconfirmedValues=pending,
                productionReady=bool(all(checks.values()) and not pending),
                note="productionReady also needs every configuration value confirmed (provenance).")


def fmt(value, digits=3):
    if isinstance(value, bool):
        return "pass" if value else "FAIL"
    if isinstance(value, float):
        return f"{value:.{digits}g}"
    return "-" if value is None else str(value)


def writeReport(run):
    out, config = run.out, run.config
    accepted = acceptance(run)
    saveJson(out / "acceptance.json", accepted)
    figures = makeFigures(run)
    physiology, window = run.result("physiology"), run.result("window")
    measures, changes, gt = run.result("measurements"), run.result("changePoints"), run.result("groundTruth")
    forward = run.result("electrograms")["forwardModelCheck"]
    lines = [f"# {config['modelName']}: run report", "", f"Output folder: `{out}`", "",
             f"**Model accepted: {fmt(accepted['modelAccepted'])}.** Production ready (also needs confirmed values): "
             f"{fmt(accepted['productionReady'])}.", ""]
    if accepted["unconfirmedValues"]:
        lines += ["## Values awaiting confirmation", ""] + [f"- `{k}`: {v['source']}" for k, v in accepted["unconfirmedValues"].items()] + [""]
    lines += ["## Checks", "", "| Check | Result |", "| --- | --- |"]
    lines += [f"| {name} | {fmt(value)} |" for name, value in accepted["checks"].items()] + [""]
    lines += ["## Physiology", ""]
    for condition, c in physiology["conditions"].items():
        lines.append(f"- {condition}: last-beat CV {c['lastBeatSpeedMPerS']:.3f} m/s (target {c['targetMPerS']} m/s), "
                     f"{c['regionNodes']} nodes in the check region")
    lines += ["", f"Measurement window: {window['windowMs'][0]}-{window['windowMs'][1]} ms after the last stimulus "
              f"(grid activation {window['gridFirstActivationMs']:.1f}-{window['gridLastActivationMs']:.1f} ms). {window['note']}", ""]
    lines += ["## Forward-model check (openCARP phi_e vs independent sigma_m solve)", ""]
    lines += [f"- {c}: relative RMS max {v['relativeRmsMax']:.2g}, correlation min {v['correlationMin']:.10f} ({v['samples']} samples, "
              f"{v['frames']} frames)" for c, v in forward.items()] + [""]
    cal = measures["calibration"]
    lines += ["## Calibration", "", f"Gain {cal['gain']:.6g} (reference median bipole {cal['rawReferenceMedianMv']:.4g} model units "
              f"-> {cal['targetBipolarMv']} mV). {cal['interpretation']}.", ""]
    lines += ["## Change points (clean, clinical preprocessing, recording anchor)", "",
              "| Condition | Gap (mm) | Recording | Model | Change point (mm) | Status | Eligible |", "| --- | --- | --- | --- | --- | --- | --- |"]
    for row in changes["primary"]:
        lines.append(f"| {row['condition']} | {row['gapMm']:g} | {row['kind']} {row['label']} | {fmt(row.get('model'))} | "
                     f"{fmt(row.get('changePointMm'))} | {row['fitStatus']} | {'yes' if row['eligibleForInterpretation'] is True else 'no'} |")
    lines += ["", "Prerequisites for interpretation: " + ", ".join(f"{k} {fmt(v)}" for k, v in changes["prerequisites"].items()), ""]
    lines += ["## Ground truth", "", f"{gt['recordings']} recordings; share-sum deviation {gt['shareSumMaxDeviation']:.2g}, "
              f"reconstruction residual {gt['reconstructionResidualMax']:.2g}, scale/gain deviation {gt['scaleOverGainMaxDeviation']:.2g}.",
              "Fields: `groundTruth/contributionField.npz` (0.5 mm voxels, signed shares, sum 1).", ""]
    lines += ["## Figures", ""] + [f"- `{Path(f).relative_to(out)}`" for f in figures] + [""]
    (out / "report.md").write_text("\n".join(lines) + "\n")
    return dict(modelAccepted=accepted["modelAccepted"], productionReady=accepted["productionReady"],
                report=str(out / "report.md"), figures=[str(f) for f in figures])


# ----------------------------------------------------------------- figures
def makeFigures(run):
    import matplotlib
    matplotlib.use("Agg")
    folder = run.folder("figures")
    made = [activationFigure(run, folder), profileFigure(run, folder), tracesFigure(run, folder), fieldFigure(run, folder)]
    return [f for f in made if f is not None]


def activationFigure(run, folder):
    import matplotlib.pyplot as plt
    from .physiology import lastBeatActivation
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), sharey=True)
    for axis, condition in zip(axes, ("intact", "lesion")):
        tissue = run.tissue(condition)
        p = tissue["tissuePositions"]
        top = np.isclose(p[:, 2], p[:, 2].max())
        t = lastBeatActivation(run.config, run.jobs[condition], tissue)
        image = axis.scatter(p[top, 0], p[top, 1], c=t[top], s=4, cmap="viridis")
        axis.axhline(run.config["lesion"]["boundaryYmm"], color="k", lw=0.8, ls="--")
        axis.set(title=f"{condition}: last-beat activation, top surface", xlabel="x (mm)", aspect="equal")
    axes[0].set_ylabel("y (mm)")
    fig.colorbar(image, ax=axes, label="ms after stimulus")
    path = folder / "01_activation.png"
    fig.savefig(path, dpi=150, bbox_inches="tight"); plt.close(fig)
    return path


def profileFigure(run, folder):
    import matplotlib.pyplot as plt
    from .changepoints import primaryFits
    fits = primaryFits(pd.read_csv(run.out / "changePoints" / "fits.csv"))
    data = pd.read_csv(run.out / "changePoints" / "fitData.csv")
    predictions = pd.read_csv(run.out / "changePoints" / "predictions.csv")
    points = pd.concat([pd.read_csv(run.out / "measurements" / f"{c}Points.csv") for c in ("intact", "lesion")])
    points = points[points.noiseRepeat == 0]
    groups = points[["kind", "label", "gapMm"]].drop_duplicates().to_dict("records")
    fig, axes = plt.subplots(len(groups), 1, figsize=(8, 2.6 * len(groups)), squeeze=False)
    for axis, key in zip(axes[:, 0], groups):
        for condition, colour in (("intact", "tab:blue"), ("lesion", "tab:orange")):
            subset = points[(points.condition == condition) & (points.kind == key["kind"]) & (points.label == key["label"])
                            & (points.gapMm == key["gapMm"])]
            axis.scatter(subset.distanceInsideMm, subset.ppMv, s=8, color=colour, label=condition)
            fit = fits[(fits.condition == condition) & (fits.kind == key["kind"]) & (fits.label == key["label"]) & (fits.gapMm == key["gapMm"])]
            for f in fit.itertuples():
                curve = predictions[(predictions.profileId == f.profileId) & (predictions.model == f.model)]
                axis.plot(curve.distanceMm, curve.predictedMv, color=colour, lw=1)
                if bool(f.descriptiveSupport):
                    axis.axvline(f.changePointMm, color=colour, ls="--", lw=0.8)
        axis.axvline(0, color="k", lw=0.6)
        axis.set(yscale="log", ylabel="PP (mV)", title=f"{key['kind']} {key['label']}, gap {key['gapMm']:g} mm")
        axis.legend(fontsize=7)
    axes[-1, 0].set_xlabel("distance into lesion (mm)")
    fig.tight_layout()
    path = folder / "02_voltage_profiles.png"
    fig.savefig(path, dpi=150); plt.close(fig)
    return path


def tracesFigure(run, folder):
    import matplotlib.pyplot as plt
    config = run.config
    gap = config["electrodes"]["gapsMm"][0]
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.5), sharey=False)
    for axis, kind in zip(axes, ("uni", "bi", "omni")):
        for condition, colour in (("intact", "tab:blue"), ("lesion", "tab:orange")):
            points = pd.read_csv(run.out / "measurements" / f"{condition}Points.csv")
            archive = np.load(run.out / "measurements" / f"{condition}Traces.npz")
            chosen = points[(points.noiseRepeat == 0) & (points.kind == kind) & (points.gapMm == gap) & np.isclose(points.centreYmm, 0)]
            if chosen.empty:
                continue
            row = chosen.iloc[0]
            axis.plot(archive["timeMs"], archive["cleanMv"][int(row.traceIndex)], color=colour, label=f"{condition} {row.label}")
        axis.set(title=f"{kind}, centre y = 0, gap {gap:g} mm", xlabel="time (ms)", ylabel="mV")
        axis.legend(fontsize=7)
    fig.tight_layout()
    path = folder / "03_example_traces.png"
    fig.savefig(path, dpi=150); plt.close(fig)
    return path


def fieldFigure(run, folder):
    import matplotlib.pyplot as plt
    archive = np.load(run.out / "groundTruth" / "contributionField.npz")
    ids = list(archive["recordingId"])
    summary = pd.read_csv(run.out / "groundTruth" / "summary.csv")
    chosen = summary[(summary.condition == "intact") & (summary.kind == "bi") & np.isclose(summary.centreYmm, 0)]
    if chosen.empty:
        return None
    record = chosen.iloc[0]
    field = archive["fields"][ids.index(record.recordingId)]                # (nx, ny, nz)
    voxel, lower = float(archive["voxelMm"]), archive["lowerMm"]
    limit = np.abs(field).max()
    extent = lambda a, b: [lower[a], lower[a] + field.shape[a] * voxel, lower[b], lower[b] + field.shape[b] * voxel]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].imshow(field.sum(axis=2).T, origin="lower", extent=extent(0, 1), cmap="RdBu_r", vmin=-limit, vmax=limit)
    axes[0].set(title=f"{record.label}, gap {record.gapMm:g} mm: share summed through the wall", xlabel="x (mm)", ylabel="y (mm)")
    j = int(np.floor((record.anchorYmm - lower[1]) / voxel))
    image = axes[1].imshow(field[:, j, :].T, origin="lower", extent=extent(0, 2), cmap="RdBu_r", vmin=-limit, vmax=limit, aspect="auto")
    axes[1].set(title=f"section at y = {record.anchorYmm:.1f} mm", xlabel="x (mm)", ylabel="z (mm)")
    fig.colorbar(image, ax=axes, label="signed share per voxel")
    path = folder / "04_ground_truth_field.png"
    fig.savefig(path, dpi=150, bbox_inches="tight"); plt.close(fig)
    return path
