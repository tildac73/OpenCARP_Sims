"""acceptance.json, report.md and the figures of one run."""
from pathlib import Path

import numpy as np
import pandas as pd

from .config import par
from .io import saveJson


def acceptance(run):
    physiology, measures = run.result("physiology"), run.result("measurements")["checks"]
    tuneResult, groundTruth = run.result("tune"), run.result("groundTruth")
    checks = {
        "tune.cvWithinTolerance": bool(tuneResult.get("passed", True)),
        "implementation.groundTruthReconstruction": bool(groundTruth["passed"]),
        **{f"physiology.{c}.{name}": value for c, item in physiology["conditions"].items() for name, value in item["criteria"].items()},
        **{f"profile.{name}": value for name, value in measures.items()},
    }
    return dict(checks=checks, modelAccepted=all(checks.values()),
                assumptions={k: v for k, v in run.config["provenance"].items() if v.get("status") == "assumption"})


def fmt(value, digits=3):
    if isinstance(value, bool):
        return "pass" if value else "FAIL"
    if isinstance(value, float):
        return f"{value:.{digits}g}"
    return "-" if value is None else str(value)


def changePointTable(run):
    path = run.out / "changePoints" / "SuppResults5_change_points_all.csv"
    if not path.exists():
        return ["No change-point table (Wang's analysis produced none)."]
    table = pd.read_csv(path)
    columns = [c for c in table.columns if c in ("dataset", "Location", "Metric", "Electrode", "cp_mm_mean", "cp_mm_lower", "cp_mm_upper",
                                                  "value_at_cp_mean")]
    lines = ["| " + " | ".join(columns) + " |", "| " + " | ".join("---" for _ in columns) + " |"]
    for row in table[columns].itertuples(index=False):
        lines.append("| " + " | ".join(fmt(v) if isinstance(v, float) else str(v) for v in row) + " |")
    return lines


def writeReport(run):
    out, config = run.out, run.config
    accepted = acceptance(run)
    saveJson(out / "acceptance.json", accepted)
    figures = makeFigures(run)
    physiology, window = run.result("physiology"), run.result("window")
    measures, changes, gt, tuned = run.result("measurements"), run.result("changePoints"), run.result("groundTruth"), run.result("tune")
    lines = [f"# {config['modelName']}: run report", "", f"Output folder: `{out}`", "",
             f"Sweep values: `{config['values']}`", "", f"**Model accepted: {fmt(accepted['modelAccepted'])}.**", "",
             "## Checks", "", "| Check | Result |", "| --- | --- |"]
    lines += [f"| {name} | {fmt(value)} |" for name, value in accepted["checks"].items()] + [""]
    lines += ["## Tuning", "", f"beta {par(config, 'imp_region[0].cellSurfVolRatio'):.6g} /um ({tuned.get('source', '')}); "
              f"strip CV {fmt(tuned.get('cvMPerS'))} m/s for target {config['values']['cvTargetMPerS']} m/s.", ""]
    lines += ["## Physiology", ""]
    for condition, c in physiology["conditions"].items():
        lines.append(f"- {condition}: last-beat surface CV {c['lastBeatSurfaceSpeedMPerS']:.3f} m/s (reported, not gated), "
                     f"{c['regionNodes']} nodes in the check region")
    lines += ["", f"Measurement window: {window['windowMs'][0]}-{window['windowMs'][1]} ms after the last stimulus "
              f"(grid activation {window['gridFirstActivationMs']:.1f}-{window['gridLastActivationMs']:.1f} ms). {window['note']}", ""]
    cal = measures["calibration"]
    lines += ["## Calibration", "", f"Gain {cal['gain']:.6g} ({cal['source']}). {cal['interpretation']}.", ""]
    lines += ["## Change points (Wang's code, one slab)", "",
              f"Eligible for interpretation: {fmt(changes['eligibleForInterpretation'])} "
              f"(prerequisites: {', '.join(f'{k} {fmt(v)}' for k, v in changes['prerequisites'].items())}).", ""]
    lines += changePointTable(run) + [""]
    lines += ["## Ground truth", "", f"{gt['recordings']} recordings ({gt['leadFieldMethod']} lead fields); share-sum deviation "
              f"{gt['shareSumMaxDeviation']:.2g}, reconstruction residual {gt['reconstructionResidualMax']:.2g}, "
              f"scale/gain deviation {gt['scaleOverGainMaxDeviation']:.2g}.",
              "Per-recording classes, profiles and shape descriptors: `groundTruth/summary.csv`, `groundTruth/profiles.npz`. "
              "Full element fields: `groundTruth/fields_<condition>.npz` and `groundTruth/shares_<condition>_gap<g>mm.vtk` (ParaView).", ""]
    if accepted["assumptions"]:
        lines += ["## Assumptions in force", ""] + [f"- {k}: {v['source']}" for k, v in accepted["assumptions"].items()] + [""]
    lines += ["## Figures", ""] + [f"- `{Path(f).relative_to(out)}`" for f in figures] + [""]
    (out / "report.md").write_text("\n".join(lines) + "\n")
    return dict(modelAccepted=accepted["modelAccepted"], report=str(out / "report.md"), figures=[str(f) for f in figures])


# ----------------------------------------------------------------- figures
def makeFigures(run):
    import matplotlib
    matplotlib.use("Agg")
    from .groundtruth_figures import makeGroundTruthFigures
    folder = run.folder("figures")
    made = [activationFigure(run, folder), profileFigure(run, folder, "ppMv", "PP (mV)", "02_voltage_profiles.png"),
            profileFigure(run, folder, "pfHz", "peak frequency (Hz)", "03_peak_frequency_profiles.png", log=False),
            tracesFigure(run, folder)]
    made += makeGroundTruthFigures(run, folder)
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
        image = axis.scatter(p[top, 0], p[top, 1], c=t[top], s=1, cmap="viridis")
        axis.axhline(run.config["fixed"]["tissue"]["boundaryYmm"], color="k", lw=0.8, ls="--")
        axis.set(title=f"{condition}: last-beat activation, catheter-facing surface", xlabel="x (mm)", aspect="equal")
    axes[0].set_ylabel("y (mm)")
    fig.colorbar(image, ax=axes, label="ms after stimulus")
    path = folder / "01_activation.png"
    fig.savefig(path, dpi=150, bbox_inches="tight"); plt.close(fig)
    return path


def profileFigure(run, folder, column, label, name, log=True):
    """Clean measures of every recording against distance inside the lesion, by kind and gap."""
    import matplotlib.pyplot as plt
    points = pd.concat([pd.read_csv(run.out / "measurements" / f"{c}Points.csv") for c in ("intact", "lesion")])
    points = points[(points.noiseRepeat == 0) & points[column].notna()]
    groups = points[["kind", "gapMm"]].drop_duplicates().sort_values(["kind", "gapMm"]).to_dict("records")
    if not groups:
        return None
    fig, axes = plt.subplots(len(groups), 1, figsize=(8, 2.6 * len(groups)), squeeze=False)
    for axis, key in zip(axes[:, 0], groups):
        for condition, colour in (("intact", "tab:blue"), ("lesion", "tab:orange")):
            subset = points[(points.condition == condition) & (points.kind == key["kind"]) & (points.gapMm == key["gapMm"])]
            axis.scatter(subset.distanceInsideMm, subset[column], s=3, color=colour, label=condition, alpha=0.5)
        axis.axvline(0, color="k", lw=0.6)
        axis.set(yscale="log" if log else "linear", ylabel=label, title=f"{key['kind']} (all recordings), gap {key['gapMm']:g} mm")
        axis.legend(fontsize=7)
    axes[-1, 0].set_xlabel("distance inside lesion (mm)")
    fig.tight_layout()
    path = folder / name
    fig.savefig(path, dpi=150); plt.close(fig)
    return path


def tracesFigure(run, folder):
    import matplotlib.pyplot as plt
    gap = min(run.config["values"]["gapsMm"])
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.5))
    for axis, (kind, label) in zip(axes, (("uni", "E5"), ("bi", "E5-E6|E9"), ("omni", "E5-E6-E9"))):
        for condition, colour in (("intact", "tab:blue"), ("lesion", "tab:orange")):
            points = pd.read_csv(run.out / "measurements" / f"{condition}Points.csv")
            archive = np.load(run.out / "measurements" / f"{condition}Traces.npz")
            chosen = points[(points.noiseRepeat == 0) & (points.label == label) & (points.gapMm == gap) & np.isclose(points.centreYmm, 0)]
            if chosen.empty:
                continue
            axis.plot(archive["timeMs"], archive["cleanMv"][int(chosen.iloc[0].traceIndex)], color=colour, label=condition)
        axis.set(title=f"{kind} {label}, centre y = 0, gap {gap:g} mm", xlabel="time (ms)", ylabel="mV")
        axis.legend(fontsize=7)
    fig.tight_layout()
    path = folder / "04_example_traces.png"
    fig.savefig(path, dpi=150); plt.close(fig)
    return path
