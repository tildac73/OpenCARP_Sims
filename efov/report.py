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
        "implementation.groundTruthShareSum": bool(groundTruth["passed"]),
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


KIND_OF_ELECTRODE = {"Unipolar": "uni", "Bipolar": "bi", "Omnipolar": "omni"}


def predictedShiftMm(model, beta1, cpMm, noiseRatio=2.0):
    """Change-point shift predicted when the plateau is set by noise and the noise SD is multiplied by noiseRatio (FM-38).

    Wang's model: ln Y = b0 + b1 x up to the change point, then flat. If the plateau is noise peak-to-peak,
    multiplying the noise by r raises the plateau by ln r and moves the change point by -ln r / |b1| on the
    x scale (x = d for the exponential model, ln d for the power law). NaN when the fitted slope does not decay.
    """
    if not np.isfinite(beta1) or beta1 >= 0:
        return np.nan
    shift = -np.log(noiseRatio) / abs(beta1)
    return float(shift) if model == "Exponential" else float(cpMm * (np.exp(shift) - 1.0))


def efovComparison(run):
    """One table (FM-38): Wang's change point and decay slope on the clean datasets, beside the boundary-matched ground truth.

    Per Location, Metric, Electrode and gap (EnSite-style PF dataset, noise repeat 0): cp_mm and its interval,
    value at the change point, the piecewise slope beta1 and the log drop |beta1| x cp it implies (the change
    point is the distance at which the fitted decay has fallen by this many log units), the shift predicted per
    doubling of noise if the plateau is noise, and the ground-truth crossing distances (FM-37, intact run,
    median over the recordings of the kind). Returns the table, or None before the change points exist.
    """
    from .export import parseDatasetName
    folder = run.out / "changePoints"
    if not (folder / "SuppResults5_change_points_all.csv").exists() or not (folder / "Table5_bayesian_posteriors_all.csv").exists():
        return None
    cps, posteriors = pd.read_csv(folder / "SuppResults5_change_points_all.csv"), pd.read_csv(folder / "Table5_bayesian_posteriors_all.csv")
    for table in (cps, posteriors):
        parsed = table.dataset.map(parseDatasetName)
        table["gapMm"], table["noiseRepeat"], table["pfDefinition"] = [q[0] for q in parsed], [q[1] for q in parsed], [q[2] for q in parsed]
    keys = ["dataset", "Location", "Metric", "Electrode"]
    slopes = posteriors[posteriors.term.isin(["beta1", "cp"])].pivot_table(index=keys + ["model"], columns="term", values="mean").reset_index()
    table = cps[(cps.noiseRepeat == 0) & (cps.pfDefinition == "ensite")].merge(slopes, on=keys, how="left")
    table["logDropAtCp"] = (table.beta1 * table.cp).abs()
    table["predictedShiftPerNoiseDoublingMm"] = [predictedShiftMm(m, b, c) for m, b, c in zip(table.model, table.beta1, table.cp_mm_mean)]
    table["kind"] = table.Electrode.map(KIND_OF_ELECTRODE)
    crossingsPath = run.out / "groundTruth" / "boundaryCrossings.csv"
    if crossingsPath.exists():
        crossings = pd.read_csv(crossingsPath)
        wide = crossings.groupby(["kind", "gapMm", "measure", "level"]).distanceMm.median().reset_index()
        wide["column"] = "gt_" + wide.measure + "_" + (wide.level * 100).round().astype(int).astype(str) + "pct_mm"
        wide = wide.pivot_table(index=["kind", "gapMm"], columns="column", values="distanceMm").reset_index()
        table = table.merge(wide, on=["kind", "gapMm"], how="left")
    columns = ["Location", "Metric", "Electrode", "gapMm", "model", "cp_mm_mean", "cp_mm_lower", "cp_mm_upper", "value_at_cp_mean",
               "beta1", "logDropAtCp", "predictedShiftPerNoiseDoublingMm"] + sorted(c for c in table.columns if c.startswith("gt_"))
    table = table[[c for c in columns if c in table.columns]].sort_values(["Metric", "Electrode", "gapMm"])
    table.to_csv(run.out / "efovComparison.csv", index=False)
    return table


def comparisonLines(table):
    if table is None or table.empty:
        return ["No comparison table (change points or ground truth missing)."]
    shown = [c for c in ("Metric", "Electrode", "gapMm", "cp_mm_mean", "beta1", "logDropAtCp", "predictedShiftPerNoiseDoublingMm",
                         "gt_signedShare_50pct_mm", "gt_signedShare_10pct_mm", "gt_mass_10pct_mm") if c in table.columns]
    lines = ["| " + " | ".join(shown) + " |", "| " + " | ".join("---" for _ in shown) + " |"]
    for row in table[shown].itertuples(index=False):
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
    lines += ["", f"Measurement windows: {window['windowLengthMs']:g} ms per placement, starts {window['windowMs'][0]}-"
              f"{max(w[0] for w in window['windowsMs'])} ms after the last stimulus (first activation under a placement "
              f"{window['firstActivationMs'][0]:.1f}-{window['firstActivationMs'][1]:.1f} ms). {window['note']}", ""]
    amplitudes = measures["amplitudes"]
    lines += ["## Amplitudes (model units, not calibrated)", "", "| kind | gap (mm) | median clean PP far in viable tissue (mV) |",
              "| --- | --- | --- |"] + [f"| {r['kind']} | {r['gapMm']:g} | {r['medianCleanPpMv']:.3g} |" for r in amplitudes["modelFarViable"]]
    lines += ["", f"Wang's clinical median bipolar voltage ({amplitudes['region']}): {amplitudes['wangBipolarMedianMv']} mV. "
              f"{amplitudes['note']}", ""]
    lines += ["## Change points (Wang's code, one slab)", "",
              f"Eligible for interpretation: {fmt(changes['eligibleForInterpretation'])} "
              f"(prerequisites: {', '.join(f'{k} {fmt(v)}' for k, v in changes['prerequisites'].items())}).", ""]
    lines += changePointTable(run) + [""]
    lines += ["## Change point beside the boundary-matched ground truth (FM-37, FM-38)", "",
              "Clean datasets, EnSite-style PF. `logDropAtCp` = |beta1| x cp: how far the fitted decay has fallen at the change point. "
              "`predictedShiftPerNoiseDoublingMm`: the shift expected if the plateau is noise (compare noiseDoubling.csv). "
              "`gt_*`: distance inside the lesion where the intact recording's retained share first falls to the level (median over "
              "the recordings of the kind). Full table: `efovComparison.csv`.", ""]
    lines += comparisonLines(efovComparison(run)) + [""]
    lines += ["## Ground truth", "", f"{gt['recordings']} recordings ({gt['leadFields']} lead fields, the same as the "
              f"electrograms'); share-sum deviation {gt['shareSumMaxDeviation']:.2g}.",
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
            profileFigure(run, folder, "pfHz", "peak frequency, EnSite-style (Hz)", "03_peak_frequency_profiles.png", log=False),
            profileFigure(run, folder, "pfGlobalHz", "peak frequency, global maximum (Hz)", "03b_peak_frequency_global_profiles.png", log=False),
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
        tissueSettings = run.config["fixed"]["tissue"]
        for y in (tissueSettings["boundaryYmm"], tissueSettings["boundaryYmm"] - tissueSettings["lesionWidthMm"]):
            axis.axhline(y, color="k", lw=0.8, ls="--")
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
        axis.axvline(run.config["fixed"]["tissue"]["lesionWidthMm"], color="k", lw=0.6, ls="--")
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
            index = int(chosen.iloc[0].traceIndex)
            trace = archive["cleanMv"][index]
            axis.plot(archive["windowStartMs"][index] + np.arange(len(trace)) * float(archive["stepMs"]), trace, color=colour, label=condition)
        axis.set(title=f"{kind} {label}, centre y = 0, gap {gap:g} mm", xlabel="time (ms)", ylabel="mV")
        axis.legend(fontsize=7)
    fig.tight_layout()
    path = folder / "04_example_traces.png"
    fig.savefig(path, dpi=150); plt.close(fig)
    return path
