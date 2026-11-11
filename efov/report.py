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
    """One table (FM-38, FM-45): Wang's change point and decay slope beside the boundary-matched ground truth.

    Per Location, Metric, Electrode and gap (EnSite-style PF):
      * cp_mm_noisy_median (min, max, realisations): Wang's change point over the noisy realisations
        (repeats >= 1). This is the row comparable with her clinical change point: her piecewise model is a
        decay followed by a plateau, and in the model the plateau exists only through noise;
      * cp_mm_clean: the same fit on the noise-free dataset, a reference only (no plateau to find: the fit
        reflects her 0.01 mV exclusion and the misfit of the decay form);
      * beta1, logDropAtCp = |beta1| x cp (how far the fitted decay has fallen at the change point) and the
        shift predicted per doubling of noise if the plateau is noise: medians over the noisy realisations;
      * gt_mass_<level>pct_mm: distance inside the lesion where the intact recording's retained |c|-mass first
        falls to the level (FM-37; median over the recordings of the kind). The |c|-mass is the primary
        ground-truth measure (FM-43); signed-share crossings are in the table too, NaN where ill-conditioned.
    Returns the table, or None before the change points exist.
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
    fits = cps[cps.pfDefinition == "ensite"].merge(slopes, on=keys, how="left")
    fits["logDropAtCp"] = (fits.beta1 * fits.cp).abs()
    fits["predictedShiftPerNoiseDoublingMm"] = [predictedShiftMm(m, b, c) for m, b, c in zip(fits.model, fits.beta1, fits.cp_mm_mean)]
    group = ["Location", "Metric", "Electrode", "gapMm"]
    noisy, clean = fits[fits.noiseRepeat > 0], fits[fits.noiseRepeat == 0]
    table = noisy.groupby(group).agg(
        cp_mm_noisy_median=("cp_mm_mean", "median"), cp_mm_noisy_min=("cp_mm_mean", "min"), cp_mm_noisy_max=("cp_mm_mean", "max"),
        noisyRealisations=("cp_mm_mean", "count"), value_at_cp_noisy_median=("value_at_cp_mean", "median"),
        model=("model", lambda m: "/".join(sorted(set(m.dropna())))), beta1=("beta1", "median"), logDropAtCp=("logDropAtCp", "median"),
        predictedShiftPerNoiseDoublingMm=("predictedShiftPerNoiseDoublingMm", "median")).reset_index()
    reference = clean[group + ["cp_mm_mean", "model"]].rename(columns={"cp_mm_mean": "cp_mm_clean", "model": "model_clean"})
    table = table.merge(reference, on=group, how="outer")
    table["kind"] = table.Electrode.map(KIND_OF_ELECTRODE)
    crossingsPath = run.out / "groundTruth" / "boundaryCrossings.csv"
    if crossingsPath.exists():
        crossings = pd.read_csv(crossingsPath)
        wide = crossings.groupby(["kind", "gapMm", "measure", "level"]).distanceMm.median().reset_index()
        wide["column"] = "gt_" + wide.measure + "_" + (wide.level * 100).round().astype(int).astype(str) + "pct_mm"
        wide = wide.pivot_table(index=["kind", "gapMm"], columns="column", values="distanceMm", dropna=False).reset_index()
        table = table.merge(wide, on=["kind", "gapMm"], how="left")
    columns = group + ["cp_mm_noisy_median", "cp_mm_noisy_min", "cp_mm_noisy_max", "noisyRealisations", "value_at_cp_noisy_median", "model",
                       "beta1", "logDropAtCp", "predictedShiftPerNoiseDoublingMm", "cp_mm_clean", "model_clean"]
    columns += sorted(c for c in table.columns if c.startswith("gt_"))
    table = table[[c for c in columns if c in table.columns]].sort_values(["Metric", "Electrode", "gapMm"])
    table.to_csv(run.out / "efovComparison.csv", index=False)
    return table


# The 50 % level is left out of the report table: for a wave parallel to the boundary, mirror symmetry puts
# the retained share of a symmetric recording at 0.5 at the boundary whatever the model's parameters (FM-43).
COMPARISON_COLUMNS = ("Metric", "Electrode", "gapMm", "cp_mm_noisy_median", "cp_mm_noisy_min", "cp_mm_noisy_max", "cp_mm_clean",
                      "beta1", "logDropAtCp", "predictedShiftPerNoiseDoublingMm", "gt_mass_20pct_mm", "gt_mass_10pct_mm")


def comparisonLines(table):
    if table is None or table.empty:
        return ["No comparison table (change points or ground truth missing)."]
    shown = [c for c in COMPARISON_COLUMNS if c in table.columns]
    lines = ["| " + " | ".join(shown) + " |", "| " + " | ".join("---" for _ in shown) + " |"]
    for row in table[shown].itertuples(index=False):
        lines.append("| " + " | ".join(fmt(v) if isinstance(v, float) else str(v) for v in row) + " |")
    return lines


def noiseFloorLines(amplitudes):
    """Relative noise floor of this run beside the clinical reference ratio (FM-44)."""
    if "relativeNoiseFloor" not in amplitudes:
        return []
    lines = [f"Noise {amplitudes['noiseSdMv']:g} mV per electrode gives a noise-only bipolar PP of {amplitudes['noiseOnlyBipolarPpMv']:.3g} mV "
             "(median, band-passed, one measurement window). Relative noise floor = that PP / healthy bipolar median: "
             + "; ".join(f"gap {r['gapMm']:g} mm: {r['relativeNoiseFloor']:.3g}" for r in amplitudes["relativeNoiseFloor"]) + "."]
    if amplitudes.get("clinicalReferenceRatio"):
        lines.append(f"Clinical reference: Wang's detection limit / healthy bipolar median = {amplitudes['wangDetectionLimitMv']:g} / "
                     f"{amplitudes['wangBipolarMedianMv']:g} = {amplitudes['clinicalReferenceRatio']:.3g} (assumes clinical noise PP is of the "
                     "order of the detection limit; not measured). A change point is comparable with Wang's only at a matching ratio: "
                     "with a noise-set plateau it moves by ln 2 / |beta1| per factor of 2 in this ratio (FM-38, FM-44).")
    return lines + [""]


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
    lines += noiseFloorLines(amplitudes)
    lines += ["## Change points (Wang's code, one slab)", "",
              f"Eligible for interpretation: {fmt(changes['eligibleForInterpretation'])} "
              f"(prerequisites: {', '.join(f'{k} {fmt(v)}' for k, v in changes['prerequisites'].items())}).", ""]
    lines += changePointTable(run) + [""]
    lines += ["## Change point beside the boundary-matched ground truth (FM-37, FM-38, FM-43, FM-45)", "",
              "EnSite-style PF. `cp_mm_noisy_*`: Wang's change point over the noisy realisations (median, min, max), the values "
              "comparable with her clinical change point at this run's relative noise floor (above). `cp_mm_clean`: the noise-free "
              "fit, a reference only (no plateau exists without noise). `logDropAtCp` = |beta1| x cp. "
              "`predictedShiftPerNoiseDoublingMm`: the shift expected if the plateau is noise (compare noiseDoubling.csv). "
              "`gt_mass_*`: distance inside the lesion where the intact recording's retained |c|-mass first falls to the level "
              "(median over the recordings of the kind). Unipolar ground-truth distances depend on the position of the reference "
              "(ground) in the bath and are model-specific. Full table, with signed-share crossings: `efovComparison.csv`.", ""]
    lines += comparisonLines(efovComparison(run)) + [""]
    lines += ["## Ground truth", "", f"{gt['recordings']} recordings ({gt['leadFields']} lead fields, the same as the "
              f"electrograms'); share-sum deviation {gt['shareSumMaxDeviation']:.2g} (an implementation identity, not a validation); "
              f"{gt.get('signedShareIllConditioned', 0)} recordings flagged as ill-conditioned for the signed share (FM-43).",
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
