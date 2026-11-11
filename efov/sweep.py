"""Parameter sweeps (FM-17): the baseline plus one-at-a-time and named variants, with a shared simulation cache.

Sweep file (config/sweeps/<name>.json):
  {"modelConfig": "../model.json",
   "oneAtATime": ["wallMm", "waveAngleDeg", ...],          each planned value of each variable, alone
   "named": {"roof": {"region": "roof", "wallMm": 1.6, "fibres": {...}}}}
Every variant is model.json's baseline with its overrides. Amplitudes are in the model's physical
units in every variant (no calibration, FM-25), so amplitude differences are predictions.
Simulations are cached under <out>/simulationCache/<simulation fingerprint>/: a variant that changes
only electrodes, noise or analysis reuses the baseline's openCARP runs, physiology and tuning.
Lead fields are cached under <out>/leadFieldCache/<conductor fingerprint>/ (FM-23): wave-angle, CV,
electrode, noise and analysis variants reuse the baseline's.
After the variants, noiseDoubling.csv holds the pre-registered primary test of FM-25: the change in
every change point between successive noise levels of the doubling series, beside the shift predicted
if the plateau were set by noise alone (FM-38), and the relative noise floor of each level (noise-only
bipolar PP / healthy bipolar median, FM-44), the scale on which a change point is comparable with Wang's.
A variant that changes the simulation is the only user of its simulation in a one-at-a-time sweep, so its raw
output is compacted after its report (FM-20); the baseline simulation, shared by the electrode, noise
and analysis variants, is never compacted.
A failed variant is recorded with its error and the sweep continues; failures are never pooled.
"""
import json
from pathlib import Path
import traceback

from .config import loadConfig
from .io import loadJson, saveJson
from .pipeline import Run, sharedFolder


def variantName(variable, value):
    text = json.dumps(value, sort_keys=True) if isinstance(value, (dict, list)) else str(value)
    for old, new in (('"', ""), (" ", ""), (":", "-"), (",", "_"), ("{", ""), ("}", ""), ("[", ""), ("]", "")):
        text = text.replace(old, new)
    return f"{variable}={text}"


def variants(sweep, modelConfigPath):
    model = loadJson(modelConfigPath)
    result = {"baseline": {}}
    for variable in sweep.get("oneAtATime", []):
        if variable not in model["sweep"]:
            raise KeyError(f"{variable} is not a sweep variable of {modelConfigPath}")
        for value in model["sweep"][variable]["planned"]:
            result[variantName(variable, value)] = {variable: value}
    for name, overrides in sweep.get("named", {}).items():
        result[name] = overrides
    return result


def configFor(modelConfigPath, overrides, compact):
    config = loadConfig(modelConfigPath, overrides)
    config["fixed"]["compactAfterReport"] = compact
    return config


def noiseDoublingTable(out, modelConfigPath):
    """Change points of the baseline and the noise variants, and their shift per doubling of noise (FM-25).

    Uses the noisy realisations only (repeat 0 is clean and identical in every noise variant). For
    each dataset family (gap, PF definition) and each of Wang's change points (her keys: Location,
    Metric, Electrode): the median over realisations of cp_mm_mean at every noise level, and the
    difference between successive levels. Returns None if no change-point table exists yet.
    """
    import pandas as pd
    from .export import parseDatasetName
    model = loadJson(modelConfigPath)
    levels = {"baseline": model["sweep"]["noiseSdMv"]["baseline"]}
    levels.update({variantName("noiseSdMv", v): v for v in model["sweep"]["noiseSdMv"]["planned"]})
    parts = []
    for name, sd in levels.items():
        path = Path(out) / "variants" / name / "changePoints" / "SuppResults5_change_points_all.csv"
        if path.exists():
            table = pd.read_csv(path)
            parsed = table.dataset.map(parseDatasetName)
            parts.append(table.assign(noiseSdMv=sd, gapMm=[q[0] for q in parsed], noiseRepeat=[q[1] for q in parsed],
                                      pfDefinition=[q[2] for q in parsed]))
    if not parts:
        return None
    table = pd.concat(parts, ignore_index=True)
    table = table[table.noiseRepeat > 0]
    keys = [k for k in ("gapMm", "pfDefinition", "Location", "Metric", "Electrode") if k in table.columns]
    summary = table.groupby(keys + ["noiseSdMv"]).cp_mm_mean.median().rename("cpMedianMm").reset_index().sort_values(keys + ["noiseSdMv"])
    summary["shiftFromHalfNoiseMm"] = summary.groupby(keys).cpMedianMm.diff()
    summary = summary.merge(predictedNoiseShift(out, levels, keys), on=keys + ["noiseSdMv"], how="left")
    floors = relativeNoiseFloors(out, levels)
    if not floors.empty:
        summary = summary.merge(floors, on=["gapMm", "noiseSdMv"], how="left")
    summary.to_csv(Path(out) / "noiseDoubling.csv", index=False)
    return summary


def relativeNoiseFloors(out, levels):
    """FM-44: per noise level and gap, noise-only bipolar PP / healthy bipolar median of that variant (amplitudes.json).

    The change point is to be read against this ratio, not against the noise SD: the amplitudes are not
    calibrated (FM-25), so only the ratio is comparable with a clinical recording. clinicalReferenceRatio
    repeats Wang's detection limit / healthy median for orientation (an assumption, see efov.measurements).
    """
    import pandas as pd
    rows = []
    for name, sd in levels.items():
        path = Path(out) / "variants" / name / "measurements" / "amplitudes.json"
        if not path.exists():
            continue
        amplitudes = loadJson(path)
        for row in amplitudes.get("relativeNoiseFloor", []):
            rows.append(dict(noiseSdMv=sd, gapMm=row["gapMm"], noiseOnlyBipolarPpMv=amplitudes["noiseOnlyBipolarPpMv"],
                             relativeNoiseFloor=row["relativeNoiseFloor"], clinicalReferenceRatio=amplitudes.get("clinicalReferenceRatio")))
    return pd.DataFrame(rows, columns=["noiseSdMv", "gapMm", "noiseOnlyBipolarPpMv", "relativeNoiseFloor", "clinicalReferenceRatio"])


def predictedNoiseShift(out, levels, keys):
    """FM-38: shift of each change point from the previous noise level predicted if the plateau is noise.

    Uses the median piecewise slope beta1 and change point of the previous level (efov.report.predictedShiftMm),
    with the actual ratio of the two noise SDs. Assumes the decay segment is unchanged by noise.
    """
    import numpy as np
    import pandas as pd
    from .export import parseDatasetName
    from .report import predictedShiftMm
    parts = []
    for name, sd in levels.items():
        folder = Path(out) / "variants" / name / "changePoints"
        if not (folder / "Table5_bayesian_posteriors_all.csv").exists() or not (folder / "SuppResults5_change_points_all.csv").exists():
            continue
        posteriors, cps = pd.read_csv(folder / "Table5_bayesian_posteriors_all.csv"), pd.read_csv(folder / "SuppResults5_change_points_all.csv")
        wide = posteriors[posteriors.term == "beta1"][["dataset", "Location", "Metric", "Electrode", "model", "mean"]].rename(columns={"mean": "beta1"})
        wide = wide.merge(cps[["dataset", "Location", "Metric", "Electrode", "cp_mm_mean"]], on=["dataset", "Location", "Metric", "Electrode"])
        parsed = wide.dataset.map(parseDatasetName)
        wide = wide.assign(noiseSdMv=sd, gapMm=[q[0] for q in parsed], noiseRepeat=[q[1] for q in parsed], pfDefinition=[q[2] for q in parsed])
        parts.append(wide[wide.noiseRepeat > 0])
    if not parts:
        return pd.DataFrame(columns=keys + ["noiseSdMv", "predictedShiftFromHalfNoiseMm"])
    table = pd.concat(parts, ignore_index=True)
    medians = table.groupby(keys + ["model", "noiseSdMv"]).agg(beta1=("beta1", "median"), cp=("cp_mm_mean", "median")).reset_index()
    medians = medians.sort_values(keys + ["noiseSdMv"])
    previous = medians.groupby(keys)[["beta1", "cp", "noiseSdMv"]].shift(1)
    medians["predictedShiftFromHalfNoiseMm"] = [
        predictedShiftMm(model, b, c, sd / psd) if np.isfinite(psd) else np.nan
        for model, b, c, sd, psd in zip(medians.model, previous.beta1, previous.cp, medians.noiseSdMv, previous.noiseSdMv)]
    return medians[keys + ["noiseSdMv", "predictedShiftFromHalfNoiseMm"]]


class Sweep:
    """One sweep: the baseline and its variants, run one at a time (used by runSweep and run_all.ipynb).

    Every variant runs in <out>/variants/<name> with the shared simulation and lead-field caches, so the
    baseline run is also the reference of every comparison, and a stage that is already complete with the
    same settings is reused. Outcomes go to <out>/sweepStatus.json; a failed variant is recorded with its
    error and never enters a summary as if it had succeeded.
    """

    def __init__(self, sweepPath, outputFolder, processCount=1):
        self.path, self.out = Path(sweepPath).resolve(), Path(outputFolder).resolve()
        self.definition = loadJson(self.path)
        self.modelConfigPath = (self.path.parent / self.definition["modelConfig"]).resolve()
        self.plan = variants(self.definition, self.modelConfigPath)
        self.np, self.cache = int(processCount), self.out / "simulationCache"
        self.statusPath = self.out / "sweepStatus.json"
        self.status = loadJson(self.statusPath) if self.statusPath.exists() else {}
        self.baseline = Run(configFor(self.modelConfigPath, {}, False), self.out / "variants" / "baseline", self.np, self.cache)

    def namesFor(self, variable):
        """Variant names of one sweep variable (one-at-a-time) or of one named variant, in sweep order."""
        if variable in self.definition.get("named", {}):
            return [variable]
        if variable not in self.definition.get("oneAtATime", []):
            return []
        return [name for name, overrides in self.plan.items() if list(overrides) == [variable]]

    def runBaseline(self):
        result = self.baseline.all()
        self._record("baseline", dict(status="complete", overrides={}))
        return result

    def runVariant(self, name, raiseErrors=False):
        """Run one variant (the baseline must be complete); returns its status entry."""
        overrides = self.plan[name]
        if self.status.get(name, {}).get("status") == "complete":
            return self.status[name]
        try:
            ownSimulation = sharedFolder(configFor(self.modelConfigPath, overrides, False), self.out, self.cache) != self.baseline.shared
            self.variantRun(name, ownSimulation).all()
            entry = dict(status="complete", overrides=overrides)
        except Exception as error:                                     # recorded, never silently pooled
            entry = dict(status="failed", overrides=overrides, error=f"{type(error).__name__}: {error}",
                         traceback=traceback.format_exc())
            if raiseErrors:
                self._record(name, entry)
                raise
        return self._record(name, entry)

    def variantRun(self, name, compact=False):
        return Run(configFor(self.modelConfigPath, self.plan[name], compact), self.out / "variants" / name, self.np, self.cache)

    def runAll(self, only=None):
        self.runBaseline()
        for name in self.plan:
            if name != "baseline" and (not only or name in only):
                self.runVariant(name)
        noiseDoublingTable(self.out, self.modelConfigPath)
        return self.status

    def _record(self, name, entry):
        self.status[name] = entry
        saveJson(self.statusPath, self.status)
        return entry

    def summary(self, names=None):
        """Long table of headline outcomes per variant (see variantSummary); failed variants keep a status row."""
        import pandas as pd
        rows = []
        for name in names or ["baseline"] + [n for n in self.plan if n != "baseline"]:
            entry = self.status.get(name, {"status": "not run"})
            if entry["status"] != "complete":
                rows.append(dict(variant=name, quantity="status", value=entry["status"], error=entry.get("error")))
                continue
            rows += [dict(variant=name, **row) for row in variantSummary(Path(self.out) / "variants" / name)]
        return pd.DataFrame(rows)


def variantSummary(folder):
    """Headline outcomes of one completed run folder, as rows of (quantity, kind, gapMm, value).

    Definitions (all from the run's own outputs; nothing recomputed):
      * modelAccepted, eligibleForInterpretation: acceptance.json, changePoints stage;
      * surfaceCvMPerS: physiology, last-beat surface CV of the lesion condition (reported, not gated);
      * medianFarPpMv: measurements amplitudes, median clean PP in far viable tissue (model units);
      * relativeNoiseFloor: noise-only bipolar PP / healthy bipolar median (FM-44);
      * cpNoisyMm (Voltage / Peak frequency): Wang's change point (posterior mean, mm from the scar boundary),
        median over the noisy realisations (repeats >= 1) of each gap, EnSite-style PF: the quantity
        comparable with her clinical change point, and the one the noise experiment acts on (FM-45);
      * cpMm: the same on the noise-free dataset (repeat 0). A reference only: without noise the lesion
        profile has no plateau, so this fit reflects her 0.01 mV exclusion and the decay form;
      * nearFieldMass, epicardialMass: ground truth, median over lesion recordings of the |c|-mass fractions
        within the 4 mm near-field convention (FM-30) and in the epicardial half (FM-15);
      * p80DepthMm: ground truth, median depth reached by the smallest element set holding 80 % of the mass.
    """
    import pandas as pd
    from .export import parseDatasetName
    folder = Path(folder)
    stages = folder / "stages"
    rows = []
    acceptance = loadJson(folder / "acceptance.json") if (folder / "acceptance.json").exists() else {}
    rows.append(dict(quantity="modelAccepted", value=acceptance.get("modelAccepted")))
    changes = loadJson(stages / "changePoints.json").get("result", {}) if (stages / "changePoints.json").exists() else {}
    rows.append(dict(quantity="eligibleForInterpretation", value=changes.get("eligibleForInterpretation")))
    shared = Path(loadJson(folder / "model.json")["sharedSimulationFolder"])
    physiology = loadJson(shared / "stages" / "physiology.json").get("result", {}) if (shared / "stages" / "physiology.json").exists() else {}
    lesion = physiology.get("conditions", {}).get("lesion", {})
    rows.append(dict(quantity="surfaceCvMPerS", value=lesion.get("lastBeatSurfaceSpeedMPerS")))
    amplitudes = folder / "measurements" / "amplitudes.json"
    for row in (loadJson(amplitudes).get("modelFarViable", []) if amplitudes.exists() else []):
        rows.append(dict(quantity="medianFarPpMv", kind=row["kind"], gapMm=row["gapMm"], value=row["medianCleanPpMv"]))
    for row in (loadJson(amplitudes).get("relativeNoiseFloor", []) if amplitudes.exists() else []):
        rows.append(dict(quantity="relativeNoiseFloor", kind="bi", gapMm=row["gapMm"], value=row["relativeNoiseFloor"]))
    table = folder / "changePoints" / "SuppResults5_change_points_all.csv"
    if table.exists():
        cps = pd.read_csv(table)
        parsed = cps.dataset.map(parseDatasetName)
        cps = cps.assign(gapMm=[q[0] for q in parsed], repeat=[q[1] for q in parsed], pf=[q[2] for q in parsed])
        for _, r in cps[(cps.repeat == 0) & (cps.pf == "ensite")].iterrows():
            rows.append(dict(quantity=f"cpMm {r.Metric}", kind=r.Electrode, gapMm=r.gapMm, value=r.cp_mm_mean))
        noisy = cps[(cps.repeat > 0) & (cps.pf == "ensite")]
        for (metric, electrode, gap), part in noisy.groupby(["Metric", "Electrode", "gapMm"]):
            rows.append(dict(quantity=f"cpNoisyMm {metric}", kind=electrode, gapMm=gap, value=float(part.cp_mm_mean.median())))
    truth = folder / "groundTruth" / "summary.csv"
    if truth.exists():
        gt = pd.read_csv(truth)
        gt = gt[gt.condition == "lesion"]
        mass = gt[[c for c in gt.columns if c.startswith("mass_") and c != "mass_lesionElements"]].sum(axis=1)
        gt = gt.assign(nearFieldMass=(gt.mass_nearEndocardial + gt.mass_nearEpicardial) / mass,
                       epicardialMass=(gt.mass_nearEpicardial + gt.mass_farEpicardial) / mass)
        for (kind, gap), part in gt.groupby(["kind", "gapMm"]):
            for quantity in ("nearFieldMass", "epicardialMass", "p80_maximumDepthMm"):
                rows.append(dict(quantity=quantity.replace("p80_maximumDepthMm", "p80DepthMm"), kind=kind, gapMm=gap,
                                 value=float(part[quantity].median())))
    return rows


def runSweep(sweepPath, outputFolder, processCount=1, only=None):
    return Sweep(sweepPath, outputFolder, processCount).runAll(only)


# ------------------------------------------------------------- comparison of one experiment with the baseline
# cpNoisyMm (median over noisy realisations) is the quantity comparable with Wang's; cpMm (noise-free) is a
# reference only: without noise the lesion profile has no plateau for her piecewise model to find (FM-45).
HEADLINE = [("cpNoisyMm Voltage", "Bipolar"), ("cpNoisyMm Voltage", "Omnipolar"), ("cpMm Voltage", "Bipolar"),
            ("cpMm Peak frequency", "Bipolar"), ("cpNoisyMm Peak frequency", "Bipolar"),
            ("medianFarPpMv", "bi"), ("nearFieldMass", "bi"), ("epicardialMass", "bi")]


def experimentTable(sweep, variable):
    """Baseline and every variant of one variable, one row per variant, headline quantities as columns.

    Columns are '<quantity> | <kind> | gap <g> mm'; rows are in order of value for a numeric variable
    (baseline in place, flagged by isBaseline), otherwise baseline first. Gap variants have only their
    own gap, so their columns are blank elsewhere. Status and acceptance are kept so failures stay visible.
    """
    import pandas as pd
    names = ["baseline"] + sweep.namesFor(variable)
    long = sweep.summary(names)
    if long.empty:
        return long
    rows = {}
    for name in names:
        part = long[long.variant == name]
        baselineValue = sweep.baseline.config["values"].get(variable)
        value = baselineValue if name == "baseline" else sweep.plan[name].get(variable, sweep.plan[name])
        row = {"value": json.dumps(value), "isBaseline": name == "baseline"}
        flags = part[part.kind.isna()] if "kind" in part else part
        for _, r in flags.iterrows():
            row[r.quantity] = r.value
        for quantity, kind in HEADLINE:
            for _, r in part[(part.quantity == quantity) & (part.get("kind") == kind)].iterrows():
                row[f"{quantity} | {kind} | gap {r.gapMm:g} mm"] = r.value
        rows[name] = row
    table = pd.DataFrame(rows).T
    numeric = pd.to_numeric(table["value"], errors="coerce")
    if numeric.notna().all():                         # numeric variable: rows in order of value, baseline in place
        table = table.iloc[numeric.argsort(kind="stable")]
    return table


def experimentFigure(sweep, variable, path):
    """Headline quantities against the variable's values (baseline marked), one panel per quantity, a line per gap."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    table = experimentTable(sweep, variable)
    if table.empty:
        return None
    labels = [f"{v}\n(baseline)" if b else v for v, b in zip(table["value"], table["isBaseline"])]
    marker = int(np.flatnonzero(table["isBaseline"].to_numpy(bool))[0])
    fig, axes = plt.subplots(2, 4, figsize=(17, 6.5))
    for ax, (quantity, kind) in zip(axes.ravel(), HEADLINE):
        columns = [c for c in table.columns if c.startswith(f"{quantity} | {kind} |")]
        for column in columns:
            values = pd_numeric(table[column])
            ax.plot(range(len(labels)), values, "o-", label=column.split("|")[-1].strip())
        ax.axvline(marker, color="0.7", lw=0.8)
        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels([l if len(l) < 24 else l[:22] + "..." for l in labels], rotation=30, ha="right", fontsize=7)
        ax.set_title(f"{quantity} ({kind})", fontsize=9)
        if columns:
            ax.legend(fontsize=7)
    fig.suptitle(f"{variable}: baseline (grey line) and variants; EnSite-style PF; cpNoisy = median over noisy realisations")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path


def pd_numeric(series):
    import pandas as pd
    return pd.to_numeric(series, errors="coerce").to_numpy(float)
