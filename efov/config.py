"""Load, check and derive the model configuration.

Two files define the model (FM-01, FM-02):
  config/model.par   every openCARP setting (read with efov.parfile)
  config/model.json  'sweep' variables (baseline + planned values) and 'fixed' non-openCARP settings

loadConfig(path, overrides) returns one dict:
  values        the sweep variables of this run: baselines with the overrides applied
  fixed         the fixed settings
  par           model.par values with this run's openCARP overrides applied (text, as in the file)
  parOverrides  {openCARP key: value} that this run changes in model.par
This module also derives the few quantities that follow from the configuration and fingerprints
the inputs of each pipeline stage, so stale outputs are never reused.
"""
import copy
import hashlib
import json
from pathlib import Path

import numpy as np

from .io import loadJson, saveJson
from .parfile import parValue, readPar

REPOSITORY = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = REPOSITORY / "config" / "model.json"
TUNING_RECORD = REPOSITORY / "config" / "tuning.json"
CONDUCTIVITY_NAMES = ("g_il", "g_it", "g_in", "g_el", "g_et", "g_en")


# ----------------------------------------------------------------- loading
def loadConfig(path=DEFAULT_CONFIG, overrides=None, extraParOverrides=None):
    """Configuration of one run: model.json baselines + sweep overrides, model.par + derived overrides."""
    path = Path(path)
    model = loadJson(path)
    overrides = dict(overrides or {})
    unknown = set(overrides) - set(model["sweep"])
    if unknown:
        raise KeyError(f"Not sweep variables of {path.name}: {sorted(unknown)}")
    config = dict(modelName=model["modelName"], sweep=model["sweep"], fixed=model["fixed"],
                  provenance=model.get("provenance", {}), source=str(path),
                  modelPar=str(path.parent / model["parFile"]), leadFieldPar=str(path.parent / model["leadFieldParFile"]),
                  values={name: copy.deepcopy(spec["baseline"]) for name, spec in model["sweep"].items()},
                  overrides=copy.deepcopy(overrides))
    config["values"].update(copy.deepcopy(overrides))
    config["parOverrides"] = parOverridesFor(config, extraParOverrides)
    config["par"] = readPar(config["modelPar"])
    config["par"].update({k: str(v) for k, v in config["parOverrides"].items()})
    validateConfig(config)
    return config


def parOverridesFor(config, extra=None):
    """openCARP overrides implied by the sweep values (scar type, ionic model) plus any extra (tuned beta)."""
    values, sweep = config["values"], config["sweep"]
    options = sweep["scar"]["options"]
    if values["scar"] not in options:
        raise ValueError(f"scar must be one of {sorted(options)}")
    result = dict(options[values["scar"]])
    if values["ionicModel"] != sweep["ionicModel"]["baseline"]:
        result["imp_region[0].im"] = values["ionicModel"]
    result.update(extra or {})
    return result


def par(config, key, kind=float):
    """A model.par value of this run (overrides applied)."""
    return parValue(config["par"], key, kind)


def withParOverrides(config, extra):
    """The same run with further openCARP overrides (e.g. the tuned beta)."""
    return loadConfig(config["source"], config["overrides"], {**config["parOverrides"], **extra})


# ----------------------------------------------------------------- derived quantities
def slabSizeMm(config):
    return np.array([*config["fixed"]["tissue"]["sizeMm"], config["values"]["wallMm"]], float)


def spacingMm(config):
    t = config["fixed"]["tissue"]
    return np.array([t["spacingInPlaneMm"], t["spacingInPlaneMm"], t["spacingTransmuralMm"]], float)


def rangeMm(spec):
    if isinstance(spec, list):
        return np.asarray(spec, float)
    count = int(round((spec["toMm"] - spec["fromMm"]) / spec["stepMm"])) + 1
    return np.round(spec["fromMm"] + np.arange(count) * spec["stepMm"], 9)


def centreYmm(config):
    return rangeMm(config["fixed"]["electrodes"]["centreYmm"])


def groundTruthCentresYmm(config):
    return rangeMm(config["fixed"]["groundTruth"]["placementsYmm"])


def cycleLengthMs(config):
    return par(config, "stim[0].ptcl.bcl")


def beatCount(config):
    return par(config, "stim[0].ptcl.npls", int)


def stimulusDurationMs(config):
    return par(config, "stim[0].ptcl.duration")


def sampleStepMs(config):
    return par(config, "spacedt")


def timeStepUs(config):
    return par(config, "dt")


def gridOffsetsMm(spacing):
    """Electrode offsets of a 4 x 4 grid with this centre spacing (mm)."""
    return (np.arange(4) - 1.5) * float(spacing)


def restartTimeMs(config):
    """Prepacing checkpoint: early enough that phi_e exists for every frame the analysis reads.

    The analysis reads the last beat from (window start - 2 x filter history); the earliest window
    start is stimulus end + minimumAfterStimulus. A margin is subtracted and the result floored to 10 ms.
    """
    if not config["fixed"]["numerics"]["prepacing"]:
        return None
    analysis = config["fixed"]["analysis"]
    earliest = (beatCount(config) - 1) * cycleLengthMs(config) + stimulusDurationMs(config) + analysis["window"]["minimumAfterStimulusMs"]
    restart = earliest - 2 * analysis["filterHistoryMs"] - config["fixed"]["numerics"]["prepacingMarginMs"]
    restart = float(np.floor(restart / 10.0) * 10.0)
    return restart if restart > 0 else None


def _sweepValues(config, name):
    """Baseline, planned and current values of a sweep variable, flattened (lists of gaps are expanded)."""
    spec = config["sweep"][name]
    flat = []
    for item in [spec["baseline"]] + list(spec["planned"]) + [config["values"][name]]:
        flat += [float(v) for v in item] if isinstance(item, list) else [float(item)]
    return flat


def phieBoxMm(config):
    """(lower, upper) corners of the bath box where openCARP writes phi_e (mm).

    Sized for every electrode placement of the one-at-a-time sweep plan (FM-17, FM-20): each extent
    is the largest over variants that change one electrode variable from its baseline, plus one mesh
    step. A sweep within the Methodology plan therefore reuses the same simulation.
    """
    sweep, e, spacing = config["sweep"], config["fixed"]["electrodes"], spacingMm(config)
    base = {name: sweep[name]["baseline"] for name in ("gapsMm", "electrodeRadiusMm", "electrodeLengthMm", "gridSpacingMm")}
    baseGap = max(base["gapsMm"])
    reach = lambda gap, radius: gap + 2 * radius
    height = max([reach(g, base["electrodeRadiusMm"]) for g in _sweepValues(config, "gapsMm")]
                 + [reach(baseGap, r) for r in _sweepValues(config, "electrodeRadiusMm")]
                 + [reach(max(config["values"]["gapsMm"]), config["values"]["electrodeRadiusMm"])])
    lateral = lambda s, length: 1.5 * s + length / 2
    halfX = abs(e["centreXmm"]) + max([lateral(s, base["electrodeLengthMm"]) for s in _sweepValues(config, "gridSpacingMm")]
                                      + [lateral(base["gridSpacingMm"], length) for length in _sweepValues(config, "electrodeLengthMm")])
    halfY = np.abs(centreYmm(config)).max() + max(1.5 * s for s in _sweepValues(config, "gridSpacingMm")) \
        + max(_sweepValues(config, "electrodeRadiusMm"))
    top = slabSizeMm(config)[2] / 2
    return (np.array([-halfX - spacing[0], -halfY - spacing[1], top]),
            np.array([halfX + spacing[0], halfY + spacing[1], top + height + spacing[2]]))


def meshNodeCount(config):
    """Nodes of the tissue + bath mesh (regular block), for planning."""
    size, spacing, bath = slabSizeMm(config), spacingMm(config), config["fixed"]["bath"]["thicknessMm"]
    extent = size + np.array([2 * bath, 2 * bath, bath * (2 if config["values"]["epicardialMedium"] == "blood" else 1)])
    return int(np.prod(np.round(extent / spacing).astype(int) + 1))


# ----------------------------------------------------------------- validation
def validateConfig(config):
    """Reject inconsistent inputs before any simulation. Raises ValueError with the reason."""
    def need(condition, message):
        if not condition:
            raise ValueError(message)

    v, f = config["values"], config["fixed"]
    size, spacing = slabSizeMm(config), spacingMm(config)
    need(np.all(size > 0) and np.all(spacing > 0), "Slab size and spacing must be positive")
    need(np.allclose(size / spacing, np.round(size / spacing)), "Every slab dimension must be a multiple of its mesh spacing")
    need(size[2] / spacing[2] >= 4, "Use at least four elements through the wall")
    need(np.isclose(f["bath"]["thicknessMm"] / spacing[0], round(f["bath"]["thicknessMm"] / spacing[0]))
         and np.isclose(f["bath"]["thicknessMm"] / spacing[2], round(f["bath"]["thicknessMm"] / spacing[2])),
         "Bath thickness must be a multiple of both mesh spacings")
    need(0 < f["bath"]["groundCubeMm"] < f["bath"]["thicknessMm"], "Ground cube must be smaller than the bath")
    need(v["epicardialMedium"] in ("none", "blood"), "epicardialMedium must be 'none' or 'blood'")
    need(-size[1] / 2 < f["tissue"]["boundaryYmm"] < size[1] / 2, "Lesion boundary must lie inside the slab")
    need(v["region"] in ("inferior", "roof"), "region must be 'inferior' or 'roof'")
    need(0 <= v["waveAngleDeg"] <= 90, "waveAngleDeg must lie in [0, 90]")
    g = {name: par(config, f"gregion[0].{name}") for name in CONDUCTIVITY_NAMES}
    need(all(value > 0 for value in g.values()), "Tissue conductivities must be positive")
    need(g["g_it"] == g["g_in"] and g["g_et"] == g["g_en"],
         "One fibre vector per element: tissue must be transversely isotropic (t = n)")
    need(par(config, "imp_region[0].cellSurfVolRatio") > 0, "cellSurfVolRatio must be positive")
    need(par(config, "bidomain", int) == 2, "The final model uses pseudo-bidomain (bidomain = 2; DL-2026-11-02-01)")
    need(np.isclose(par(config, "tend"), beatCount(config) * cycleLengthMs(config)), "tend must equal npls x bcl")
    need(beatCount(config) >= 3, "Physiology checks compare the last three beats: npls >= 3")
    step, dt = sampleStepMs(config), timeStepUs(config)
    need(np.isclose(step * 1000 / dt, round(step * 1000 / dt)) and step * 1000 >= dt, "spacedt must be a multiple of dt")
    rate = 1000.0 / step
    a = f["analysis"]
    need(rate > a["acquisitionRateHz"], "The simulation output rate must exceed the acquisition rate")
    from fractions import Fraction
    ratio = Fraction(a["acquisitionRateHz"] / rate).limit_denominator(1000)
    need(np.isclose(float(ratio), a["acquisitionRateHz"] / rate), "Acquisition / simulation rate must be a simple ratio")
    band = a["differentialBandHz"]
    need(0 < band[0] < band[1] < a["acquisitionRateHz"] / 2, "Differential band must lie below the acquisition Nyquist frequency")
    pf = a["peakFrequency"]
    need(pf["minimumHz"] >= band[0] and pf["maximumHz"] <= band[1], "Peak-frequency range must lie inside the analysed band")
    need(min(v["gapsMm"]) > 0 and v["electrodeRadiusMm"] > 0 and v["electrodeLengthMm"] > 0, "Invalid electrode size or gap")
    need(v["gridSpacingMm"] > max(v["electrodeLengthMm"], 2 * v["electrodeRadiusMm"]), "Electrodes of the grid would overlap")
    lower, upper = phieBoxMm(config)
    need(np.all(np.abs(lower[:2]) < size[:2] / 2) and upper[2] < size[2] / 2 + f["bath"]["thicknessMm"],
         "The phi_e output box must lie inside the bath above the tissue")
    calibration = a["calibration"]
    need(np.any(np.isclose(centreYmm(config), calibration["referenceCentreYmm"])), "Calibration centre must be a catheter placement")
    need(all(np.any(np.isclose(centreYmm(config), c)) for c in groundTruthCentresYmm(config)), "Ground-truth centres must be catheter placements")
    need(f["groundTruth"]["leadFieldMethod"] in ("openCARP", "python"), "leadFieldMethod must be 'openCARP' or 'python'")
    need(restartTimeMs(config) is not None or not f["numerics"]["prepacing"], "No room for prepacing before the analysis frames")
    from .fibres import validateSpec
    validateSpec(v["fibres"], size[2])
    return config


# ----------------------------------------------------------------- fingerprints
STAGE_INPUTS = {
    # Configuration each stage reads; a stage's fingerprint also includes its parents'.
    "tune": ["values.cvTargetMPerS", "values.ionicModel", "fixed.tissue", "fixed.tune", "parNoBeta"],
    "conditioning": ["values.ionicModel", "fixed.conditioning", "parConditioning"],
    "simulation": ["values.wallMm", "values.fibres", "values.epicardialMedium", "values.scar", "values.waveAngleDeg",
                   "fixed.tissue", "fixed.bath", "fixed.stimulus", "fixed.numerics", "fixed.analysis.window",
                   "fixed.analysis.filterHistoryMs", "phieBox", "parNoBeta"],
    "physiology": ["fixed.acceptance.physiology"],
    "window": ["fixed.analysis.window", "values.gridSpacingMm"],
    "electrograms": ["values.gapsMm", "values.electrodeRadiusMm", "values.electrodeLengthMm", "values.gridSpacingMm",
                     "fixed.electrodes", "fixed.analysis.filterHistoryMs"],
    "measurements": ["values.noiseSdMv", "fixed.analysis", "fixed.acceptance.profile"],
    "export": ["values.region", "fixed.export"],
    "changePoints": [],
    "groundTruth": ["fixed.groundTruth", "fixed.acceptance.groundTruth"],
}


def section(config, dotted):
    if dotted == "phieBox":
        return [list(map(float, c)) for c in phieBoxMm(config)]
    if dotted == "parNoBeta":                # beta is the tune stage's output, so it is a parent, not an input
        return {k: v for k, v in config["par"].items() if k != "imp_region[0].cellSurfVolRatio"}
    if dotted == "parConditioning":
        keys = ["imp_region[0].im", "dt", "stim[0].ptcl.bcl", "stim[0].pulse.strength", "stim[0].ptcl.duration"]
        return {k: config["par"][k] for k in keys}
    value = config
    for key in dotted.split("."):
        value = value[key]
    return value


def fingerprint(config, stage, parents=()):
    """Hash of the configuration a stage uses plus its parents' fingerprints (documentation keys excluded)."""
    def strip(value):
        if isinstance(value, dict):
            return {k: strip(v) for k, v in sorted(value.items()) if not str(k).startswith("_")}
        return value
    payload = dict(inputs={name: strip(section(config, name)) for name in STAGE_INPUTS.get(stage, [])}, parents=list(parents))
    return hashlib.sha256(json.dumps(payload, sort_keys=True, default=str).encode()).hexdigest()[:16]


# ----------------------------------------------------------------- tuning record
def tuningKey(config):
    """What the tuned beta depends on: mesh spacing, CV target, ionic model, conductivities, tuning method."""
    return fingerprint(config, "tune")


def tunedBetaFor(config, record=TUNING_RECORD):
    """beta recorded for this configuration's tuning key, or None."""
    if not Path(record).exists():
        return None
    entry = loadJson(record).get(tuningKey(config))
    return None if entry is None else float(entry["cellSurfVolRatioPerUm"])


def recordTuning(config, result, record=TUNING_RECORD):
    entries = loadJson(record) if Path(record).exists() else {}
    entries[tuningKey(config)] = result
    saveJson(record, entries)
