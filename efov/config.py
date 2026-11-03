"""Load, check and derive the model configuration (config/final_model.json).

The configuration is the only place scientific values are set. This module:
  * validates it before any expensive step (loadConfig),
  * derives the few quantities that follow from it (electrode placements, prepacing restart time,
    expected mesh size),
  * fingerprints the parts of it each pipeline stage uses, so stale outputs are never reused,
  * imports the rule-chosen values from the workstation final settings run (importProductionModel).
"""
import copy
import hashlib
import json
from pathlib import Path

import numpy as np

from .io import loadJson, saveJson

REPOSITORY = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = REPOSITORY / "config" / "final_model.json"
CONDUCTIVITY_NAMES = ("g_il", "g_it", "g_in", "g_el", "g_et", "g_en")


def loadConfig(path=DEFAULT_CONFIG):
    config = loadJson(path)
    validateConfig(config)
    return config


def centreYmm(config):
    """Catheter centre positions across the boundary (mm)."""
    spec = config["electrodes"]["centreYmm"]
    if isinstance(spec, list):
        return np.asarray(spec, float)
    count = int(round((spec["toMm"] - spec["fromMm"]) / spec["stepMm"])) + 1
    return np.round(spec["fromMm"] + np.arange(count) * spec["stepMm"], 9)


def slabSizeMm(config):
    return np.array([*config["tissue"]["sizeMm"], config["tissue"]["wallMm"]], float)


def restartTimeMs(config):
    """Prepacing checkpoint: early enough that phi_e exists for every frame the analysis reads.

    The analysis reads the last beat from (window start - 2 x filter history); the earliest possible
    window start is stimulus end + minimumAfterStimulus. A margin is subtracted and the result floored
    to 10 ms. None if prepacing is off or there is no room.
    """
    if not config["numerics"]["prepacing"]:
        return None
    pace, analysis = config["pacing"], config["analysis"]
    earliestWindow = ((pace["beatCount"] - 1) * pace["cycleLengthMs"] + pace["stimulusDurationMs"]
                      + analysis["window"]["minimumAfterStimulusMs"])
    restart = earliestWindow - 2 * analysis["filterHistoryMs"] - config["numerics"]["prepacingMarginMs"]
    restart = float(np.floor(restart / 10.0) * 10.0)
    return restart if restart > 0 else None


def meshNodeCount(config):
    """Nodes of the tissue + bath mesh (regular block), for planning."""
    size, spacing = slabSizeMm(config), config["tissue"]["spacingMm"]
    bath = config["bath"]["thicknessMm"]
    padded = size + (2 * bath if config["bath"]["allSides"] else np.array([0, 0, bath]))
    return int(np.prod(np.round(padded / spacing).astype(int) + 1))


def validateConfig(config):
    """Reject inconsistent inputs before any simulation. Raises ValueError with the reason."""
    def need(condition, message):
        if not condition:
            raise ValueError(message)

    tissue, lesion, bath, pace = config["tissue"], config["lesion"], config["bath"], config["pacing"]
    numerics, electrodes, analysis = config["numerics"], config["electrodes"], config["analysis"]
    size, spacing = slabSizeMm(config), tissue["spacingMm"]
    need(np.all(size > 0) and spacing > 0, "Slab size and spacing must be positive")
    need(np.allclose(size / spacing, np.round(size / spacing)), "Every slab dimension must be a multiple of spacingMm")
    need(size[2] / spacing >= 2, "Use at least two elements through the wall")
    need(set(tissue["conductivitiesSPerM"]) == set(CONDUCTIVITY_NAMES), f"conductivitiesSPerM needs {CONDUCTIVITY_NAMES}")
    need(all(v > 0 for v in tissue["conductivitiesSPerM"].values()), "Conductivities must be positive")
    g = tissue["conductivitiesSPerM"]
    need(g["g_it"] == g["g_in"] and g["g_et"] == g["g_en"],
         "Fibres are given as one vector per element, so tissue must be transversely isotropic (t = n)")
    need(tissue["cellSurfVolRatioPerUm"] > 0, "cellSurfVolRatioPerUm must be positive")
    need(tissue["ionicModel"] == "Courtemanche", "Only the Courtemanche model is wired in (conditioning, checks)")
    need(lesion["scar"] == "insulating", "Only the insulating scar (R10) is implemented in the final model")
    need(-size[1] / 2 < lesion["boundaryYmm"] < size[1] / 2, "Lesion boundary must lie inside the slab")
    need(bath["thicknessMm"] > 0 and bath["bloodSPerM"] > 0, "Bath thickness and conductivity must be positive")
    need(np.isclose(bath["thicknessMm"] / spacing, round(bath["thicknessMm"] / spacing)), "Bath thickness must be a multiple of spacingMm")
    need(bath["allSides"] and 0 < bath["groundCubeMm"] < bath["thicknessMm"],
         "Final model: bath on all sides with a ground cube smaller than the bath thickness")
    need(pace["beatCount"] >= 3, "Physiology checks compare the last three beats: beatCount >= 3")
    need(0 < pace["stimulusDurationMs"] < pace["cycleLengthMs"] and pace["stimulusStrengthUAPerCm2"] > 0, "Invalid pacing")
    need(0 < pace["stripWidthMm"] < size[0] / 2, "Stimulus strip must be narrower than half the slab")
    need(isinstance(pace["conditioningBeats"], int) and pace["conditioningBeats"] >= 3, "conditioningBeats must be an integer >= 3")
    step, dt = numerics["sampleStepMs"], numerics["timeStepUs"]
    need(step > 0 and dt > 0 and np.isclose(step * 1000 / dt, round(step * 1000 / dt)) and step * 1000 >= dt,
         "sampleStepMs must be a positive integer multiple of timeStepUs")
    need(numerics["forwardModel"] == "pseudoBidomain", "The final model uses openCARP pseudo-bidomain (DL-2026-11-02-01)")
    band = analysis["differentialBandHz"]
    need(0 < band[0] < band[1] < 500 / step, "Filter band must lie below Nyquist")
    need(analysis["filterHistoryMs"] > 0, "Keep positive filter history")
    offsets = np.asarray(electrodes["gridOffsetsMm"])
    need(len(offsets) == 4 and np.allclose(np.diff(offsets), np.diff(offsets)[0]) and np.diff(offsets)[0] > 0,
         "The 4 x 4 recording code needs four equally spaced increasing offsets")
    centres = centreYmm(config)
    need(len(np.unique(centres)) == len(centres), "Catheter placements must be unique")
    need(electrodes["radiusMm"] > 0 and electrodes["lengthMm"] > 0 and min(electrodes["gapsMm"]) > 0, "Invalid electrode size or gap")
    need(electrodes["angularSamples"] >= 4 and electrodes["axialSamples"] >= 1, "Use >= 4 angular and >= 1 axial samples")
    reach = max(electrodes["gapsMm"]) + 2 * electrodes["radiusMm"]
    need(numerics["phieOutputHeightMm"] >= reach + spacing,
         f"phieOutputHeightMm must be >= largest gap + electrode diameter + one element ({reach + spacing} mm)")
    need(numerics["phieOutputHeightMm"] < bath["thicknessMm"], "phi_e output layer must lie inside the bath")
    halfX = abs(electrodes["centreXmm"]) + offsets.max() + electrodes["lengthMm"] / 2
    halfY = np.abs(centres).max() + offsets.max() + electrodes["radiusMm"]
    need(halfX < size[0] / 2 and halfY < size[1] / 2, "Every electrode must lie over the tissue")
    calibration = analysis["calibration"]
    need(np.any(np.isclose(centres, calibration["referenceCentreYmm"])) and calibration["referenceGapMm"] in electrodes["gapsMm"],
         "Calibration reference must be a simulated placement")
    need(all(np.any(np.isclose(centres, c)) for c in config["groundTruth"]["centresYmm"]),
         "Ground-truth centres must be catheter placements")
    need(np.allclose(size / config["groundTruth"]["voxelMm"], np.round(size / config["groundTruth"]["voxelMm"])),
         "Slab size must be a multiple of groundTruth.voxelMm")
    from .recordings import availableLabels
    for kind, labels in analysis["recordings"].items():
        need(kind in ("uni", "bi", "omni") and labels and set(labels) <= availableLabels(kind), f"Invalid {kind} recording labels")
    window = analysis["window"]
    need(window["leadMs"] >= 0 and window["trailMs"] > 0 and window["minimumAfterStimulusMs"] >= 0, "Invalid window rule")
    response = config["acceptance"]["physiology"]["responseWindowMs"]
    need(0 <= response[0] < response[1] <= pace["cycleLengthMs"], "Physiology response window must fit in one cycle")
    restart = restartTimeMs(config)
    need(not numerics["prepacing"] or restart is not None, "No room for prepacing before the analysis frames")
    return config


# ----------------------------------------------------------------- fingerprints
STAGE_INPUTS = {
    # Config sections each stage reads; a stage's fingerprint also includes its parents'.
    "conditioning": ["tissue.ionicModel", "pacing", "numerics.timeStepUs"],
    "simulation": ["tissue", "lesion", "bath", "pacing", "numerics", "analysis.window", "analysis.filterHistoryMs"],
    "physiology": ["acceptance.physiology"],
    "window": ["analysis.window", "electrodes"],
    "electrograms": ["electrodes", "analysis.filterHistoryMs", "acceptance.implementation"],
    "measurements": ["analysis", "acceptance.profile"],
    "changePoints": ["changePoints"],
    "groundTruth": ["groundTruth", "acceptance.implementation"],
}


def section(config, dotted):
    value = config
    for key in dotted.split("."):
        value = value[key]
    return value


def fingerprint(config, stage, parents=()):
    """Hash of the config sections a stage uses plus its parents' fingerprints (documentation keys excluded)."""
    def strip(value):
        if isinstance(value, dict):
            return {k: strip(v) for k, v in sorted(value.items()) if not k.startswith("_")}
        return value
    payload = dict(inputs={name: strip(section(config, name)) for name in STAGE_INPUTS.get(stage, [])},
                   parents=list(parents))
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()[:16]


# ----------------------------------------------------------------- workstation values
def importProductionModel(productionModelPath, configPath=DEFAULT_CONFIG, outputPath=None):
    """Copy the rule-chosen values of the final settings run (productionModel.json) into a config.

    Returns the new config (and writes it to outputPath if given). Changed keys are marked
    status "confirmed" in provenance, with the file they came from.
    """
    model = loadJson(productionModelPath)
    config = copy.deepcopy(loadJson(configPath))
    source = f"workstation final settings run: {Path(productionModelPath).name} ({model.get('createdUtc', 'no date')})"
    updates = {
        "tissue.sizeMm": list(model["domainMm"][:2]),
        "tissue.wallMm": model["domainMm"][2],
        "tissue.spacingMm": model["spacingMm"],
        "tissue.cellSurfVolRatioPerUm": model["cellSurfVolRatioPerUm"],
        "tissue.conductivitiesSPerM": model["conductivitiesSPerM"],
        "pacing.beatCount": model["beatCount"],
        "pacing.stimulusStrengthUAPerCm2": model["stimulusStrength"],
        "numerics.timeStepUs": model["timeStepUs"],
        "numerics.sampleStepMs": model["sampleStepMs"],
    }
    for dotted, value in updates.items():
        *parents, last = dotted.split(".")
        target = config
        for key in parents:
            target = target[key]
        target[last] = value
        config["provenance"][dotted] = dict(status="confirmed", source=source)
    validateConfig(config)
    if outputPath is not None:
        saveJson(outputPath, config)
    return config


def unconfirmed(config):
    """Provenance entries that still await a value from the workstation, or are open."""
    return {k: v for k, v in config.get("provenance", {}).items()
            if not k.startswith("_") and v.get("status") in ("awaitingWorkstation", "open")}
