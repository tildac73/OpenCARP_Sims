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
every change point between successive noise levels of the doubling series.
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
    summary.to_csv(Path(out) / "noiseDoubling.csv", index=False)
    return summary


def runSweep(sweepPath, outputFolder, processCount=1, only=None):
    sweepPath, out = Path(sweepPath).resolve(), Path(outputFolder).resolve()
    sweep = loadJson(sweepPath)
    modelConfigPath = (sweepPath.parent / sweep["modelConfig"]).resolve()
    plan = variants(sweep, modelConfigPath)
    statusPath = out / "sweepStatus.json"
    status = loadJson(statusPath) if statusPath.exists() else {}
    cache = out / "simulationCache"
    baseline = Run(configFor(modelConfigPath, {}, False), out / "variants" / "baseline", processCount, cache)
    baseline.all()
    status["baseline"] = dict(status="complete", overrides={})
    saveJson(statusPath, status)
    for name, overrides in plan.items():
        if name == "baseline" or (only and name not in only) or status.get(name, {}).get("status") == "complete":
            continue
        try:
            ownSimulation = sharedFolder(configFor(modelConfigPath, overrides, False), out, cache) != baseline.shared
            Run(configFor(modelConfigPath, overrides, ownSimulation), out / "variants" / name, processCount, cache).all()
            status[name] = dict(status="complete", overrides=overrides)
        except Exception as error:                                     # recorded, never silently pooled
            status[name] = dict(status="failed", overrides=overrides, error=f"{type(error).__name__}: {error}",
                                traceback=traceback.format_exc())
        saveJson(statusPath, status)
    noiseDoublingTable(out, modelConfigPath)
    return status
