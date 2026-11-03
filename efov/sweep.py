"""Parameter sweeps (FM-17): the baseline plus one-at-a-time and named variants, with a shared simulation cache.

Sweep file (config/sweeps/<name>.json):
  {"modelConfig": "../model.json",
   "oneAtATime": ["wallMm", "waveAngleDeg", ...],          each planned value of each variable, alone
   "named": {"roof": {"region": "roof", "wallMm": 1.6, "fibres": {...}}}}
Every variant is model.json's baseline with its overrides. The baseline runs first; its calibration
gain is then fixed for every other variant (FM-19), so amplitude differences are predictions.
Simulations are cached under <out>/simulationCache/<simulation fingerprint>/: a variant that changes
only electrodes, noise or analysis reuses the baseline's openCARP runs, physiology and tuning.
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


def configFor(modelConfigPath, overrides, gain, compact):
    config = loadConfig(modelConfigPath, overrides)
    config["fixed"]["analysis"]["calibration"]["fixedGain"] = gain
    config["fixed"]["compactAfterReport"] = compact
    return config


def runSweep(sweepPath, outputFolder, processCount=1, only=None):
    sweepPath, out = Path(sweepPath).resolve(), Path(outputFolder).resolve()
    sweep = loadJson(sweepPath)
    modelConfigPath = (sweepPath.parent / sweep["modelConfig"]).resolve()
    plan = variants(sweep, modelConfigPath)
    statusPath = out / "sweepStatus.json"
    status = loadJson(statusPath) if statusPath.exists() else {}
    cache = out / "simulationCache"
    baseline = Run(configFor(modelConfigPath, {}, None, False), out / "variants" / "baseline", processCount, cache)
    baseline.all()
    gain = baseline.result("measurements")["calibration"]["gain"]
    status["baseline"] = dict(status="complete", overrides={}, gain=gain)
    saveJson(statusPath, status)
    for name, overrides in plan.items():
        if name == "baseline" or (only and name not in only) or status.get(name, {}).get("status") == "complete":
            continue
        try:
            ownSimulation = sharedFolder(configFor(modelConfigPath, overrides, gain, False), out, cache) != baseline.shared
            Run(configFor(modelConfigPath, overrides, gain, ownSimulation), out / "variants" / name, processCount, cache).all()
            status[name] = dict(status="complete", overrides=overrides)
        except Exception as error:                                     # recorded, never silently pooled
            status[name] = dict(status="failed", overrides=overrides, error=f"{type(error).__name__}: {error}",
                                traceback=traceback.format_exc())
        saveJson(statusPath, status)
    return status
