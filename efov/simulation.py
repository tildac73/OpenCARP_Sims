"""Run the model in openCARP: single-cell conditioning, then one tissue run per condition.

    conditionCell(config, folder)              bench limit cycle -> Courtemanche initial state
    runSpec(config, condition, state, folder)  every value slab_experiment.py needs, from the config
    runCondition(config, condition, ...)       carputils slab_experiment.py; refuses stale or partial runs
    tissueModel(jobFolder)                     tissue nodes (vm.igb order) and their positions

openCARP writes vm.igb and LATs for tissue nodes only, in ascending global node order; phie.igb
covers the listed output nodes (phieNodes.npy), also in ascending order (checked 1 Nov 2026).
"""
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

import numpy as np

from .config import restartTimeMs, slabSizeMm
from .io import igbHeader, loadJson, readMesh, saveJson

EXPERIMENT = Path(__file__).resolve().parent / "opencarp" / "slab_experiment.py"
CONDITIONS = ("intact", "lesion")
BATH_TAG, LESION_TAG = 0, 5


def findExecutable(name, environmentVariable):
    path = os.environ.get(environmentVariable) or shutil.which(name)
    if not path or not (shutil.which(path) or os.access(path, os.X_OK)):
        raise RuntimeError(f"{name} not found: activate the openCARP environment or set {environmentVariable}")
    return str(path)


def runLogged(command, logPath, env=None):
    """Run a command, keep its full output in logPath, and remove any credentials openCARP prints."""
    import re
    with open(logPath, "w") as log:
        log.write("Command: " + " ".join(map(str, command)) + "\n")
        log.flush()
        code = subprocess.call([str(c) for c in command], stdout=log, stderr=subprocess.STDOUT, env=env)
    text = Path(logPath).read_text(errors="replace")
    cleaned = re.sub(r"(://)[^/\s:@]+:[^/\s@]+@", r"\1<credentials removed>@", text)
    if cleaned != text:
        Path(logPath).write_text(cleaned)
    return code


# ----------------------------------------------------------------- carputils environment
def carputilsEnvironment(outputFolder):
    """Use CARPUTILS_SETTINGS if set; otherwise write a run-local settings file (MPIEXEC = $MPIEXEC_BIN)."""
    env = dict(os.environ)
    if env.get("CARPUTILS_SETTINGS"):
        return env
    path = Path(outputFolder).resolve() / "carputils_settings.yaml"
    if not path.exists():
        carp = findExecutable("openCARP", "OPENCARP_BIN")
        mpiexec = os.environ.get("MPIEXEC_BIN", "mpiexec")
        code = ("from carputils.settings.config import new; "
                f"new({str(path)!r}, software_root={str(Path(carp).resolve().parent)!r}, mpiexec={mpiexec!r})")
        path.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run([sys.executable, "-c", code], check=True, capture_output=True)
    env["CARPUTILS_SETTINGS"] = str(path)
    return env


# ----------------------------------------------------------------- conditioning
def conditionCell(config, folder):
    """Pace one Courtemanche cell conditioningBeats times (bench) and save its state.

    The openCARP limit-cycle workflow. The tissue run starts from this state; tissue stability is
    checked separately on the last three beats (physiology stage).
    """
    folder = Path(folder)
    state, record = folder / "Courtemanche.sv", folder / "conditioning.json"
    if record.exists() and state.is_file() and state.stat().st_size:
        return state
    folder.mkdir(parents=True, exist_ok=True)
    pace = config["pacing"]
    duration = pace["conditioningBeats"] * pace["cycleLengthMs"]
    command = [findExecutable("bench", "BENCH_BIN"), "--imp", config["tissue"]["ionicModel"],
               "--dt", config["numerics"]["timeStepUs"] / 1000, "--duration", duration,
               "--numstim", pace["conditioningBeats"], "--bcl", pace["cycleLengthMs"], "--stim-start", 0,
               "--stim-curr", pace["stimulusStrengthUAPerCm2"], "--stim-dur", pace["stimulusDurationMs"],
               "--stim-assign", "on", "--no-trace", "--save-ini-file", state, "--save-ini-time", duration]
    if runLogged(command, folder / "bench.log") != 0 or not state.is_file() or "Courtemanche" not in state.read_text():
        raise RuntimeError(f"bench did not write a Courtemanche state; see {folder / 'bench.log'}")
    saveJson(record, dict(beats=pace["conditioningBeats"], cycleLengthMs=pace["cycleLengthMs"], savedAtMs=duration,
                          command=[str(c) for c in command]))
    return state


# ----------------------------------------------------------------- tissue runs
def runSpec(config, condition, initialState, meshRoot, durationMs=None):
    """Everything slab_experiment.py needs for one condition (a flat dict of values with units)."""
    t, b, p, n = config["tissue"], config["bath"], config["pacing"], config["numerics"]
    spec = dict(condition=condition, tissueSizeMm=slabSizeMm(config).tolist(), spacingMm=t["spacingMm"],
                conductivitiesSPerM=t["conductivitiesSPerM"], cellSurfVolRatioPerUm=t["cellSurfVolRatioPerUm"],
                boundaryYmm=config["lesion"]["boundaryYmm"], lesionConductivitySPerM=config["lesion"]["conductivitySPerM"],
                bathThicknessMm=b["thicknessMm"], groundCubeMm=b["groundCubeMm"], bloodSPerM=b["bloodSPerM"],
                cycleLengthMs=p["cycleLengthMs"], beatCount=p["beatCount"], stimulusDurationMs=p["stimulusDurationMs"],
                stimulusStrengthUAPerCm2=p["stimulusStrengthUAPerCm2"], stripWidthMm=p["stripWidthMm"],
                timeStepUs=n["timeStepUs"], sampleStepMs=n["sampleStepMs"], phieOutputHeightMm=n["phieOutputHeightMm"],
                restartMs=restartTimeMs(config), initialState=str(initialState), meshRoot=str(meshRoot))
    if durationMs is not None:                                      # short verification runs only
        spec["durationMs"], spec["restartMs"] = float(durationMs), None
    return spec


def expectedFrames(spec, fromMs=0.0):
    end = spec.get("durationMs") or spec["cycleLengthMs"] * spec["beatCount"]
    return int(round((end - fromMs) / spec["sampleStepMs"])) + 1


def runComplete(job, spec):
    job = Path(job)
    phieStart = spec.get("restartMs") or 0.0
    for name, start in [("vm.igb", 0.0), ("phie.igb", phieStart)]:
        if not (job / name).exists() or igbHeader(job / name)[2] != expectedFrames(spec, start):
            return False
    return (job / "act-thresh.dat").exists() and (not spec.get("restartMs") or (job / "restart.json").exists())


def runCondition(config, condition, initialState, outputFolder, processCount=1, flavor=None, durationMs=None, dry=False):
    """One openCARP tissue run in <outputFolder>/simulations/<condition>; reused only if complete and identical."""
    outputFolder = Path(outputFolder).resolve()
    job = outputFolder / "simulations" / condition
    specPath = job.parent / f"{condition}.json"
    spec = runSpec(config, condition, initialState, outputFolder / "meshes", durationMs)
    if job.exists() and runComplete(job, spec):
        if loadJson(specPath) != json.loads(json.dumps(spec)):
            raise ValueError(f"{job} was run with a different spec; use a new output folder")
        return loadJson(job.parent / f"{condition}.timing.json")
    if job.exists() and not dry:
        raise FileExistsError(f"{job} is incomplete (interrupted run?). Move it aside, then rerun.")
    saveJson(specPath, spec)
    flavor = flavor or config["numerics"]["solverFlavor"]
    command = [sys.executable, str(EXPERIMENT), "--spec", str(specPath), "--np", str(processCount), "--ID", str(job),
               "--overwrite-behaviour", "error", "--flavor", flavor] + (["--dry"] if dry else ["--silent"])
    log = job.parent / f"{condition}.log"
    started = time.time()
    code = runLogged(command, log, env=carputilsEnvironment(outputFolder))
    if dry:
        return dict(dryRun=log.read_text(errors="replace"))
    if code != 0 or not runComplete(job, spec):
        raise RuntimeError(f"openCARP run '{condition}' failed (exit {code}); see {log}")
    launches = log.read_text(errors="replace").count("*** Launching simulation ***")
    expected = 2 if spec.get("restartMs") else 1
    if launches != expected:
        raise RuntimeError(f"{condition}: {launches} openCARP launch messages, expected {expected} (MPI launcher mismatch?)")
    timing = dict(wallSeconds=round(time.time() - started, 1), processCount=processCount, solverFlavor=flavor)
    saveJson(job.parent / f"{condition}.timing.json", timing)
    return timing


def solverFlavorWorks(config, outputFolder, flavor, processCount=1):
    """5 ms pseudo-bidomain run of a tiny slab: does this openCARP build support the solver flavour?"""
    folder = Path(outputFolder).resolve() / "solverCheck" / flavor
    if folder.exists():
        shutil.rmtree(folder)
    small = json.loads(json.dumps(config))
    small["tissue"].update(sizeMm=[4.0, 4.0], wallMm=1.0, spacingMm=0.5)
    small["bath"].update(thicknessMm=2.0, groundCubeMm=1.0)
    small["numerics"]["phieOutputHeightMm"] = 1.5
    small["pacing"]["conditioningBeats"] = 3
    state = conditionCell(small, folder / "conditioning")
    try:
        runCondition(small, "intact", state, folder, min(processCount, 2), flavor=flavor, durationMs=5.0)
        return True
    except (RuntimeError, ValueError, OSError):
        return False


# ----------------------------------------------------------------- reading the result
def runRecord(job):
    return loadJson(Path(job) / "runRecord.json")


def tissueModel(job):
    """Tissue nodes of the openCARP mesh in vm.igb order (ascending), their positions (mm) and the mesh."""
    meshBase = runRecord(job)["meshBase"]
    positions, elements, tags, fibres = readMesh(meshBase)
    tissueNodes = np.unique(elements[tags != BATH_TAG])
    if igbHeader(Path(job) / "vm.igb")[1] != len(tissueNodes):
        raise ValueError(f"{job}: vm.igb node count differs from the mesh's tissue nodes")
    lesionElements = tags == LESION_TAG
    return dict(meshBase=meshBase, positions=positions, elements=elements, tags=tags, fibres=fibres,
                tissueNodes=tissueNodes, tissuePositions=positions[tissueNodes], lesionElements=lesionElements)
