"""Run the model in openCARP: single-cell conditioning, then one tissue run per condition.

    conditionCell(config, folder)               bench limit cycle -> initial state
    runSpec(config, condition, state, ...)      what slab_experiment.py needs that is not in model.par
    runCondition(config, condition, ...)        carputils slab_experiment.py; refuses stale or partial runs
    tissueModel(job)                            tissue nodes (vm.igb order) and their positions
    compactRun(job, keepFrames)                 keep only the analysis frames of Vm (FM-20)

openCARP writes vm.igb and LATs for tissue nodes only, in ascending global node order; phie.igb
covers the listed output nodes (phieNodes.npy), also in ascending order (checked 1 Nov 2026).
"""
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import time

import numpy as np

from .config import beatCount, cycleLengthMs, par, phieBoxMm, restartTimeMs, slabSizeMm, spacingMm, stimulusDurationMs, timeStepUs
from .io import igbHeader, igbTimes, loadJson, readIgb, readMesh, saveJson

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
    with open(logPath, "w") as log:
        log.write("Command: " + " ".join(map(str, command)) + "\n")
        log.flush()
        code = subprocess.call([str(c) for c in command], stdout=log, stderr=subprocess.STDOUT, env=env)
    text = Path(logPath).read_text(errors="replace")
    cleaned = re.sub(r"(://)[^/\s:@]+:[^/\s@]+@", r"\1<credentials removed>@", text)
    if cleaned != text:
        Path(logPath).write_text(cleaned)
    return code


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
    """Pace one cell `conditioning.beats` times with bench at the model.par cycle length and save its state."""
    folder = Path(folder)
    model = par(config, "imp_region[0].im", str)
    state, record = folder / f"{model}.sv", folder / "conditioning.json"
    if record.exists() and state.is_file() and state.stat().st_size:
        return state
    folder.mkdir(parents=True, exist_ok=True)
    beats, cycle = config["fixed"]["conditioning"]["beats"], cycleLengthMs(config)
    duration = beats * cycle
    command = [findExecutable("bench", "BENCH_BIN"), "--imp", model, "--dt", timeStepUs(config) / 1000, "--duration", duration,
               "--numstim", beats, "--bcl", cycle, "--stim-start", 0, "--stim-curr", par(config, "stim[0].pulse.strength"),
               "--stim-dur", stimulusDurationMs(config), "--stim-assign", "on", "--no-trace",
               "--save-ini-file", state, "--save-ini-time", duration]
    if runLogged(command, folder / "bench.log") != 0 or not state.is_file() or model not in state.read_text():
        raise RuntimeError(f"bench did not write a {model} state; see {folder / 'bench.log'}")
    saveJson(record, dict(beats=beats, cycleLengthMs=cycle, savedAtMs=duration, command=[str(c) for c in command]))
    return state


# ----------------------------------------------------------------- tissue runs
def runSpec(config, condition, initialState, meshRoot, kind="slab", durationMs=None):
    """Everything slab_experiment.py needs that model.par does not hold (geometry, paths, overrides)."""
    v, f = config["values"], config["fixed"]
    size = slabSizeMm(config) if kind == "slab" else np.asarray(f["tune"]["stripSizeMm"], float)
    lower, upper = phieBoxMm(config)
    spec = dict(kind=kind, condition=condition, modelPar=config["modelPar"], parOverrides=config["parOverrides"],
                tissueSizeMm=size.tolist(), spacingMm=spacingMm(config).tolist(),
                bathThicknessMm=f["bath"]["thicknessMm"], groundCubeMm=f["bath"]["groundCubeMm"],
                epicardialMedium=v["epicardialMedium"], boundaryYmm=f["tissue"]["boundaryYmm"], fibres=v["fibres"],
                waveAngleDeg=v["waveAngleDeg"] if kind == "slab" else 0.0, stripWidthMm=f["stimulus"]["stripWidthMm"],
                phieBoxMm=[lower.tolist(), upper.tolist()], restartMs=restartTimeMs(config) if kind == "slab" else None,
                initialState=str(initialState), meshRoot=str(meshRoot))
    if kind == "strip":
        spec["durationMs"] = f["tune"]["beats"] * cycleLengthMs(config)
    if durationMs is not None:                                      # short verification runs only
        spec["durationMs"], spec["restartMs"] = float(durationMs), None
    return spec


def expectedFrames(config, spec, fromMs=0.0):
    end = spec.get("durationMs") or par(config, "tend")
    return int(round((end - fromMs) / par(config, "spacedt"))) + 1


def runComplete(config, job, spec):
    job = Path(job)
    needed = [("vm.igb", 0.0)] + ([("phie.igb", spec.get("restartMs") or 0.0)] if spec["kind"] == "slab" else [])
    for name, start in needed:
        if not (job / name).exists() or igbHeader(job / name)[2] != expectedFrames(config, spec, start):
            return False
    return (job / "act-thresh.dat").exists() and (not spec.get("restartMs") or (job / "restart.json").exists())


def runCondition(config, condition, initialState, simulationFolder, processCount=1, flavor=None, durationMs=None,
                 dry=False, kind="slab"):
    """One openCARP run in <simulationFolder>/<condition>; reused only if complete and made from the same spec."""
    simulationFolder = Path(simulationFolder).resolve()
    job = simulationFolder / condition
    specPath = simulationFolder / f"{condition}.json"
    spec = runSpec(config, condition, initialState, simulationFolder.parent / "meshes", kind, durationMs)
    if job.exists() and (job / "runRecord.json").exists() and runComplete(config, job, spec):
        if loadJson(specPath) != json.loads(json.dumps(spec)):
            raise ValueError(f"{job} was run with a different spec; use a new output folder")
        return loadJson(simulationFolder / f"{condition}.timing.json")
    if job.exists() and not dry:
        raise FileExistsError(f"{job} is incomplete (interrupted run?). Move it aside, then rerun.")
    saveJson(specPath, spec)
    flavor = flavor or config["fixed"]["numerics"]["solverFlavor"]
    command = [sys.executable, str(EXPERIMENT), "--spec", str(specPath), "--np", str(processCount), "--ID", str(job),
               "--overwrite-behaviour", "error", "--flavor", flavor] + (["--dry"] if dry else ["--silent"])
    log = simulationFolder / f"{condition}.log"
    started = time.time()
    code = runLogged(command, log, env=carputilsEnvironment(simulationFolder))
    if dry:
        return dict(dryRun=log.read_text(errors="replace"))
    if code != 0 or not runComplete(config, job, spec):
        raise RuntimeError(f"openCARP run '{condition}' failed (exit {code}); see {log}")
    launches = log.read_text(errors="replace").count("*** Launching simulation ***")
    expected = 2 if spec.get("restartMs") else 1
    if launches != expected:
        raise RuntimeError(f"{condition}: {launches} openCARP launch messages, expected {expected} (MPI launcher mismatch?)")
    timing = dict(wallSeconds=round(time.time() - started, 1), processCount=processCount, solverFlavor=flavor)
    saveJson(simulationFolder / f"{condition}.timing.json", timing)
    return timing


def solverFlavorWorks(config, outputFolder, flavor, processCount=1):
    """5 ms pseudo-bidomain run of a tiny slab: does this openCARP build support the solver flavour?"""
    from .config import loadConfig
    folder = Path(outputFolder).resolve() / "solverCheck" / flavor
    if folder.exists():
        shutil.rmtree(folder)
    small = loadConfig(config["source"], config["overrides"], config["parOverrides"])
    small["fixed"] = json.loads(json.dumps(small["fixed"]))
    small["fixed"]["tissue"]["sizeMm"] = [6.0, 6.0]
    small["fixed"]["electrodes"]["centreYmm"] = [0.0]
    small["fixed"]["conditioning"]["beats"] = 3
    small["fixed"]["bath"].update(thicknessMm=2.0)
    state = conditionCell(small, folder / "conditioning")
    try:
        runCondition(small, "intact", state, folder / "simulations", min(processCount, 2), flavor=flavor, durationMs=5.0)
        return True
    except (RuntimeError, ValueError, OSError):
        return False


# ----------------------------------------------------------------- reading the result
def runRecord(job):
    return loadJson(Path(job) / "runRecord.json")


def tissueModel(job):
    """Tissue nodes of the run mesh in vm.igb order (ascending), their positions (mm) and the mesh."""
    meshBase = runRecord(job)["meshBase"]
    positions, elements, tags, fibres = readMesh(meshBase)
    tissueNodes = np.unique(elements[tags != BATH_TAG])
    if igbHeader(Path(job) / "vm.igb")[1] != len(tissueNodes):
        raise ValueError(f"{job}: vm.igb node count differs from the mesh's tissue nodes")
    return dict(meshBase=meshBase, positions=positions, elements=elements, tags=tags, fibres=fibres,
                tissueNodes=tissueNodes, tissuePositions=positions[tissueNodes], lesionElements=tags == LESION_TAG)


def compactRun(job, startMs, endMs):
    """Keep Vm only for [startMs, endMs] (vm_analysis.npz); delete vm.igb and phie.igb (FM-20).

    Only for a simulation that no other variant reuses: its electrode signals are already extracted
    (electrograms/<condition>.npz) and the ground truth reads Vm from vm_analysis.npz. Recorded in
    compact.json; the raw output is regenerated by rerunning the resolved .par files.
    """
    job = Path(job)
    times = igbTimes(job / "vm.igb")
    frames = slice(int(np.searchsorted(times, startMs - 1e-9)), int(np.searchsorted(times, endMs + 1e-9)))
    np.savez_compressed(job / "vm_analysis.npz", vm=readIgb(job / "vm.igb", frames), timeMs=times[frames])
    removed = []
    for name in ("vm.igb", "phie.igb"):
        if (job / name).exists():
            (job / name).unlink(); removed.append(name)
    saveJson(job / "compact.json", dict(keptVmMs=[float(times[frames][0]), float(times[frames][-1])], removed=removed,
                                        note="Raw output deleted after the report; rerun the resolved .par files to regenerate it"))
