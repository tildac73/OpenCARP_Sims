"""Step 7 verification: does openCARP build the same conductivity tensors from a rotated fibre file?

Two small runs (mpi_equivalence geometry: 20 x 30 x 4 mm, 0.5 mm, 3 beats) with the tuned
conductivities (unequal Roberts and Scher anisotropy, so the monodomain tensor differs from the
intracellular one) and fibres:
  uniform  : 30 deg everywhere
  bilayer  : 0 deg above 2 mm depth, 90 deg below (abrupt)
For each, phi_e recovered by openCARP (phie_recovery.igb) is compared with electrode_synthesis from
the same Vm (level-2 near-field quadrature, DL-2026-10-28-06; sigma_b = 1.0 S/m, openCARP's default
g_bath). Gate (as Step 2, PROPOSED): correlation >= 0.999 at every recovery point.
Control: the same comparison with the fibres wrongly taken as x-aligned must be clearly worse; this
shows the check is sensitive to the fibre field.
"""
import copy
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

import common
import boundary_setup
import electrode_synthesis as synthesis
import fibres as fibreTools
import ground_truth
import igb_io

RUNNER = common.REPOSITORY / "3D_Code" / "run_boundary.py"
CASES = {
    "uniform30": {"model": "uniform", "angleDeg": 30.0},
    "bilayer0_90": {"model": "bilayer", "endocardialAngleDeg": 0.0, "epicardialAngleDeg": 90.0,
                    "transitionDepthMm": 2.0, "transitionWidthMm": 0.0},
}
GATE_MIN_CORRELATION = 0.999


def caseConfig(tuned, fibreSpec):
    config = json.loads((common.REPOSITORY / "3D_Code/configs/mpi_equivalence.json").read_text())
    config["fibres"] = fibreSpec
    config["tissue"] = {"conductivitiesSPerM": dict(tuned["conductivitiesSPerM"]), "source": "readiness CV stage"}
    config["pacing"]["strength"] = float(tuned["stimulusStrength"])
    config["_purpose"] = "Readiness fibre check (openCARP tensor convention); not a scientific result."
    boundary_setup.validateConfig(config)
    return config


def compare(runFolder, condition, config, assumeXFibres=False):
    geometry = ground_truth.readElementGeometry(runFolder, condition)
    if assumeXFibres:
        geometry.fibreFrames = None                                   # deliberately wrong: control
    points = np.loadtxt(Path(runFolder) / "electrodes" / "recoveryPoints.pts", skiprows=1, ndmin=2) / 1000.0
    times = igb_io.timeAxis(Path(runFolder) / condition / "vm.igb")
    beatStart = (config["pacing"]["beatCount"] - 1) * config["pacing"]["cycleLengthMs"]
    frames = slice(int(np.searchsorted(times, beatStart)), int(np.searchsorted(times, beatStart + 100.0)))
    vm = igb_io.read(Path(runFolder) / condition / "vm.igb", frames).T.astype(float)
    recovered = igb_io.read(Path(runFolder) / condition / "phie_recovery.igb", frames).T.astype(float)
    spacing = config["meshSpacingMm"]
    synthesised = synthesis.pointOperator(geometry, points, sigmaBathSPerM=1.0, nearRadiusMm=3 * spacing,
                                          subdivisionLevel=2) @ vm
    a = synthesised - synthesised.mean(axis=1, keepdims=True)
    b = recovered - recovered.mean(axis=1, keepdims=True)
    correlation = np.sum(a * b, axis=1) / np.sqrt(np.sum(a * a, axis=1) * np.sum(b * b, axis=1))
    ratio = np.ptp(synthesised, axis=1) / np.ptp(recovered, axis=1)
    relative = np.linalg.norm(synthesised - recovered, axis=1) / np.linalg.norm(recovered, axis=1)
    return dict(points=int(len(points)), correlationMin=float(correlation.min()),
                correlationMedian=float(np.median(correlation)), amplitudeRatioMin=float(ratio.min()),
                amplitudeRatioMax=float(ratio.max()), relativeRmsMax=float(relative.max()),
                relativeRmsMedian=float(np.median(relative)))


def runStage(stageFolder, tuned, processCount):
    stageFolder = Path(stageFolder)
    stageFolder.mkdir(parents=True, exist_ok=True)
    results = {}
    for name, spec in CASES.items():
        folder = stageFolder / name
        config = caseConfig(tuned, spec)
        configPath = stageFolder / f"{name}.config.json"
        if not configPath.exists():
            boundary_setup.saveJson(configPath, config)
        if not (folder / "lesion" / "meta.json").exists():
            folder.mkdir(parents=True, exist_ok=True)
            code = common.runLogged([sys.executable, RUNNER, "--config", configPath, "--output", folder,
                                     "--simulate-only", "--np", processCount], stageFolder / f"{name}.simulate.log", check=False)
            if code != 0:
                results[name] = dict(error=f"simulation failed (exit {code}); see {name}.simulate.log")
                continue
        correct = compare(folder, "intact", config)
        control = compare(folder, "intact", config, assumeXFibres=True)
        results[name] = dict(fibres=spec, withFibres=correct, controlAssumingXFibres=control,
                             gatePassed=bool(correct["correlationMin"] >= GATE_MIN_CORRELATION),
                             controlClearlyWorse=bool(control["relativeRmsMedian"] > 2 * correct["relativeRmsMedian"]))
    summary = dict(cases=results, gateMinCorrelationProposed=GATE_MIN_CORRELATION,
                   passed=bool(results and all(r.get("gatePassed") for r in results.values())))
    common.saveJson(stageFolder / "fibreCheck.json", summary)
    return summary


def smoke(stageFolder, provisionalTuned, processCount):
    """Prepare + condition the bilayer case, then 1 ms of openCARP: does this build read the fibre file?"""
    stageFolder = Path(stageFolder); stageFolder.mkdir(parents=True, exist_ok=True)
    config = caseConfig(provisionalTuned, CASES["bilayer0_90"])
    configPath = stageFolder / "smoke_fibres.config.json"
    boundary_setup.saveJson(configPath, config)
    folder = stageFolder / "smoke_fibres"
    code = common.runLogged([sys.executable, RUNNER, "--config", configPath, "--output", folder, "--condition-only",
                             "--np", processCount], stageFolder / "smoke_fibres.prepare.log", check=False)
    if code != 0:
        raise RuntimeError(f"Fibre smoke preparation failed (exit {code})")
    nodes = len(boundary_setup.readNodePositions(folder / "mesh" / "intact"))
    prefix, _ = common.simulatorPrefix(processCount, nodes)
    smokePar = common.smokeCopy(folder / "intact.par", stageFolder / "smoke_fibres_intact")
    common.runLogged(prefix + ["+F", smokePar], stageFolder / "smoke_fibres_intact.log")
    common.checkSingleLaunch(stageFolder / "smoke_fibres_intact.log")
