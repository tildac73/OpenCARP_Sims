"""Prepare and run the two matched openCARP slabs.

Read prepareExperiment first: mesh -> electrodes -> stimulus -> parameter files.
Then read simulateExperiment: run each condition -> average electrode samples.
All settings use mm and ms, except timeStepUs. Unit conversions happen when
writing openCARP inputs (mesh points in micrometres, mesher dimensions in cm).
"""

from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import numpy as np

projectFolder = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(projectFolder / "Shared_Pipeline"))
import igb_io
from boundary_geometry import makeElectrodeLayout, signedDistance


def prepareExperiment(runFolder, config):
    """Prepare matched mesh, electrode, stimulus and parameter files; save provenance."""
    validateConfig(config)
    runFolder = Path(runFolder).resolve()
    if (runFolder / "manifest.json").exists():
        checkSimulationSettings(runFolder, config)
        return runFolder
    findExecutable("mesher", "MESHER_BIN")
    runFolder.mkdir(parents=True, exist_ok=True)
    if any(runFolder.iterdir()):
        raise FileExistsError("An incomplete preparation exists. Choose a new run folder.")
    meshNames = buildMesh(runFolder, config)
    recoveryBase = writeElectrodes(runFolder, config)
    for condition, meshBase in meshNames.items():
        stimulusBase = runFolder / f"{condition}Stimulus"
        count = writeStimulus(meshBase, condition, config, stimulusBase)
        writeParameters(runFolder, condition, meshBase, stimulusBase, recoveryBase, config)
        print(f"{condition}: {count} stimulated viable nodes", flush=True)
    saveJson(runFolder / "config.json", config)
    saveJson(
        runFolder / "manifest.json",
        dict(
            model="Courtemanche, unmodified",
            forwardMethod="openCARP phie recovery; unbounded homogeneous medium",
            pilotOnly=True,
            spatialConvergenceVerified=False,
            domainConvergenceVerified=False,
            distanceConvention="positive inside lesion; virtual boundary retained in intact",
        ),
    )
    sourceFolder = runFolder / "sourceSnapshot"
    sourceFolder.mkdir()
    for folder in ["3D_Code", "Shared_Pipeline"]:
        destination = sourceFolder / folder
        destination.mkdir()
        for path in (projectFolder / folder).glob("*.py"):
            shutil.copy2(path, destination / path.name)
    shutil.copy2(projectFolder / "3D_Code/3d_pilot.par", sourceFolder / "3d_pilot.par")
    return runFolder


def simulateExperiment(runFolder, config, processCount=1):
    """Run both conditions and export the averaged finite-electrode signals."""
    checkSimulationSettings(runFolder, config)
    simulator = findExecutable("openCARP", "OPENCARP_BIN")
    commandPrefix = [simulator]
    if processCount > 1:
        commandPrefix = [findExecutable("mpiexec", "MPIEXEC_BIN"), "-n", processCount, simulator]
    if processCount < 1:
        raise ValueError("Process count must be positive")
    runFolder = Path(runFolder).resolve()
    for condition in ["intact", "lesion"]:
        simulationFolder = runFolder / condition
        completedPath = simulationFolder / "meta.json"
        if completedPath.exists():
            if not json.loads(completedPath.read_text()).get("complete"):
                raise ValueError(f"{condition} has incomplete completion metadata")
            for name in ["vm.igb", "phie_recovery.igb", "parameters.par", "egm.npz"]:
                if not (simulationFolder / name).exists():
                    raise FileNotFoundError(f"Completed run is missing {name}")
            print(f"Reusing completed {condition}", flush=True)
            continue
        if simulationFolder.exists() and any(simulationFolder.iterdir()):
            raise FileExistsError(
                f"{condition} is incomplete. Use a new run folder or move its incomplete directory aside."
            )
        runCommand(
            commandPrefix + ["+F", runFolder / f"{condition}.par"], runFolder / f"{condition}.log"
        )
        for name in ["vm.igb", "phie_recovery.igb", "parameters.par"]:
            if not (simulationFolder / name).is_file():
                raise FileNotFoundError(f"{condition}: simulator did not write {name}")
        (simulationFolder / "meshname.txt").write_text(str(runFolder / "mesh" / condition) + "\n")
        buildEGMs(simulationFolder, runFolder, config)
        saveJson(
            completedPath,
            dict(
                condition=condition,
                run=condition,
                state="pre" if condition == "intact" else "post",
                model="boundaryPilot",
                complete=True,
                processCount=processCount,
            ),
        )


def buildMesh(runFolder, config):
    """Use the notebook's mesher call, then tag the same nodes in both conditions."""
    meshFolder = Path(runFolder) / "mesh"
    meshFolder.mkdir(exist_ok=True)
    rawMesh = meshFolder / "slabRaw"
    mesherCommand = [findExecutable("mesher", "MESHER_BIN")]
    for axisNumber in range(3):
        mesherCommand += [
            f"-size[{axisNumber}]",
            config["slabSizeMm"][axisNumber] / 10,
            f"-resolution[{axisNumber}]",
            config["meshSpacingMm"] * 1000,
        ]
    mesherCommand += ["-mesh", rawMesh]
    runCommand(mesherCommand, Path(runFolder) / "mesher.log")
    nodePositions = readNodePositions(rawMesh)
    elementNodes = readElements(rawMesh)
    expectedHalfSize = np.asarray(config["slabSizeMm"]) / 2
    if not np.allclose(nodePositions.min(axis=0), -expectedHalfSize) or not np.allclose(
        nodePositions.max(axis=0), expectedHalfSize
    ):
        raise ValueError("The generated slab is not centred on the origin")
    elementCentres = nodePositions[elementNodes].mean(axis=1)
    meshNames = {}
    for condition in ["intact", "lesion"]:
        meshBase = meshFolder / condition
        tags = np.ones(len(elementNodes), dtype=int)
        if condition == "lesion":
            tags[signedDistance(elementCentres, config["boundary"]) > 0] = 5
        with open(str(meshBase) + ".elem", "w") as meshFile:
            meshFile.write(f"{len(elementNodes)}\n")
            for nodeIds, tag in zip(elementNodes, tags):
                meshFile.write("Tt " + " ".join(map(str, nodeIds)) + f" {tag}\n")
        shutil.copyfile(str(rawMesh) + ".pts", str(meshBase) + ".pts")
        # Explicit fibre direction; the contribution approximation assumes x-aligned fibres.
        with open(str(meshBase) + ".lon", "w") as fibreFile:
            fibreFile.write("1\n")
            np.savetxt(fibreFile, np.tile([1.0, 0.0, 0.0], (len(elementNodes), 1)), fmt="%.1f")
        meshNames[condition] = meshBase
    return meshNames


def writeElectrodes(runFolder, config):
    layout = makeElectrodeLayout(config)
    electrodeFolder = Path(runFolder) / "electrodes"
    electrodeFolder.mkdir(exist_ok=True)
    pointFile = electrodeFolder / "recoveryPoints.pts"
    np.savetxt(
        pointFile,
        layout["recoveryPoints"] * 1000,
        fmt="%.6f",
        header=str(len(layout["recoveryPoints"])),
        comments="",
    )
    np.savez_compressed(electrodeFolder / "layout.npz", **layout)
    return pointFile.with_suffix("")


def writeStimulus(meshBase, condition, config, stimulusBase):
    positions = readNodePositions(meshBase)
    stimulusMask = (
        positions[:, 0] <= positions[:, 0].min() + config["pacing"]["stripWidthMm"] + 1e-9
    )
    if condition == "lesion":
        # Exclude every node touching a lesion element, including interface nodes.
        elements = readElements(meshBase)
        lesionElements = signedDistance(positions[elements].mean(axis=1), config["boundary"]) > 0
        stimulusMask[np.unique(elements[lesionElements])] = False
    nodeIds = np.flatnonzero(stimulusMask)
    if not len(nodeIds):
        raise ValueError("No viable stimulus nodes")
    with open(str(stimulusBase) + ".vtx", "w") as stimulusFile:
        stimulusFile.write(f"{len(nodeIds)}\nextra\n")
        np.savetxt(stimulusFile, nodeIds, fmt="%d")
    return len(nodeIds)


def writeParameters(runFolder, condition, meshBase, stimulusBase, recoveryBase, config):
    """Combine the fixed model/solver template with this experiment's explicit settings."""
    baseParameters = (projectFolder / "3D_Code/3d_pilot.par").read_text()
    simulationFolder = runFolder / condition
    pace = config["pacing"]
    parameters = dict(
        meshname=meshBase,
        simID=simulationFolder,
        dt=config["timeStepUs"],
        spacedt=config["sampleStepMs"],
        tend=pace["cycleLengthMs"] * pace["beatCount"],
        phie_rec_ptf=recoveryBase,
    )
    parameters.update(
        {
            "stim[0].elec.vtx_file": str(stimulusBase) + ".vtx",
            "stim[0].pulse.strength": pace["strength"],
            "stim[0].ptcl.duration": pace["durationMs"],
            "stim[0].ptcl.bcl": pace["cycleLengthMs"],
            "stim[0].ptcl.npls": pace["beatCount"],
        }
    )
    parameterFile = runFolder / f"{condition}.par"
    conditionParameters = baseParameters
    if condition == "intact":
        conditionParameters = "\n".join(
            line for line in baseParameters.splitlines() if not line.startswith("gregion[1]")
        )
        conditionParameters = conditionParameters.replace("num_gregions = 2", "num_gregions = 1")
    resolvedLines = []
    for parameterName, parameterValue in parameters.items():
        if isinstance(parameterValue, (Path, str)):
            parameterValue = json.dumps(str(parameterValue))
        resolvedLines.append(f"{parameterName} = {parameterValue}\n")
    parameterFile.write_text(
        conditionParameters + "\n# Values resolved for this experiment\n" + "".join(resolvedLines)
    )


def buildEGMs(simulationFolder, runFolder, config):
    simulationFolder = Path(simulationFolder)
    voltageTime = igb_io.timeAxis(simulationFolder / "vm.igb")
    recoveryTime = igb_io.timeAxis(simulationFolder / "phie_recovery.igb")
    expectedEnd = config["pacing"]["cycleLengthMs"] * config["pacing"]["beatCount"]
    if (
        not np.array_equal(voltageTime, recoveryTime)
        or not np.isclose(voltageTime[0], 0)
        or not np.isclose(voltageTime[-1], expectedEnd)
    ):
        raise ValueError("Vm and recovery must cover the complete same simulation interval")
    if len(voltageTime) < 2 or not np.allclose(np.diff(voltageTime), config["sampleStepMs"]):
        raise ValueError("Recorded sample interval does not match the requested configuration")
    layout = dict(np.load(Path(runFolder) / "electrodes/layout.npz"))
    recovery = igb_io.read(simulationFolder / "phie_recovery.igb").T
    if recovery.shape[0] != len(layout["recoveryPoints"]) or not np.isfinite(recovery).all():
        raise ValueError("Invalid extracellular recovery data")
    electrodeSignals = recovery[layout["recoveryIndex"]].mean(axis=2)
    np.savez_compressed(
        simulationFolder / "egm.npz",
        uni=electrodeSignals,
        uni_xy=layout["centres"],
        uni_sample_xyz=layout["samples"],
        d_mm=signedDistance(layout["centres"], config["boundary"]),
        centre_xy_mm=layout["placements"][:, :2],
        p_gap_mm=layout["placements"][:, 2],
        t_ms=voltageTime,
    )


def startAnalysis(runFolder, config):
    """Archive the last derived analysis; never change raw simulation outputs."""
    runFolder = Path(runFolder)
    checkSimulationSettings(runFolder, config)
    previousNames = ["results", "figures", "physiologyChecks.csv", "physiologyStatus.json"]
    previousPaths = [runFolder / name for name in previousNames if (runFolder / name).exists()]
    if previousPaths:
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        archiveFolder = runFolder / "analysisArchive" / stamp
        archiveFolder.mkdir(parents=True)
        for path in previousPaths:
            shutil.move(str(path), archiveFolder / path.name)
    resultFolder = runFolder / "results"
    resultFolder.mkdir(exist_ok=True)
    saveJson(resultFolder / "analysisConfig.json", config)
    saveJson(
        resultFolder / "analysisManifest.json",
        dict(
            pythonVersion=sys.version,
            numpyVersion=np.__version__,
            startedUtc=datetime.now(timezone.utc).isoformat(),
        ),
    )
    snapshotFolder = resultFolder / "analysisSource"
    snapshotFolder.mkdir()
    for folder in [projectFolder / "3D_Code", projectFolder / "Shared_Pipeline"]:
        for path in folder.glob("*.py"):
            shutil.copy2(path, snapshotFolder / path.name)


def checkCompletedSimulations(runFolder):
    """Require both raw recordings and completed finite-electrode exports."""
    runFolder = Path(runFolder)
    for condition in ["intact", "lesion"]:
        simulationFolder = runFolder / condition
        for filename in ["meta.json", "vm.igb", "phie_recovery.igb", "parameters.par", "egm.npz"]:
            if not (simulationFolder / filename).is_file():
                raise FileNotFoundError(
                    f"{condition}: missing {filename}; finish simulation/export before processing"
                )
        if not json.loads((simulationFolder / "meta.json").read_text()).get("complete"):
            raise ValueError(f"{condition} has not completed")


def checkSimulationSettings(runFolder, config):
    """Allow reanalysis, but reject changes that need a new tissue simulation.

    Saved settings and the parameter template are compared directly. Source copies
    provide provenance; if you edit mesh/stimulus/recovery logic, use a new folder.
    """
    runFolder = Path(runFolder)
    savedConfig = json.loads((runFolder / "config.json").read_text())
    for name, value in config.items():
        if name not in ["analysis", "acceptance"] and savedConfig.get(name) != value:
            raise ValueError(f"Simulation setting {name} changed. Use a new run folder.")
    savedTemplate = runFolder / "sourceSnapshot/3d_pilot.par"
    currentTemplate = projectFolder / "3D_Code/3d_pilot.par"
    if not savedTemplate.is_file() or savedTemplate.read_text() != currentTemplate.read_text():
        raise ValueError(
            "The openCARP parameter template changed or is missing. Use a new run folder."
        )


def readNodePositions(meshBase):
    """Mesh coordinates are stored in micrometres; return millimetres."""
    return np.loadtxt(str(meshBase) + ".pts", skiprows=1, ndmin=2) / 1000


def readElements(meshBase):
    elementRows = np.loadtxt(str(meshBase) + ".elem", skiprows=1, dtype=str, ndmin=2)
    if elementRows.shape[1] != 6 or not np.all(elementRows[:, 0] == "Tt"):
        raise ValueError("Expected tagged tetrahedra")
    return elementRows[:, 1:5].astype(int)


def runCommand(command, logPath):
    print("Running:", " ".join(map(str, command)), flush=True)
    with open(logPath, "w") as logFile:
        logFile.write("Command: " + " ".join(map(str, command)) + "\n")
        logFile.flush()
        subprocess.run(
            [str(value) for value in command], stdout=logFile, stderr=subprocess.STDOUT, check=True
        )


def findExecutable(name, environmentName):
    executable = os.environ.get(environmentName) or shutil.which(name)
    if not executable or not (shutil.which(executable) or os.access(executable, os.X_OK)):
        raise RuntimeError(
            f"{name} is unavailable. Activate your openCARP environment or set {environmentName}."
        )
    return str(executable)


def loadConfig(configPath):
    config = json.loads(Path(configPath).read_text())
    validateConfig(config)
    return config


def saveJson(path, values):
    Path(path).write_text(json.dumps(values, indent=2, allow_nan=False) + "\n")


def validateConfig(config):
    """Reject inconsistent inputs before the expensive mesh and simulation steps."""
    integerSettings = {
        "beatCount": config["pacing"]["beatCount"],
        "axialSamples": config["electrodes"]["axialSamples"],
        "angularSamples": config["electrodes"]["angularSamples"],
        "filterOrder": config["analysis"]["filterOrder"],
        "noiseRepeats": config["analysis"]["noiseRepeats"],
        "noiseSeed": config["analysis"]["noiseSeed"],
    }
    for name, value in integerSettings.items():
        if not isinstance(value, int) or isinstance(value, bool) or value < 0:
            raise ValueError(f"{name} must be a non-negative integer")
    if config["analysis"]["filterOrder"] < 1:
        raise ValueError("Filter order must be positive")
    if not 0 < config["analysis"]["omniAngleStepDeg"] <= 180:
        raise ValueError("Omnipolar angle step must be in (0, 180] degrees")
    size = np.asarray(config["slabSizeMm"], float)
    spacing = config["meshSpacingMm"]
    if size.shape != (3,) or not np.isfinite(size).all() or np.any(size <= 0) or spacing <= 0:
        raise ValueError("Slab dimensions and mesh spacing must be positive")
    if not np.allclose(size / spacing, np.round(size / spacing)):
        raise ValueError("Each slab dimension must be a multiple of meshSpacingMm")
    if min(size / spacing) < 2:
        raise ValueError("Use at least two elements through each slab direction")
    if config["boundary"]["shape"] not in ("straight", "circle"):
        raise ValueError("Supported boundary shapes: straight, circle")
    if (
        config["boundary"]["shape"] == "straight"
        and not -size[1] / 2 < config["boundary"]["positionMm"] < size[1] / 2
    ):
        raise ValueError("Straight boundary must lie inside the slab")
    if config["boundary"]["shape"] == "circle":
        radius = config["boundary"]["radiusMm"]
        if radius <= 0 or np.any(np.abs(config["boundary"]["centreMm"]) + radius >= size[:2] / 2):
            raise ValueError("Circle must fit inside the slab")
    pace = config["pacing"]
    if (
        pace["beatCount"] < 3
        or pace["cycleLengthMs"] <= pace["durationMs"]
        or pace["durationMs"] <= 0
        or pace["strength"] <= 0
    ):
        raise ValueError("Need at least three beats and a positive, valid pacing protocol")
    if not 0 < pace["stripWidthMm"] < size[0] / 2:
        raise ValueError("Stimulus strip must be narrower than half the slab")
    sampleStep = config["sampleStepMs"]
    if sampleStep <= 0 or config["timeStepUs"] <= 0 or sampleStep * 1000 < config["timeStepUs"]:
        raise ValueError("Invalid integration or recording time step")
    if not np.isclose(
        sampleStep * 1000 / config["timeStepUs"], round(sampleStep * 1000 / config["timeStepUs"])
    ):
        raise ValueError("sampleStepMs must be an integer multiple of the integration step")
    analysis = config["analysis"]
    start, end = analysis["beatWindowMs"]
    if not 0 <= start < end <= pace["cycleLengthMs"]:
        raise ValueError("Measurement window must fit within one pacing cycle")
    if analysis["filterHistoryMs"] <= 0:
        raise ValueError("Keep positive filter history")
    band = analysis["differentialBandHz"]
    if not 0 < band[0] < band[1] < 500 / sampleStep:
        raise ValueError("Filter band must lie strictly below Nyquist")
    electrodes = config["electrodes"]
    if len(electrodes["gridOffsetsMm"]) != 4 or not np.allclose(
        np.diff(electrodes["gridOffsetsMm"]), np.diff(electrodes["gridOffsetsMm"])[0]
    ):
        raise ValueError("The existing recording code needs four equally spaced offsets")
    if np.diff(electrodes["gridOffsetsMm"])[0] <= 0 or len(set(electrodes["centreYmm"])) != len(
        electrodes["centreYmm"]
    ):
        raise ValueError("Grid offsets must increase and placements must be unique")
    if (
        not electrodes["centreYmm"]
        or not electrodes["gapsMm"]
        or min(electrodes["gapsMm"]) <= 0
        or electrodes["radiusMm"] <= 0
        or electrodes["lengthMm"] <= 0
    ):
        raise ValueError("Electrode dimensions, standoffs and placements must be valid")
    if electrodes["axialSamples"] < 1 or electrodes["angularSamples"] < 4:
        raise ValueError("Use positive axial and at least four angular samples")
    layout = makeElectrodeLayout(config)
    if np.any(np.abs(layout["samples"][..., :2]) >= size[:2] / 2):
        raise ValueError("All finite electrode samples must project inside the slab")
    if analysis["noiseSdMv"] < 0 or analysis["noiseRepeats"] < 0:
        raise ValueError("Noise SD and repetition count cannot be negative")
    if len(analysis["groundTruthCentreYmm"]) == 0 or not set(
        analysis["groundTruthCentreYmm"]
    ).issubset(electrodes["centreYmm"]):
        raise ValueError("Ground-truth centres must be actual catheter placements")
    if (
        analysis["referenceCentreYmm"] not in electrodes["centreYmm"]
        or analysis["referenceGapMm"] not in electrodes["gapsMm"]
    ):
        raise ValueError("The calibration reference must be a simulated placement")
    if len(set(electrodes["gapsMm"])) != len(electrodes["gapsMm"]):
        raise ValueError("Electrode gaps must be unique")
    if analysis["targetBipolarMv"] <= 0:
        raise ValueError("Calibration target must be positive")
    return config
