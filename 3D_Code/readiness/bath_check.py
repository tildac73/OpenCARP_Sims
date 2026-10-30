"""Step 6 feasibility: openCARP pseudo-bidomain with a blood bath, checked against an independent solve.

Setup (readiness_kit.json "bath"): tissue slab (fibres x, tuned conductivities, Courtemanche limit
cycle) with a blood layer on the catheter side (z > 2 mm), conductivity 0.74 S/m (Traugott et al.
1981). openCARP parameters used and where they are documented:
  bidomain = 2 (pseudo-bidomain)                      openCARP parameter list
  phys_region[].ptype 0 = intracellular, 1 = extracellular electrics   openCARP parameter list
  gregion[].g_bath (isotropic, non-myocardium)        openCARP parameter list
  stim[].crct.type 3 = extracellular ground (0 mV)    openCARP parameter list (stimulus types)

Checks, in order (each recorded; later ones are only interpretable if earlier ones pass):
  1. openCARP ran once and wrote Vm and phi_e;
  2. Vm has one value per tissue node. Tissue nodes are numbered first, and the planar activation
     map confirms that Vm index i is global node i (an ordering ASSUMPTION about openCARP's
     intracellular submesh, verified here rather than presumed);
  3. phi_e is ~0 on the grounded face;
  4. phi_e at every electrode sample equals the independent P1 finite-element solve
     (fem_extracellular.py) of the same discrete problem from the same Vm (PROPOSED tolerance:
     relative RMS <= 1 %, correlation >= 0.9999).
"""
import json
from pathlib import Path

import numpy as np

import common
import boundary_setup
import igb_io
from boundary_checks import activationTimes
from boundary_geometry import makeElectrodeLayout
import fem_extracellular as fem


def buildMesh(folder, settings):
    """Mesher box, shifted so tissue is z in [-2, 2] mm and bath z in [2, 2 + B]; tissue nodes first."""
    tissue = np.asarray(settings["tissueSizeMm"], float)
    bath = float(settings["bathThicknessMm"])
    size = [tissue[0], tissue[1], tissue[2] + bath]
    positions, elements = common.generateBox(Path(folder) / "mesh", "raw", size, settings["spacingMm"])
    positions = positions.copy()
    positions[:, 2] += bath / 2                                  # mesher z: [-(t+B)/2, (t+B)/2]
    surface = tissue[2] / 2
    centres = positions[elements].mean(axis=1)
    isTissueElement = centres[:, 2] < surface
    tissueNodes = np.unique(elements[isTissueElement])
    order = np.concatenate([tissueNodes, np.setdiff1d(np.arange(len(positions)), tissueNodes)])
    newIndex = np.empty(len(positions), int); newIndex[order] = np.arange(len(positions))
    positions, elements = positions[order], newIndex[elements]
    tags = np.where(isTissueElement, 1, 2)
    common.writeMesh(Path(folder) / "mesh" / "bath", positions, elements, tags)
    return positions, elements, isTissueElement, len(tissueNodes)


def electrodeSamples(settings):
    base = json.loads((common.REPOSITORY / "3D_Code/configs/boundary_pilot.json").read_text())
    base["slabSizeMm"] = list(settings["tissueSizeMm"])
    base["electrodes"]["centreYmm"] = [settings["electrodeCentreYmm"]]
    base["electrodes"]["gapsMm"] = list(settings["gapsMm"])
    layout = makeElectrodeLayout(base)
    return layout["samples"], layout["placements"]                  # (placements, 16, S, 3) mm


def parameterText(tuned, settings):
    text = common.templateForTissue(tuned["conductivitiesSPerM"])
    replacements = [("bidomain = 0", f"bidomain = {settings['bidomainMode']}"),
                    ("num_gregions = 1", "num_gregions = 2"), ("num_stim = 1", "num_stim = 2")]
    for old, new in replacements:
        if text.count(old) != 1:
            raise ValueError(f"Template line '{old}' not found exactly once")
        text = text.replace(old, new)
    text += f"""
# Readiness kit, Step 6 feasibility: blood bath (tag 2) on the catheter side.
num_phys_regions = 2
phys_region[0].ptype = 0
phys_region[0].name = "intracellular"
phys_region[0].num_IDs = 1
phys_region[0].ID[0] = 1
phys_region[1].ptype = 1
phys_region[1].name = "extracellular"
phys_region[1].num_IDs = 2
phys_region[1].ID[0] = 1
phys_region[1].ID[1] = 2
gregion[1].num_IDs = 1
gregion[1].ID[0] = 2
gregion[1].g_bath = {settings['bloodConductivitySPerM']}
stim[1].crct.type = 3
"""
    return text


def prepare(folder, settings, tuned, kitSettings):
    folder = Path(folder)
    positions, elements, isTissue, tissueNodeCount = buildMesh(folder, settings)
    tissueNode = np.arange(len(positions)) < tissueNodeCount
    stimulus = np.flatnonzero(tissueNode & (positions[:, 0] <= positions[:, 0].min() + kitSettings["cvTuning"]["stimulus"]["stripWidthMm"] + 1e-9))
    ground = np.flatnonzero(np.isclose(positions[:, 2], positions[:, 2].max()))
    common.writeVertices(folder / "stimulus.vtx", stimulus)
    common.writeVertices(folder / "ground.vtx", ground)
    cv = kitSettings["cvTuning"]
    config = dict(timeStepUs=settings["timeStepUs"], conditioning=dict(enabled=True, beatCount=cv["conditioningBeats"]),
                  pacing=dict(cycleLengthMs=cv["cycleLengthMs"], durationMs=cv["stimulus"]["durationMs"],
                              strength=tuned["stimulusStrength"]))
    boundary_setup.conditionCells(folder, config)
    values = {"meshname": folder / "mesh" / "bath", "simID": folder / "sim", "dt": settings["timeStepUs"],
              "spacedt": settings["outputStepMs"], "tend": settings["durationMs"],
              "stim[0].elec.vtx_file": folder / "stimulus.vtx", "stim[0].pulse.strength": tuned["stimulusStrength"],
              "stim[0].ptcl.duration": cv["stimulus"]["durationMs"], "stim[0].ptcl.bcl": settings["durationMs"],   # one pulse; openCARP needs duration <= bcl <= tend
              "stim[0].ptcl.npls": 1, "stim[1].elec.vtx_file": folder / "ground.vtx",
              "imp_region[0].im_sv_init": folder / "conditioning" / "Courtemanche.sv"}
    parameterFile = common.writeParameterFile(folder / "bath.par", parameterText(tuned, settings), values)
    np.savez_compressed(folder / "bathMesh.npz", positions=positions, elements=elements, isTissueElement=isTissue,
                        tissueNodeCount=tissueNodeCount, groundNodes=ground)
    return parameterFile, positions


def findPhie(simulationFolder):
    folder = Path(simulationFolder)
    candidates = [p for p in sorted(folder.glob("phie*.igb")) if "recovery" not in p.name]
    if len(candidates) != 1:
        raise FileNotFoundError(f"{folder}: expected one extracellular potential file, found {[p.name for p in candidates]}")
    return candidates[0]


def check(folder, settings, tuned):
    folder = Path(folder)
    mesh = dict(np.load(folder / "bathMesh.npz"))
    positions, elements = mesh["positions"], mesh["elements"]
    tissueCount = int(mesh["tissueNodeCount"])
    vmPath, phiePath = folder / "sim" / "vm.igb", findPhie(folder / "sim")
    _, vmNodes, frames, _ = igb_io.header(vmPath)
    _, phieNodes, phieFrames, _ = igb_io.header(phiePath)
    result = dict(vmNodes=vmNodes, tissueNodes=tissueCount, phieNodes=phieNodes, meshNodes=len(positions),
                  vmFrames=frames, phieFrames=phieFrames, phieFile=phiePath.name)
    result["vmIsTissueOnly"] = vmNodes == tissueCount
    result["phieIsAllNodes"] = phieNodes == len(positions)
    if not (result["vmIsTissueOnly"] and result["phieIsAllNodes"] and frames == phieFrames):
        result["passed"] = False
        result["reason"] = "Output sizes do not match the tissue/whole-mesh assumption; FEM comparison not attempted."
        return result
    time = igb_io.timeAxis(vmPath)
    vm = igb_io.read(vmPath)
    activation, counts = activationTimes(time, vm, -20.0)
    surfaceTissue = positions[:tissueCount]          # all tissue nodes (Vm index = global index, if confirmed)
    interior = (np.abs(surfaceTissue[:, 0]) <= 0.4 * settings["tissueSizeMm"][0])
    try:
        plane = common.fitPlanarSpeed(surfaceTissue[interior], activation[interior], 0)
        result["vmOrderingPlaneFit"] = plane
        result["vmOrderingConfirmed"] = bool(plane["directionErrorDeg"] <= 5 and plane["planeResidual95Ms"] <= 2.0)
    except ValueError as error:          # no propagation: report, still compare phi_e (same Vm feeds both)
        result["vmOrderingPlaneFit"] = f"not available: {error}"
        result["vmOrderingConfirmed"] = False
    conductivities = tuned["conductivitiesSPerM"]
    sigmaIntra = np.tile([conductivities["g_il"], conductivities["g_it"], conductivities["g_in"]], (len(elements), 1))
    sigmaExtra = np.tile([conductivities["g_el"], conductivities["g_et"], conductivities["g_en"]], (len(elements), 1))
    solver = fem.ExtracellularSolver(positions, elements, mesh["isTissueElement"], sigmaIntra, sigmaExtra,
                                     settings["bloodConductivitySPerM"], mesh["groundNodes"])
    samples, placements = electrodeSamples(settings)
    points = samples.reshape(-1, 3)
    interpolate = fem.barycentricInterpolator(positions, elements, points)
    femSamples = np.empty((len(points), frames)); carpSamples = np.empty((len(points), frames))
    groundMax, nodeErrorMax, nodeScale = 0.0, 0.0, 0.0
    for first in range(0, frames, 100):
        block = slice(first, min(first + 100, frames))
        vmAll = np.zeros((len(positions), block.stop - block.start)); vmAll[:tissueCount] = vm[block].T
        phiFem = solver.solve(vmAll)
        phiCarp = igb_io.read(phiePath, block).T.astype(float)
        femSamples[:, block] = interpolate @ phiFem
        carpSamples[:, block] = interpolate @ phiCarp
        groundMax = max(groundMax, float(np.abs(phiCarp[mesh["groundNodes"]]).max()))
        nodeErrorMax = max(nodeErrorMax, float(np.abs(phiCarp - phiFem).max()))
        nodeScale = max(nodeScale, float(np.abs(phiFem).max()))
    difference = carpSamples - femSamples
    rms = lambda x: np.sqrt(np.mean(x ** 2, axis=1))
    relativeRms = rms(difference) / np.maximum(rms(femSamples), 1e-30)
    correlation = np.array([np.corrcoef(a, b)[0, 1] for a, b in zip(carpSamples, femSamples)])
    result.update(groundMaxAbsMv=groundMax, nodeMaxAbsErrorMv=nodeErrorMax, nodeMaxAbsPhiMv=nodeScale,
                  sampleRelativeRmsMax=float(relativeRms.max()), sampleRelativeRmsMedian=float(np.median(relativeRms)),
                  sampleCorrelationMin=float(correlation.min()),
                  sampleCount=int(len(points)), placements=placements.tolist(),
                  electrodePpMvByGap={f"{gap:g}": float(np.median(np.ptp(femSamples.reshape(len(placements), -1, frames)[i], axis=1)))
                                      for i, gap in enumerate(placements[:, 2])})
    result["passed"] = bool(result["vmOrderingConfirmed"] and result["sampleRelativeRmsMax"] <= settings["relativeRmsTolerance"]
                            and result["sampleCorrelationMin"] >= settings["correlationTolerance"])
    np.savez_compressed(folder / "bathSignals.npz", openCarpMv=carpSamples.astype(np.float32),
                        femMv=femSamples.astype(np.float32), timeMs=time, samples=samples, placements=placements)
    return result


def runStage(stageFolder, settings, tuned, kitSettings, processCount, smoke=False):
    folder = Path(stageFolder) / ("smoke_bath" if smoke else "bath")
    summaryPath = folder / "bathSummary.json"
    if summaryPath.exists() and not smoke:
        return json.loads(summaryPath.read_text())
    if folder.exists() and any(folder.iterdir()) and not smoke:
        raise FileExistsError(f"{folder} is incomplete; move it aside before rerunning")
    parameterFile, positions = prepare(folder, settings, tuned, kitSettings)
    prefix, ranks = common.simulatorPrefix(processCount, len(positions))
    if smoke:
        smokePar = common.smokeCopy(parameterFile, folder / "smoke")
        common.runLogged(prefix + ["+F", smokePar], folder / "smoke.log")
        common.checkSingleLaunch(folder / "smoke.log")
        findPhie(folder / "smoke")
        return dict(smokePassed=True)
    common.runLogged(prefix + ["+F", parameterFile], folder / "openCARP.log")
    common.checkSingleLaunch(folder / "openCARP.log")
    summary = check(folder, settings, tuned)
    summary.update(ranks=ranks, wallSeconds=common.lastProgressSeconds(folder / "openCARP.log"))
    common.saveJson(summaryPath, summary)
    return summary
