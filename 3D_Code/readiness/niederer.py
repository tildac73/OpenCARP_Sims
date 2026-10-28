"""Step 3: N-version benchmark of Niederer et al. 2011 on this openCARP build.

Specification: Niederer et al., Phil Trans R Soc A 369:4331-4351 (2011), doi:10.1098/rsta.2011.0139.
Parameter blocks (ionic model, conductivities, stimulus, LAT) are copied from the official openCARP
example 02_EP_tissue/03G_study_resolution. Solver settings are the pilot's (3d_pilot.par:
monodomain, Crank-Nicolson parabolic solve, cg_tol_parab 1e-8, mass lumping), because the point is
to check the discretisation this project uses.

Geometry: 20 x 7 x 3 mm cuboid, fibres along x, stimulus in a 1.5 mm cube at one corner.
The mesher centres the box at the origin, so the benchmark origin is (-10, -3.5, -1.5) mm.
Output: activation time (0 mV upward crossing) along the diagonal from the stimulated corner to the
opposite corner, for every spacing x time-step combination. No reference values are hard-coded:
the result is compared with Niederer et al. Fig. 3 by the reader, and convergence is reported.
"""
from pathlib import Path

import numpy as np
import pandas as pd

import common

BLOCKS = """# Niederer et al. 2011 benchmark; blocks from openCARP example 02_EP_tissue/03G_study_resolution.
bidomain = 0
parab_solve = 1
cg_tol_parab = 1e-8
mass_lumping = {massLumping}

num_imp_regions                 = 1
imp_region[0].name              = "Ventricle"
imp_region[0].im                = tenTusscherPanfilov
imp_region[0].im_param          = "flags=EPI"
imp_region[0].cellSurfVolRatio  = 0.14
imp_region[0].volFrac           = 1

num_gregions    = 1
gregion[0].num_IDs = 1
gregion[0].ID[0] = 1
gregion[0].g_il = 0.17
gregion[0].g_it = 0.019
gregion[0].g_in = 0.019
gregion[0].g_el = 0.62
gregion[0].g_et = 0.24
gregion[0].g_en = 0.24

num_stim                = 1
stim[0].crct.type       = 0
stim[0].pulse.shape     = 0
stim[0].elec.geom_type  = 2
stim[0].elec.p0[0]      = {p0x}
stim[0].elec.p0[1]      = {p0y}
stim[0].elec.p0[2]      = {p0z}
stim[0].elec.p1[0]      = {p1x}
stim[0].elec.p1[1]      = {p1y}
stim[0].elec.p1[2]      = {p1z}
stim[0].pulse.strength  = {strength}
stim[0].ptcl.duration   = {duration}
stim[0].ptcl.start      = 0.00
stim[0].ptcl.npls       = 1

timedt = 1
num_LATs             = 1
lats[0].ID           = act
lats[0].measurand    = 0
lats[0].all          = 0
lats[0].mode         = 0
lats[0].threshold    = 0.
lats[0].method       = 1
"""


def diagonalPoints(sizeMm, count=21):
    """Benchmark diagonal (stimulated corner -> opposite corner) in mesher-centred coordinates (mm)."""
    half = np.asarray(sizeMm, float) / 2
    fraction = np.linspace(0, 1, count)
    return fraction, -half + fraction[:, None] * (2 * half)


def parameterText(settings, massLumping):
    half = np.asarray(settings["slabSizeMm"], float) / 2
    p0 = -half * 1000
    p1 = (-half + np.asarray(settings["stimulusBoxMm"], float)) * 1000
    return BLOCKS.format(massLumping=massLumping, p0x=p0[0], p0y=p0[1], p0z=p0[2], p1x=p1[0], p1y=p1[1], p1z=p1[2],
                         strength=settings["stimulusStrength"], duration=settings["stimulusDurationMs"])


def runCase(stageFolder, settings, spacingMm, timeStepUs, massLumping, processCount, smoke=False):
    job = Path(stageFolder) / f"dx{spacingMm:g}_dt{timeStepUs:g}_ml{massLumping}"
    resultPath = job / "result.csv"
    if resultPath.exists() and not smoke:
        return pd.read_csv(resultPath)
    if job.exists() and any(job.iterdir()) and not smoke:
        raise FileExistsError(f"{job} is incomplete; move it aside before rerunning")
    positions, _ = common.generateBox(job / "mesh", "block", settings["slabSizeMm"], spacingMm)
    values = {"meshname": job / "mesh" / "block", "simID": job / "sim", "dt": timeStepUs,
              "spacedt": settings["outputStepMs"], "tend": settings["durationMs"]}
    parameterFile = common.writeParameterFile(job / "benchmark.par", parameterText(settings, massLumping), values)
    prefix, ranks = common.simulatorPrefix(processCount, len(positions))
    if smoke:
        smokePar = common.smokeCopy(parameterFile, job / "smoke")
        common.runLogged(prefix + ["+F", smokePar], job / "smoke.log")
        common.checkSingleLaunch(job / "smoke.log")
        return None
    common.runLogged(prefix + ["+F", parameterFile], job / "openCARP.log")
    common.checkSingleLaunch(job / "openCARP.log")
    nodes, times = common.readActivationFile(common.findActivationFile(job / "sim"), len(positions))
    first = common.firstActivationInWindow(nodes, times, len(positions), 0.0, np.inf)
    fraction, points = diagonalPoints(settings["slabSizeMm"])
    rows = []
    for f, point in zip(fraction, points):
        nearest = int(np.argmin(np.linalg.norm(positions - point, axis=1)))
        rows.append(dict(spacingMm=spacingMm, timeStepUs=timeStepUs, massLumping=massLumping, diagonalFraction=f,
                         distanceAlongDiagonalMm=float(f * np.linalg.norm(settings["slabSizeMm"])),
                         nodeOffsetMm=float(np.linalg.norm(positions[nearest] - point)),
                         activationMs=float(first[nearest]), ranks=ranks,
                         unactivatedNodeFraction=float(np.mean(~np.isfinite(first)))))
    table = pd.DataFrame(rows)
    table.to_csv(resultPath, index=False)
    return table


def runStage(stageFolder, settings, processCount):
    tables = []
    for massLumping in settings["massLumping"]:
        for spacing in settings["spacingsMm"]:
            for timeStep in settings["timeStepsUs"]:
                tables.append(runCase(stageFolder, settings, spacing, timeStep, massLumping, processCount))
    table = pd.concat(tables, ignore_index=True)
    table.to_csv(Path(stageFolder) / "benchmarkActivation.csv", index=False)
    corner = table[np.isclose(table.diagonalFraction, 1.0)]
    finest = corner.sort_values(["spacingMm", "timeStepUs"]).iloc[0]
    summary = dict(
        farCornerActivationMs=corner[["spacingMm", "timeStepUs", "massLumping", "activationMs"]].to_dict("records"),
        finestCase=dict(spacingMm=float(finest.spacingMm), timeStepUs=float(finest.timeStepUs),
                        farCornerMs=float(finest.activationMs)),
        allActivated=bool((table.unactivatedNodeFraction == 0).all()),
        interpretation="Compare the activation-versus-diagonal curves with Niederer et al. 2011 Fig. 3; "
                       "no reference numbers are embedded here.",
    )
    common.saveJson(Path(stageFolder) / "benchmarkSummary.json", summary)
    return summary


def smoke(stageFolder, settings, processCount):
    runCase(stageFolder, settings, settings["spacingsMm"][0], settings["timeStepsUs"][0], settings["massLumping"][0],
            processCount, smoke=True)
