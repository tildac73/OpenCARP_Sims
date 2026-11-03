"""The analysis stages on a synthetic openCARP-format run (no openCARP needed).

The small test mesh (config/test_small.json) is built with efov.meshbuild; Vm is an analytic planar
wave (four beats, x direction), and phi_e at the output nodes is the exact solution of the discrete
pseudo-bidomain problem K phi = -S_m Vm with the corner ground. The stages window -> electrograms ->
measurements -> export -> groundTruth (Python lead fields) then run on these files. Passing means the
stages agree with each other: node order, electrode node sets, time axes, the acquisition chain and
the reconstruction of every ground-truth recording.
"""
import json

import numpy as np
import pytest
from scipy.sparse.linalg import splu

from efov import config as cfg, electrograms, export, groundtruth, leadfield, measurements, meshbuild, physiology, simulation
from efov.io import saveJson, writeMesh, writeVertices
from helpers import SMALL


def writeIgb(path, values, startMs, stepMs):
    """Minimal openCARP-style IGB writer (1024-byte header, little-endian float32, (frames, nodes))."""
    frames, nodes = values.shape
    header = (f"x:{nodes} y:1 z:1 t:{frames} type:float systeme:little_endian org_t:{startMs:g} "
              f"dim_t:{(frames - 1) * stepMs:g} inc_t:{stepMs:g}").ljust(1023) + "\x0c"
    with open(path, "wb") as stream:
        stream.write(header.encode("latin-1"))
        stream.write(np.asarray(values, "<f4").tobytes())


def actionPotential(t, arrival):
    """Upstroke (0.5 ms) and exponential repolarisation (150 ms) from -80 mV: a smooth stand-in for Courtemanche."""
    dt = t - arrival
    up = 0.5 * (1 + np.tanh(dt / 1.0))                    # logistic with slope 1/0.5 ms, no overflow
    return -80 + 110 * up * np.exp(-np.clip(dt, 0, None) / 150.0)


def writeCondition(config, job, lesion):
    """Mesh, Vm, activation times and exact phi_e of one synthetic condition in openCARP's file layout."""
    import itertools
    job.mkdir(parents=True)
    size, spacing, bath = cfg.slabSizeMm(config), cfg.spacingMm(config), config["fixed"]["bath"]["thicknessMm"]
    lower, upper = -size / 2 - bath, size / 2 + bath
    axes = [np.linspace(lo, hi, int(round((hi - lo) / s)) + 1) for lo, hi, s in zip(lower, upper, spacing)]
    grid = np.stack(np.meshgrid(*axes, indexing="ij"), axis=-1)
    shape = grid.shape[:3]
    cells = np.array(list(itertools.product(*(range(n - 1) for n in shape))))
    tets = []
    for order in itertools.permutations(range(3)):                     # Kuhn tetrahedra, per-axis spacing
        corner = cells.copy(); ids = [np.ravel_multi_index(corner.T, shape)]
        for axis in order:
            corner = corner.copy(); corner[:, axis] += 1; ids.append(np.ravel_multi_index(corner.T, shape))
        tets.append(np.stack(ids, axis=1))
    positions, elements = grid.reshape(-1, 3), np.concatenate(tets)
    centres = positions[elements].mean(axis=1)
    inside = np.all(np.abs(centres) < size / 2, axis=1)
    isLesion = lesion & (centres[:, 1] < config["fixed"]["tissue"]["boundaryYmm"])
    tags = np.where(inside, np.where(isLesion, 5, 1), 0)
    positions, elements, tags, fibres = meshbuild.runMesh(positions, elements, tags, size[2], "none", config["values"]["fibres"])
    meshBase = writeMesh(job / "mesh" / "slab", positions, elements, tags, fibres)
    ground = meshbuild.groundNodes(positions, elements, tags, config["fixed"]["bath"]["groundCubeMm"])
    writeVertices(job / "ground.vtx", ground)
    phieNodes = meshbuild.boxNodes(positions, *cfg.phieBoxMm(config))
    np.save(job / "phieNodes.npy", phieNodes)
    tissueNodes = np.unique(elements[tags != 0])
    viable = np.isin(tissueNodes, np.unique(elements[tags == 1]))
    p = positions[tissueNodes]
    cycle, beats, step = cfg.cycleLengthMs(config), cfg.beatCount(config), cfg.sampleStepMs(config)
    times = np.round(np.arange(0, beats * cycle + 1e-9, step), 6)
    arrivals = [b * cycle + 5 + (p[:, 0] + size[0] / 2) / 0.9 for b in range(beats)]
    vm = np.full((len(times), len(tissueNodes)), -80.0)
    for arrival in arrivals:
        vm[:, viable] = np.maximum(vm[:, viable], actionPotential(times[:, None], arrival[None, viable]))
    writeIgb(job / "vm.igb", vm, 0.0, step)
    rows = [(n, a[n]) for a in arrivals for n in np.flatnonzero(viable)]
    np.savetxt(job / "act-thresh.dat", np.array(rows), fmt=["%d", "%.6f"], delimiter="\t")
    restart = cfg.restartTimeMs(config)
    inTissue = tags != 0
    _, gBath = leadfield.principalConductivities(config)
    bulk = np.repeat((gBath * np.eye(3))[None], len(tags), axis=0)
    bulk[inTissue] = leadfield.tissueTensors(config, tags[inTissue], fibres[inTissue], "bulk")
    source = np.zeros_like(bulk)
    source[inTissue] = leadfield.tissueTensors(config, tags[inTissue], fibres[inTissue], "monodomain")
    K = leadfield.assemble(positions, elements, bulk)
    S = leadfield.assemble(positions, elements, source)
    free = np.setdiff1d(np.arange(len(positions)), ground)
    solve = splu(K[free][:, free].tocsc()).solve
    frames = np.flatnonzero(times >= restart - 1e-9)
    phie = np.zeros((len(frames), len(phieNodes)), np.float32)
    for first in range(0, len(frames), 250):                      # blocks of frames keep memory small
        block = frames[first:first + 250]
        full = np.zeros((len(positions), len(block)))
        full[tissueNodes] = vm[block].T
        phi = np.zeros_like(full)
        phi[free] = solve(-(S @ full)[free])
        phie[first:first + len(block)] = phi[phieNodes].T
    writeIgb(job / "phie.igb", phie, restart, step)
    saveJson(job / "restart.json", dict(restartMs=restart, phieStartMs=restart))
    saveJson(job / "runRecord.json", dict(meshBase=str(meshBase), nodeSets=dict(nodes=len(positions))))


@pytest.fixture(scope="module")
def syntheticRun(tmp_path_factory):
    config = cfg.loadConfig(SMALL)
    config["fixed"]["groundTruth"]["leadFieldMethod"] = "python"
    root = tmp_path_factory.mktemp("synthetic")
    for condition in ("intact", "lesion"):
        writeCondition(config, root / "simulations" / condition, condition == "lesion")
    return config, root


def test_analysis_stages_agree_on_a_synthetic_run(syntheticRun):
    config, root = syntheticRun
    jobs = {c: root / "simulations" / c for c in ("intact", "lesion")}
    tissues = {c: simulation.tissueModel(j) for c, j in jobs.items()}
    window = physiology.deriveWindow(config, jobs["intact"], tissues["intact"])["windowMs"]
    egms = root / "electrograms"; egms.mkdir()
    info = electrograms.runElectrograms(config, jobs, tissues, window, egms)
    assert info["lesion"]["nodesPerElectrodeMin"] > 0
    folder = root / "measurements"; folder.mkdir()
    points, calibration, status = measurements.runMeasurements(config, egms, window, folder)
    assert abs(calibration["targetBipolarMv"] - 1.44) < 1e-12 and status["filterHistory"]
    datasets = export.runExport(config, folder, root / "export")
    assert len(datasets) == len(config["values"]["gapsMm"]) * (config["fixed"]["analysis"]["noise"]["repeats"] + 1)
    gt = root / "groundTruth"; gt.mkdir()
    checks = groundtruth.runGroundTruth(config, jobs, tissues, egms, folder, calibration["gain"], window, gt)
    assert checks["passed"], checks
    summary = __import__("pandas").read_csv(gt / "summary.csv")
    assert set(summary.kind) == {"uni", "bi", "omni"} and (gt / "shares_lesion_gap0.25mm.vtk").exists()
    # Ground-truth figures from the same outputs (a minimal stand-in for pipeline.Run)
    from types import SimpleNamespace
    from efov.groundtruth_figures import makeGroundTruthFigures
    fake = SimpleNamespace(out=root, config=config, tissue=lambda c: tissues[c])
    figures = makeGroundTruthFigures(fake, root)
    assert any("06_sections_lesion" in str(f) for f in figures) and all(f.exists() for f in figures)
    print(json.dumps(checks, indent=1))
