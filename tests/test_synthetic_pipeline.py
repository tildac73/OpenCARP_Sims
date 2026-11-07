"""The analysis stages on a synthetic openCARP-format run (no openCARP needed).

The small test mesh (config/test_small.json) is built with efov.meshbuild, with the lesion strip
and isolated tissue beyond it (FM-24); Vm is an analytic planar wave (four beats, x direction; no
activation at y < 0 in the lesion condition). The stages window -> leadFields -> electrograms ->
measurements -> export -> groundTruth -> figures then run on these files, with the openCARP Laplace
runs replaced by the test oracle (helpers.patchLaplace). One electrode's electrogram is checked
against a direct frame-by-frame solve of K phi = -S_i Vm with the electrode as a floating
equipotential (independent of reciprocity and of the lead fields).
"""
import json

import numpy as np
import pytest

from efov import config as cfg, electrograms, export, groundtruth, leadfield, measurements, meshbuild, physiology, simulation
from efov.io import readVertices, saveJson, writeMesh, writeVertices
from helpers import SMALL, equipotentialElectrogram, patchLaplace


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


def buildMesh(config, lesion):
    """Kuhn-tetrahedral slab with bath on every side, run through efov.meshbuild (no-flux epicardium, fibres)."""
    import itertools
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
    boundary, width = config["fixed"]["tissue"]["boundaryYmm"], config["fixed"]["tissue"]["lesionWidthMm"]
    isLesion = lesion & (centres[:, 1] < boundary) & (centres[:, 1] > boundary - width)
    tags = np.where(inside, np.where(isLesion, 5, 1), 0)
    return meshbuild.runMesh(positions, elements, tags, size[2], "none", config["values"]["fibres"])


def writeCondition(config, job, lesion):
    """Mesh, Vm and activation times of one synthetic condition in openCARP's file layout."""
    job.mkdir(parents=True)
    positions, elements, tags, fibres = buildMesh(config, lesion)
    meshBase = writeMesh(job / "mesh" / "slab", positions, elements, tags, fibres)
    writeVertices(job / "ground.vtx", meshbuild.groundNodes(positions, elements, tags, config["fixed"]["bath"]["groundCubeMm"]))
    tissueNodes = np.unique(elements[tags != 0])
    p = positions[tissueNodes]
    active = np.isin(tissueNodes, np.unique(elements[tags == 1]))
    if lesion:
        active &= p[:, 1] >= config["fixed"]["tissue"]["boundaryYmm"]          # isolated tissue never activates
    size = cfg.slabSizeMm(config)
    cycle, beats, step = cfg.cycleLengthMs(config), cfg.beatCount(config), cfg.sampleStepMs(config)
    times = np.round(np.arange(0, beats * cycle + 1e-9, step), 6)
    arrivals = [b * cycle + 5 + (p[:, 0] + size[0] / 2) / 0.9 for b in range(beats)]
    vm = np.full((len(times), len(tissueNodes)), -80.0)
    for arrival in arrivals:
        vm[:, active] = np.maximum(vm[:, active], actionPotential(times[:, None], arrival[None, active]))
    writeIgb(job / "vm.igb", vm, 0.0, step)
    rows = [(n, a[n]) for a in arrivals for n in np.flatnonzero(active)]
    np.savetxt(job / "act-thresh.dat", np.array(rows), fmt=["%d", "%.6f"], delimiter="\t")
    saveJson(job / "runRecord.json", dict(meshBase=str(meshBase), nodeSets=dict(nodes=len(positions))))


def directElectrogram(config, tissue, ground, nodes, job, timesMs):
    vm, _ = electrograms.readVmFrames(job, timesMs[0], timesMs[-1])
    return equipotentialElectrogram(config, tissue, ground, nodes, vm.T)


@pytest.fixture(scope="module")
def syntheticRun(tmp_path_factory):
    config = cfg.loadConfig(SMALL)
    root = tmp_path_factory.mktemp("synthetic")
    for condition in ("intact", "lesion"):
        writeCondition(config, root / "simulations" / condition, condition == "lesion")
    patch = pytest.MonkeyPatch()
    patchLaplace(patch)
    yield config, root
    patch.undo()


def test_analysis_stages_agree_on_a_synthetic_run(syntheticRun):
    import pandas as pd
    config, root = syntheticRun
    jobs = {c: root / "simulations" / c for c in ("intact", "lesion")}
    tissues = {c: simulation.tissueModel(j) for c, j in jobs.items()}
    window = physiology.deriveWindow(config, jobs["intact"], tissues["intact"])
    windows = window["windowsMs"]
    assert len(windows) == len(cfg.centreYmm(config)) * len(config["values"]["gapsMm"]) and len({tuple(w) for w in windows}) == 1
    leadFolder = root / "leadFields"; leadFolder.mkdir()
    fields = electrograms.runLeadFields(config, jobs, tissues, root / "leadFieldCache", leadFolder, log=lambda *_: None)
    assert fields["conditions"]["lesion"]["computedNow"] == fields["conditions"]["lesion"]["electrodes"]
    egms = root / "electrograms"; egms.mkdir()
    info = electrograms.runElectrograms(config, jobs, tissues, windows, leadFolder, egms)
    assert info["lesion"]["nodesPerElectrodeMin"] > 0
    # Reciprocity against a direct solve, at one electrode over the lesion (code check, Python only)
    archive = np.load(egms / "lesion.npz")
    sets = electrograms.nodeSetsFromArchive(archive)
    k = int(np.argmin(np.linalg.norm(archive["uniqueCentres"][:, :2] - [0.0, -2.0], axis=1)))
    p, e = np.argwhere(archive["electrodeIndex"] == k)[0]
    frames = slice(len(archive["timeMs"]) - 300, len(archive["timeMs"]))
    direct = directElectrogram(config, tissues["lesion"], readVertices(jobs["lesion"] / "ground.vtx"), sets[k], jobs["lesion"], archive["timeMs"][frames])
    np.testing.assert_allclose(archive["uni"][p, e, frames], direct, rtol=1e-4, atol=1e-5 * np.abs(direct).max())
    folder = root / "measurements"; folder.mkdir()
    points, amplitudes, status = measurements.runMeasurements(config, egms, windows, folder)
    assert status["filterHistory"] and amplitudes["wangBipolarMedianMv"] == 1.44
    datasets = export.runExport(config, folder, root / "export")
    assert len(datasets) == len(config["values"]["gapsMm"]) * (config["fixed"]["analysis"]["noise"]["repeats"] + 1) * 2
    gt = root / "groundTruth"; gt.mkdir()
    checks = groundtruth.runGroundTruth(config, jobs, tissues, egms, folder, windows, gt)
    assert checks["passed"], checks
    summary = pd.read_csv(gt / "summary.csv")
    assert set(summary.kind) == {"uni", "bi", "omni"} and (gt / "shares_lesion_gap0.25mm.vtk").exists()
    # the ground-truth signal is the measured clean signal (same lead fields, same chain)
    clean = np.load(folder / "lesionTraces.npz")["cleanMv"]
    lesionPoints = pd.read_csv(folder / "lesionPoints.csv").set_index("recordingId")
    row = summary[summary.condition == "lesion"].iloc[0]
    assert np.isclose(row.signalPpMv, np.ptp(clean[int(lesionPoints.loc[row.recordingId].traceIndex)]), rtol=1e-4)
    # Ground-truth figures from the same outputs (a minimal stand-in for pipeline.Run)
    from types import SimpleNamespace
    from efov.groundtruth_figures import makeGroundTruthFigures
    fake = SimpleNamespace(out=root, config=config, tissue=lambda c: tissues[c])
    figures = makeGroundTruthFigures(fake, root)
    assert any("06_sections_lesion" in str(f) for f in figures) and all(f.exists() for f in figures)
    print(json.dumps(checks, indent=1))
