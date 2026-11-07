"""Shared test fixtures: a structured tetrahedral box and a synthetic openCARP-like run."""
import copy
import itertools
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def boxMesh(lower, upper, spacing):
    """Regular lattice box, 6 tetrahedra per cube (conforming Kuhn decomposition)."""
    axes = [np.linspace(lo, hi, round((hi - lo) / spacing) + 1) for lo, hi in zip(lower, upper)]
    grid = np.stack(np.meshgrid(*axes, indexing="ij"), axis=-1)
    shape = grid.shape[:3]
    cells = np.array(list(itertools.product(*(range(n - 1) for n in shape))))
    elements = []
    for order in itertools.permutations(range(3)):
        corner = cells.copy()
        ids = [np.ravel_multi_index(corner.T, shape)]
        for axis in order:
            corner = corner.copy(); corner[:, axis] += 1
            ids.append(np.ravel_multi_index(corner.T, shape))
        elements.append(np.stack(ids, axis=1))
    return grid.reshape(-1, 3), np.concatenate(elements)


MODEL = ROOT / "config" / "model.json"
SMALL = ROOT / "config" / "test_small.json"


def boxWithBath(size, wall, bath, spacing):
    """Kuhn-tetrahedral box: tissue |z| <= wall/2 (tag 1) inside bath (tag 0) on every side."""
    lower = [-size[0] / 2 - bath, -size[1] / 2 - bath, -wall / 2 - bath]
    upper = [size[0] / 2 + bath, size[1] / 2 + bath, wall / 2 + bath]
    positions, elements = boxMesh(lower, upper, spacing)
    centres = positions[elements].mean(axis=1)
    inside = (np.abs(centres[:, 0]) < size[0] / 2) & (np.abs(centres[:, 1]) < size[1] / 2) & (np.abs(centres[:, 2]) < wall / 2)
    return positions, elements, np.where(inside, 1, 0)


# ----------------------------------------------------------------- test oracle for the lead fields
# Independent reference solves used only by the tests (the pipeline gets its lead fields and electrograms
# from openCARP's lead-field module, FM-35). Direct sparse solves of the same discrete problem: K with
# sigma_i + sigma_e in tissue and lesion and g_bath in the bath, the corner ground at 0 mV, and openCARP's
# electrode current: 1 uA spread over the electrode nodes with the mass matrix, normalised by the
# electrode volume (sum of the lumped masses of its nodes; physics/leadfield.cc apply_unit_curr_lead).
def _bulkAndSource(config, tissue, source="intracellular"):
    from efov import leadfield
    tags, inTissue = tissue["tags"], tissue["tags"] != 0
    _, bath = leadfield.principalConductivities(config)
    bulk = np.repeat((bath * np.eye(3))[None], len(tags), axis=0)
    bulk[inTissue] = leadfield.tissueTensors(config, tags[inTissue], tissue["fibres"][inTissue], "bulk")
    sourceTensor = np.zeros_like(bulk)
    sourceTensor[inTissue] = leadfield.tissueTensors(config, tags[inTissue], tissue["fibres"][inTissue], source)
    return (leadfield.assemble(tissue["positions"], tissue["elements"], bulk),
            leadfield.assemble(tissue["positions"], tissue["elements"], sourceTensor))


def massMatrix(positions, elements):
    from scipy import sparse
    from efov.leadfield import elementGradients
    volumes, _ = elementGradients(positions, elements)
    local = volumes[:, None, None] * (np.ones((4, 4)) + np.eye(4))[None] / 20.0
    i, j = np.repeat(elements, 4, axis=1).ravel(), np.tile(elements, (1, 4)).ravel()
    return sparse.csr_matrix((local.ravel(), (i, j)), shape=(len(positions), len(positions)))


def electrodeCurrent(tissue, electrodeNodes):
    """Nodal currents (uA) of a 1 uA lead-field electrode: M chi_S / (sum of lumped masses over S)."""
    M = massMatrix(tissue["positions"], tissue["elements"])
    chi = np.zeros(len(tissue["positions"])); chi[np.unique(electrodeNodes)] = 1.0
    lumped = np.asarray(M.sum(axis=1)).ravel()
    return (M @ chi) / lumped[np.unique(electrodeNodes)].sum()


_FACTORS = {}


def _groundedFactor(config, tissue, ground):
    """LU of K on every node but the ground, cached per (mesh, ground, conductivities)."""
    from scipy.sparse.linalg import splu
    key = (id(tissue["elements"]), len(tissue["positions"]), tuple(np.unique(ground)[:5]), len(ground), str(config["par"].get("gregion[0].g_il")))
    if key not in _FACTORS:
        K, _ = _bulkAndSource(config, tissue)
        free = np.setdiff1d(np.arange(len(tissue["positions"])), np.unique(ground))
        _FACTORS.clear()
        _FACTORS[key] = (free, splu(K[free][:, free].tocsc()))
    return _FACTORS[key]


def referenceLeadField(config, tissue, ground, electrodeNodes):
    """Z (mV/uA, all nodes) of a 1 uA lead-field electrode, Z = 0 on the ground."""
    free, factor = _groundedFactor(config, tissue, ground)
    z = np.zeros(len(tissue["positions"]))
    z[free] = factor.solve(electrodeCurrent(tissue, electrodeNodes)[free])
    return z


def weightedElectrogram(config, tissue, ground, electrodeNodes, vmOnTissue, source="intracellular"):
    """Electrode reading b^T phi from K phi = -S Vm solved directly frame by frame (b: the electrode's current weights).

    Independent of reciprocity and of the lead fields.
    """
    free, factor = _groundedFactor(config, tissue, ground)
    _, S = _bulkAndSource(config, tissue, source)
    full = np.zeros((len(tissue["positions"]), vmOnTissue.shape[1]))
    full[tissue["tissueNodes"]] = vmOnTissue
    phi = np.zeros_like(full)
    phi[free] = factor.solve(-(S @ full)[free])
    return electrodeCurrent(tissue, electrodeNodes) @ phi


def patchLeadfield(monkeypatch, tissues=None):
    """Replace the openCARP lead-field run by the oracle; writes the same files (LF_<name>.igb, ECG_<name>.dat).

    tissues: {meshBase: tissue}; a mesh not listed is read from its openCARP files. Electrograms follow the
    pipeline's convention phi = -Z^T S_i Vm (what efov.electrograms checks openCARP against).
    """
    from efov import leadfield
    from efov.io import igbTimes, readIgb, readMesh, writeIgb
    tissues = dict(tissues or {})

    def fake(config, meshBase, groundVertices, leads, vmFile, folder, ecgStepMs, ranks=1, leadFieldDir=None):
        folder = Path(folder); out = folder / "POSTPROC_DIR"; out.mkdir(parents=True, exist_ok=True)
        if str(meshBase) not in tissues:
            positions, elements, tags, fibres = readMesh(meshBase)
            tissues[str(meshBase)] = dict(positions=positions, elements=elements, tags=tags, fibres=fibres,
                                          tissueNodes=np.unique(elements[tags != 0]))
        tissue = tissues[str(meshBase)]
        model = leadfield.ContributionModel(config, tissue)
        vm, times = readIgb(vmFile).astype(np.float64), igbTimes(vmFile)
        fields, traces = {}, {}
        for name, nodes in leads:
            if leadFieldDir is None:
                z = referenceLeadField(config, tissue, groundVertices, nodes)[tissue["tissueNodes"]]
                fields[name] = out / f"LF_{name}.igb"
                writeIgb(fields[name], z[None])
            else:
                z, fields[name] = readIgb(Path(leadFieldDir) / f"LF_{name}.igb")[0].astype(float), None
            traces[name] = out / f"ECG_{name}.dat"
            np.savetxt(traces[name], np.column_stack([times - times[0], vm @ model.nodalWeights(z)]), fmt="%.10g")
        return fields, traces
    monkeypatch.setattr(leadfield, "runOpenCarpLeadfield", fake)
