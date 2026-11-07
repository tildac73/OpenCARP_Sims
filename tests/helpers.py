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
# Independent reference solves used only by the tests (the pipeline gets its lead fields from openCARP,
# FM-34). Direct sparse solves of the same discrete problem: K with sigma_i + sigma_e in tissue and
# lesion and g_bath in the bath, the corner ground at 0 mV.
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


def referenceLaplacePotential(config, tissue, ground, electrodeNodes, potentialMv=1.0):
    """u with u = potentialMv on the electrode nodes, 0 on the ground, K u = 0 elsewhere (what openCARP's Laplace run returns).

    With K factorised once on the non-ground nodes: the field of a unit current at each electrode node,
    x_j = K^-1 e_j, then the current split w (sum 1) that makes x = sum_j w_j x_j equal on all electrode
    nodes. That x is the equipotential field of a 1 uA electrode; u = potentialMv x / x_S.
    """
    free, factor = _groundedFactor(config, tissue, ground)
    count = len(tissue["positions"])
    electrode = np.unique(electrodeNodes)
    position = np.full(count, -1); position[free] = np.arange(len(free))
    rhs = np.zeros((len(free), len(electrode))); rhs[position[electrode], np.arange(len(electrode))] = 1.0
    fields = factor.solve(rhs)                                         # (free, |S|)
    M = fields[position[electrode]]                                    # M[i, j] = x_j at electrode node i
    A = np.block([[M, -np.ones((len(electrode), 1))], [np.ones((1, len(electrode))), np.zeros((1, 1))]])
    solution = np.linalg.solve(A, np.r_[np.zeros(len(electrode)), 1.0])
    weights, level = solution[:-1], solution[-1]
    u = np.zeros(count)
    u[free] = fields @ weights * potentialMv / level
    u[electrode] = potentialMv
    return u


def patchLaplace(monkeypatch, tissues=None):
    """Replace the openCARP Laplace run by referenceLaplacePotential.

    tissues: {meshBase: tissue}; a mesh not listed is read from its openCARP files.
    """
    from efov import leadfield
    from efov.io import readMesh
    tissues = dict(tissues or {})

    def fake(config, meshBase, groundVertices, electrodeNodes, nodeCount, folder, ranks=1):
        Path(folder).mkdir(parents=True, exist_ok=True)
        if str(meshBase) not in tissues:
            positions, elements, tags, fibres = readMesh(meshBase)
            tissues[str(meshBase)] = dict(positions=positions, elements=elements, tags=tags, fibres=fibres,
                                          tissueNodes=np.unique(elements[tags != 0]))
        return referenceLaplacePotential(config, tissues[str(meshBase)], groundVertices, electrodeNodes,
                                         leadfield.electrodePotentialMv(config))
    monkeypatch.setattr(leadfield, "laplacePotential", fake)


def equipotentialElectrogram(config, tissue, ground, electrodeNodes, vmOnTissue, source="intracellular"):
    """Potential of a floating equipotential electrode from K phi = -S Vm solved directly frame by frame.

    The electrode nodes are tied into one unknown (phi = P y) and carry no net current (P^T K P y = P^T f):
    independent of reciprocity and of the lead fields.
    """
    from scipy import sparse
    from scipy.sparse.linalg import splu
    K, S = _bulkAndSource(config, tissue, source)
    count = len(tissue["positions"])
    electrode, ground = np.unique(electrodeNodes), np.unique(ground)
    free = np.setdiff1d(np.arange(count), np.union1d(electrode, ground))
    columns = np.zeros(count, np.int64)
    columns[free] = np.arange(len(free))
    columns[electrode] = len(free)                                   # one shared unknown for the electrode
    keep = np.union1d(free, electrode)
    P = sparse.csr_matrix((np.ones(len(keep)), (keep, columns[keep])), shape=(count, len(free) + 1))
    solve = splu((P.T @ K @ P).tocsc()).solve
    full = np.zeros((count, vmOnTissue.shape[1]))
    full[tissue["tissueNodes"]] = vmOnTissue
    y = solve(P.T @ (-(S @ full)))
    return y[-1]
