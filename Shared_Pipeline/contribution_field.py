"""Exact contributing-tissue field of a recording, in a slab with a blood bath (ground truth, Nov 2026).

Definition (replaces FOV80 as the ground truth; DL-2026-10-28-07 superseded 1 Nov 2026).
For a recording with electrode weights w (16 values; unipole, bipole or the selected omnipolar
combination) and finite electrodes sampled at surface points, the recorded potential is the sum of
element contributions (reciprocity, exact for the discrete problem):

    phi(t) = sum_e c_e(t),      c_e(t) = coeff_e . Vm_e(t)      (fem_extracellular.elementCoefficients)

The share of element e over the measurement window is its signed part of the signal energy

    s_e = sum_t c_e(t) phi(t) / sum_t phi(t)^2,      sum_e s_e = 1 exactly.

No cut-off radius is applied: every element of the domain is kept. Cancellation is kept (s_e < 0
where an element works against the signal). The source tensor in coeff is the forward model's:
the monodomain tensor sigma_i sigma_e / (sigma_i + sigma_e) for openCARP pseudo-bidomain, sigma_i
for full bidomain (verified 1 Nov 2026 on openCARP 19.0.0; see 3D_Code/production/README.md).

Computed without forming c_e(t): phi = a^T Vm with a the node-accumulated coefficients, then
u = Vm phi and s_e = coeff_e . u_e / (phi . phi).

Outputs per recording
  * shares aggregated into cubic voxels of a fixed grid over the tissue (sums are preserved), so
    fields from different meshes can be compared voxel by voxel;
  * cumulative share against lateral distance and against depth from the recording anchor
    (descriptive curves for reading the field; not a radius).
Comparison of two fields: L1 distance sum|a - b| / sum|b| over voxels.
"""
from pathlib import Path

import numpy as np

import fem_extracellular as fem
import fibres as fibreTools

TISSUE_TAG, LESION_TAG, BATH_TAG = 1, 5, 0


# ----------------------------------------------------------------- mesh and model
def readMesh(meshBase):
    """openCARP mesh (um) -> positions (mm), tetrahedra, tags, fibre vectors."""
    meshBase = str(meshBase)
    with open(meshBase + ".pts") as stream:
        count = int(stream.readline())
        positions = np.loadtxt(stream, ndmin=2)[:count] / 1000.0
    with open(meshBase + ".elem") as stream:
        count = int(stream.readline())
        rows = np.loadtxt(stream, dtype=str, ndmin=2)[:count]
    if not np.all(rows[:, 0] == "Tt"):
        raise ValueError("Expected tetrahedra")
    with open(meshBase + ".lon") as stream:
        vectors = int(stream.readline().split()[0])
        fibreRows = np.loadtxt(stream, ndmin=2)[:count]
    return positions, rows[:, 1:5].astype(int), rows[:, 5].astype(int), fibreRows[:, :3]


def sourceTensors(source, intracellular, extracellular):
    """(source, rest) principal values such that source + rest = sigma_i + sigma_e (the elliptic operator)."""
    gi, ge = np.asarray(intracellular, float), np.asarray(extracellular, float)
    if source == "sigmaI":
        src = gi
    elif source == "sigmaM":
        src = gi * ge / (gi + ge)
    else:
        raise ValueError("source must be 'sigmaI' (bidomain) or 'sigmaM' (openCARP pseudo-bidomain)")
    return src, gi + ge - src


class BathModel:
    """Tissue + lesion + bath mesh with the extracellular solver of one forward model.

    conductivities: dict g_il.. g_en for viable tissue (S/m); lesion: near-zero value used for all six
    (insulating scar) or scar="conducting" (lesion passive with the healthy extracellular tensor;
    ASSUMPTION, no scar conductivity sourced); blood: bath conductivity (S/m).
    Ground: groundNodes (the openCARP run's ground.vtx) if given, else the nodes on the bath face
    farthest from the tissue (older slab.py geometry). Other outer walls are no-flux (natural condition).
    """

    def __init__(self, meshBase, conductivities, lesionConductivity, blood, source, scar="insulating",
                 method="auto", luMaxNodes=400_000, vmTissuePositions=None, groundNodes=None):
        positions, elements, tags, fibres = readMesh(meshBase)
        self.positions, self.elements, self.tags = positions, elements, tags
        tissue = tags != BATH_TAG
        lesion = tags == LESION_TAG
        self.tissueNodes = np.unique(elements[tissue])                    # = openCARP vm.igb order
        self.centres = positions[elements].mean(axis=1)
        frames = np.zeros((len(elements), 3, 3))
        frames[tissue] = fibreTools.localFrames(fibres[tissue])
        healthyI = [conductivities[f"g_i{a}"] for a in "ltn"]
        healthyE = [conductivities[f"g_e{a}"] for a in "ltn"]
        src, rest = sourceTensors(source, healthyI, healthyE)
        lsrc, lrest = sourceTensors(source, [lesionConductivity] * 3, [lesionConductivity] * 3)
        principalSrc = np.zeros((len(elements), 3)); principalRest = np.zeros((len(elements), 3))
        principalSrc[tissue] = src; principalRest[tissue] = rest
        active = tissue.copy()
        passive = np.broadcast_to(blood * np.eye(3), (len(elements), 3, 3)).copy()
        if scar == "insulating":
            principalSrc[lesion] = lsrc; principalRest[lesion] = lrest
        elif scar == "conducting":
            active &= ~lesion
            passive[lesion] = fibreTools.tensors(np.tile(healthyE, (lesion.sum(), 1)), frames[lesion])
        else:
            raise ValueError("scar must be 'insulating' or 'conducting'")
        sigmaSrc = np.zeros((len(elements), 3, 3)); sigmaRest = np.zeros((len(elements), 3, 3))
        sigmaSrc[active] = fibreTools.tensors(principalSrc[active], frames[active])
        sigmaRest[active] = fibreTools.tensors(principalRest[active], frames[active])
        if groundNodes is None:
            self.ground = np.flatnonzero(np.isclose(positions[:, 2], positions[:, 2].max()))
        else:
            self.ground = np.unique(np.asarray(groundNodes, int))
            if not len(self.ground):
                raise ValueError("groundNodes is empty")
        if method == "auto":
            method = "lu" if len(positions) <= luMaxNodes else ("amg" if fem._haveAmg() else "ilu")
        self.solver = fem.ExtracellularSolver(positions, elements, active, sigmaSrc, sigmaRest, blood,
                                              self.ground, method=method, passiveConductivity=passive)
        self.source, self.scar, self.method = source, scar, method
        self.tissueTop = float(positions[self.tissueNodes, 2].max())
        # Vm recorded on another mesh of the same tissue (e.g. the run's own bath thickness):
        # map its tissue nodes onto this mesh by position.
        self.vmOrder = None
        if vmTissuePositions is not None:
            key = lambda xyz: [tuple(k) for k in np.round(np.asarray(xyz) * 1000).astype(np.int64)]
            lookup = {k: i for i, k in enumerate(key(vmTissuePositions))}
            try:
                self.vmOrder = np.array([lookup[k] for k in key(positions[self.tissueNodes])])
            except KeyError as error:
                raise ValueError("Tissue nodes of this mesh do not match the Vm mesh") from error

    def fullVm(self, tissueVm):
        """(tissue nodes, T) Vm in vm.igb order -> (all nodes, T), zero in the bath."""
        tissueVm = np.asarray(tissueVm, float)
        if tissueVm.shape[0] != len(self.tissueNodes):
            raise ValueError(f"Vm has {tissueVm.shape[0]} nodes, the mesh has {len(self.tissueNodes)} tissue nodes")
        out = np.zeros((len(self.positions),) + tissueVm.shape[1:])
        out[self.tissueNodes] = tissueVm if self.vmOrder is None else tissueVm[self.vmOrder]
        return out

    def toVmOrder(self, allNodes):
        """Adjoint of fullVm: (all nodes,) -> (tissue nodes,) in vm.igb order."""
        values = np.asarray(allNodes, float)[self.tissueNodes]
        if self.vmOrder is None:
            return values
        out = np.empty_like(values)
        out[self.vmOrder] = values
        return out


# ----------------------------------------------------------------- electrode transfer (reciprocity)
_TRANSFER = {}


def _transferRow(index):
    model, rows = _TRANSFER["model"], _TRANSFER["rows"]
    z = model.solver.solveLoad(np.asarray(rows[index].todense()).ravel())   # K z = w (grounded, K symmetric)
    return model.toVmOrder(-(model.solver.source.T @ z))


def electrodeTransfer(model, rows, processes=1):
    """(U, tissue nodes) matrix T with phi_u(t) = T[u] . Vm(t) (Vm in vm.igb order), for interpolation rows (U x N).

    Same linear algebra as solving phi = K^-1 (-S Vm) every frame and interpolating, rearranged by
    reciprocity (K symmetric): w^T phi = -(S^T K^-1 w)^T Vm. One solve per electrode instead of one per
    output frame; the electrode solves are independent and run in `processes` forked workers (Linux).
    """
    from scipy import sparse
    rows = sparse.csr_matrix(rows)
    _TRANSFER.update(model=model, rows=rows)
    try:
        if processes > 1:
            import multiprocessing
            with multiprocessing.get_context("fork").Pool(processes) as pool:
                result = pool.map(_transferRow, range(rows.shape[0]), chunksize=1)
        else:
            result = [_transferRow(k) for k in range(rows.shape[0])]
    finally:
        _TRANSFER.clear()
    return np.asarray(result)


# ----------------------------------------------------------------- interpolation
def pointInterpolator(positions, elements, points, neighbours=24):
    """Sparse (P x N) P1 interpolation at points (containing tetrahedron), via a k-d tree of element centres."""
    from scipy import sparse
    from scipy.spatial import cKDTree
    points = np.asarray(points, float)
    centres = positions[elements].mean(axis=1)
    _, candidates = cKDTree(centres).query(points, k=min(neighbours, len(elements)))
    rows, cols, values = [], [], []
    for index, (point, near) in enumerate(zip(points, np.atleast_2d(candidates))):
        for element in near:
            vertices = positions[elements[element]]
            matrix = np.vstack([np.ones(4), vertices.T])
            weights = np.linalg.solve(matrix, np.concatenate([[1.0], point]))
            if weights.min() >= -1e-9:
                rows += [index] * 4; cols += list(elements[element]); values += list(weights)
                break
        else:
            raise ValueError(f"Point {point} lies in none of its {len(near)} nearest elements (outside the mesh?)")
    return sparse.csr_matrix((values, (rows, cols)), shape=(len(points), len(positions)))


def recordingRow(model, electrodeSamples, weights):
    """Interpolation row of a recording: electrode weights x mean over each electrode's surface samples."""
    weights = np.asarray(weights, float)
    used = np.flatnonzero(np.abs(weights) > 1e-12)
    if not len(used):
        raise ValueError("Recording has no non-zero electrode weight")
    row = None
    for electrode in used:
        part = weights[electrode] * pointInterpolator(model.positions, model.elements, electrodeSamples[electrode]).mean(axis=0)
        row = part if row is None else row + part
    return np.asarray(row).ravel()


# ----------------------------------------------------------------- shares
def shares(model, row, tissueVm):
    """Element energy shares s_e, the reconstructed signal phi(t), and the sum check."""
    z = model.solver.leadField(row[None, :])[:, 0]
    coefficients = model.solver.elementCoefficients(z)                    # (E, 4); zero outside active tissue
    accumulated = np.zeros(len(model.positions))
    np.add.at(accumulated, model.elements, coefficients)
    vm = model.fullVm(tissueVm)
    phi = accumulated @ vm                                                # (T,)
    energy = float(phi @ phi)
    if energy <= 0:
        raise ValueError("Recording has zero signal energy in the window")
    u = vm @ phi                                                          # (N,)
    elementShares = np.einsum("ej,ej->e", coefficients, u[model.elements]) / energy
    return elementShares, phi, float(elementShares.sum())


def voxelGrid(tissueSizeMm, voxelMm):
    size = np.asarray(tissueSizeMm, float)
    counts = np.round(size / voxelMm).astype(int)
    if not np.allclose(counts * voxelMm, size):
        raise ValueError("Tissue size must be a multiple of the voxel size")
    return dict(lower=-size / 2, voxelMm=float(voxelMm), shape=tuple(int(c) for c in counts))


def voxelise(model, elementShares, grid):
    """Sum element shares into voxels by element centre (tissue elements only; sum preserved)."""
    tissue = model.tags != BATH_TAG
    index = np.floor((model.centres[tissue] - grid["lower"]) / grid["voxelMm"] + 1e-9).astype(int)
    index = np.clip(index, 0, np.asarray(grid["shape"]) - 1)
    flat = np.ravel_multi_index(index.T, grid["shape"])
    field = np.bincount(flat, weights=elementShares[tissue], minlength=int(np.prod(grid["shape"])))
    return field.reshape(grid["shape"])


def curves(model, elementShares, anchor, stepMm):
    """Cumulative share within lateral distance d and within depth d of the anchor (descriptive)."""
    tissue = model.tags != BATH_TAG
    centres, values = model.centres[tissue], elementShares[tissue]
    lateral = np.linalg.norm(centres[:, :2] - np.asarray(anchor)[:2], axis=1)
    depth = np.clip(model.tissueTop - centres[:, 2], 0, None)
    out = {}
    for name, key in [("lateral", lateral), ("depth", depth)]:
        edges = np.arange(0, key.max() + stepMm, stepMm)
        cumulative = np.array([values[key <= edge].sum() for edge in edges])
        out[name] = dict(distanceMm=edges, cumulativeShare=cumulative)
    return out


def fieldDistance(fieldA, fieldB):
    """L1 distance between two voxel share fields, relative to the second (reference) field."""
    fieldA, fieldB = np.asarray(fieldA, float), np.asarray(fieldB, float)
    if fieldA.shape != fieldB.shape:
        raise ValueError("Fields are on different voxel grids")
    return float(np.abs(fieldA - fieldB).sum() / np.abs(fieldB).sum())
