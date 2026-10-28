"""Independent P1 finite-element solve of the pseudo-bidomain extracellular problem (Step 6 check).

Strong form, tissue (intracellular sigma_i, extracellular sigma_e) plus bath (sigma_b):
    div((sigma_i + sigma_e) grad phi) = -div(sigma_i grad Vm)   in tissue
    div(sigma_b grad phi)             = 0                       in bath
with total current continuous between them, no flux through the outer boundary, and phi = 0 on
grounded nodes. Weak form with linear tetrahedra:
    K phi = -B Vm,   K = sum_e vol_e G_e^T S_e G_e,   B = sum_tissue vol_e G_e^T Si_e G_e
(S_e = sigma_i + sigma_e in tissue, sigma_b I in bath; Si_e = sigma_i). Conductivity tensors are
diagonal in x, y, z: this checker assumes x-aligned fibres, as the pilot does.

This is not the production route (openCARP computes phi_e, DL-2026-10-27-05). It recomputes the
same discrete problem from openCARP's own Vm to check that the bath, its conductivity, the tissue
conductivities and the ground were set up as intended.
"""
import numpy as np


def elementGradients(positions, elements):
    """Per-element volume and gradients of the four P1 basis functions: (E,), (E, 3, 4)."""
    vertices = positions[elements]
    matrices = np.concatenate([np.ones((len(elements), 4, 1)), vertices], axis=2)
    volumes = np.abs(np.linalg.det(matrices)) / 6
    if np.any(volumes <= 0):
        raise ValueError("Degenerate tetrahedron")
    gradients = np.linalg.inv(matrices)[:, 1:, :]
    return volumes, gradients


def assemble(positions, elements, diagonalConductivity, rows=None):
    """Sparse sum_e vol_e G_e^T diag(sigma_e) G_e (N x N). rows: optional element subset."""
    from scipy import sparse
    if rows is not None:
        elements, diagonalConductivity = elements[rows], diagonalConductivity[rows]
    volumes, gradients = elementGradients(positions, elements)
    local = np.einsum("e,eia,ei,eib->eab", volumes, gradients, diagonalConductivity, gradients)
    i = np.repeat(elements, 4, axis=1).ravel()
    j = np.tile(elements, (1, 4)).ravel()
    return sparse.csr_matrix((local.ravel(), (i, j)), shape=(len(positions), len(positions)))


class ExtracellularSolver:
    """Factorise once; solve phi for many Vm frames."""

    def __init__(self, positions, elements, tissueMask, sigmaIntra, sigmaExtra, sigmaBath, groundNodes):
        from scipy.sparse.linalg import splu
        tissueMask = np.asarray(tissueMask, bool)
        total = np.where(tissueMask[:, None], np.asarray(sigmaIntra) + np.asarray(sigmaExtra), sigmaBath)
        intra = np.where(tissueMask[:, None], np.asarray(sigmaIntra), 0.0)
        self.stiffness = assemble(positions, elements, total)
        self.source = assemble(positions, elements, intra, rows=tissueMask)
        self.ground = np.unique(np.asarray(groundNodes, int))
        if not len(self.ground):
            raise ValueError("At least one grounded node is required")
        self.free = np.setdiff1d(np.arange(len(positions)), self.ground)
        self.factor = splu(self.stiffness[self.free][:, self.free].tocsc())

    def solve(self, vmAllNodes):
        """vmAllNodes: (N,) or (N, T) Vm on all nodes (bath values are ignored). Returns phi (N[, T])."""
        rhs = -(self.source @ vmAllNodes)
        phi = np.zeros_like(rhs, dtype=float)
        phi[self.free] = self.factor.solve(np.asarray(rhs[self.free], dtype=float))
        return phi


def barycentricInterpolator(positions, elements, points, searchRadiusMm=1.0):
    """Sparse (P x N) matrix interpolating nodal values at points (P1, containing tetrahedron)."""
    from scipy import sparse
    centres = positions[elements].mean(axis=1)
    rows, cols, values = [], [], []
    for index, point in enumerate(np.asarray(points, float)):
        candidates = np.flatnonzero(np.linalg.norm(centres - point, axis=1) <= searchRadiusMm)
        found = False
        for element in candidates:
            vertices = positions[elements[element]]
            matrix = np.vstack([np.ones(4), vertices.T])
            weights = np.linalg.solve(matrix, np.concatenate([[1.0], point]))
            if weights.min() >= -1e-9:
                rows += [index] * 4; cols += list(elements[element]); values += list(weights)
                found = True
                break
        if not found:
            raise ValueError(f"Point {point} lies in no element within {searchRadiusMm} mm")
    return sparse.csr_matrix((values, (rows, cols)), shape=(len(points), len(positions)))
