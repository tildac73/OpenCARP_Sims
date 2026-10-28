"""P1 finite-element solve of the pseudo-bidomain extracellular problem, with lead fields (Step 6).

Strong form, active tissue (intracellular sigma_i, extracellular sigma_e) plus passive conductors
(blood bath, conducting scar, an optional outer medium; isotropic or tensor sigma_p):
    div((sigma_i + sigma_e) grad phi) = -div(sigma_i grad Vm)   in active tissue
    div(sigma_p grad phi)             = 0                       in passive regions
with current continuous between regions, no flux through the outer boundary, and phi = 0 on
grounded nodes. Weak form with linear tetrahedra:
    K phi = -B Vm,   K = sum_e vol_e G_e^T S_e G_e,   B = sum_active vol_e G_e^T Si_e G_e.
Conductivities are (E, 3) diagonals in x, y, z (fibres along x) or (E, 3, 3) tensors.

Lead field (reciprocity). The potential at an electrode sample p is phi_p = w_p^T phi, with w_p
the P1 interpolation weights. Since K is symmetric, phi_p = -z_p^T B Vm with K z_p = w_p, so each
element contributes c_e = -vol_e (G_e z_p,e) . sigma_i,e (G_e Vm_e): the bath-aware version of the
ground-truth contribution in ground_truth.py (where the lead field is the unbounded 1/r field).
The sum of the c_e equals the forward solution at p exactly (tests/test_bath.py).

Role: an independent check of openCARP's pseudo-bidomain phi_e (bath_check.py), and, once that
check passes, the bath ground truth and bath electrograms computed from openCARP's Vm
(bath_analysis.py). openCARP's own phi_e remains the reference for the electrograms
(DL-2026-10-27-05); the two agree by construction when the check passes.
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


def asTensors(conductivity):
    conductivity = np.asarray(conductivity, float)
    if conductivity.ndim == 2:
        return np.einsum("ei,ij->eij", conductivity, np.eye(3))
    return conductivity


def assemble(positions, elements, conductivity, rows=None):
    """Sparse sum_e vol_e G_e^T sigma_e G_e (N x N); sigma (E, 3) diagonal or (E, 3, 3). rows: element subset."""
    from scipy import sparse
    conductivity = asTensors(conductivity)
    if rows is not None:
        elements, conductivity = elements[rows], conductivity[rows]
    volumes, gradients = elementGradients(positions, elements)
    local = np.einsum("e,eia,eij,ejb->eab", volumes, gradients, conductivity, gradients)
    i = np.repeat(elements, 4, axis=1).ravel()
    j = np.tile(elements, (1, 4)).ravel()
    return sparse.csr_matrix((local.ravel(), (i, j)), shape=(len(positions), len(positions)))


def structuredBox(lowerMm, upperMm, spacingMm):
    """Regular lattice box, 6 tetrahedra per cube (same decomposition as quadrature_study.boxMesh), vectorised."""
    import itertools
    axes = [np.linspace(lo, hi, round((hi - lo) / spacingMm) + 1) for lo, hi in zip(lowerMm, upperMm)]
    if any(len(axis) < 2 for axis in axes):
        raise ValueError("Box needs at least one element along each axis")
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


class ExtracellularSolver:
    """Assemble once; solve phi for many Vm frames, or lead fields for many electrode samples.

    method: "lu" (sparse direct, fastest for small meshes), "amg" (pyamg-preconditioned CG, if
    installed), "ilu" (incomplete-LU-preconditioned GMRES), or "auto" (lu up to 150,000 free nodes,
    else amg if available, else ilu). Iterative solves use relative tolerance `tolerance`.
    """

    def __init__(self, positions, elements, tissueMask, sigmaIntra, sigmaExtra, sigmaBath, groundNodes,
                 method="auto", tolerance=1e-10, passiveConductivity=None):
        tissueMask = np.asarray(tissueMask, bool)
        sigmaIntra, sigmaExtra = asTensors(sigmaIntra), asTensors(sigmaExtra)
        passive = (np.broadcast_to(sigmaBath * np.eye(3), (len(elements), 3, 3)) if passiveConductivity is None
                   else asTensors(passiveConductivity))
        total = np.where(tissueMask[:, None, None], sigmaIntra + sigmaExtra, passive)
        intra = np.where(tissueMask[:, None, None], sigmaIntra, 0.0)
        self.positions, self.elements, self.tissueMask = positions, elements, tissueMask
        self.sigmaIntra = intra
        self.stiffness = assemble(positions, elements, total)
        self.source = assemble(positions, elements, intra, rows=tissueMask)
        self.ground = np.unique(np.asarray(groundNodes, int))
        if not len(self.ground):
            raise ValueError("At least one grounded node is required")
        self.free = np.setdiff1d(np.arange(len(positions)), self.ground)
        self.tolerance = tolerance
        system = self.stiffness[self.free][:, self.free].tocsr()
        if method == "auto":
            method = "lu" if len(self.free) <= 150_000 else ("amg" if _haveAmg() else "ilu")
        self.method = method
        if method == "lu":
            from scipy.sparse.linalg import splu
            self._factor = splu(system.tocsc())
            self._solve = self._factor.solve
        elif method in ("amg", "ilu"):
            from scipy.sparse.linalg import LinearOperator, cg, gmres, spilu
            if method == "amg":
                import pyamg
                preconditioner = pyamg.smoothed_aggregation_solver(system).aspreconditioner()
                krylov = cg                                     # symmetric preconditioner
            else:
                ilu = spilu(system.tocsc(), drop_tol=1e-4, fill_factor=10)
                preconditioner = LinearOperator(system.shape, ilu.solve)
                krylov = lambda A, b, **kw: gmres(A, b, restart=100, **kw)   # ILU is not symmetric: not CG

            def solveIterative(rhs):
                rhs = np.asarray(rhs, float)
                columns = rhs.reshape(len(rhs), -1)
                out = np.empty_like(columns)
                for k in range(columns.shape[1]):
                    scale = np.linalg.norm(columns[:, k])
                    if scale == 0:
                        out[:, k] = 0; continue
                    solution, info = krylov(system, columns[:, k], rtol=tolerance, maxiter=5000, M=preconditioner)
                    if info != 0:
                        raise RuntimeError(f"Iterative solve did not converge (info {info})")
                    out[:, k] = solution
                return out.reshape(rhs.shape)
            self._solve = solveIterative
        else:
            raise ValueError(f"Unknown solver method {method}")

    def solveLoad(self, load):
        """phi for a nodal load vector (N,) or (N, T); grounded nodes held at 0."""
        phi = np.zeros(np.shape(load), dtype=float)
        phi[self.free] = self._solve(np.asarray(load, float)[self.free])
        return phi

    def solve(self, vmAllNodes):
        """vmAllNodes: (N,) or (N, T) Vm on all nodes (passive-region values are ignored). Returns phi."""
        return self.solveLoad(-(self.source @ vmAllNodes))

    def leadField(self, interpolationRows):
        """Nodal z with K z = w for each row of a (P x N) interpolation matrix: (N, P)."""
        loads = np.asarray(interpolationRows.T.todense()) if hasattr(interpolationRows, "todense") else np.asarray(interpolationRows).T
        return self.solveLoad(loads)

    def elementCoefficients(self, nodalLeadField):
        """(E, 4) weights: phi_p = sum_e coeff_e . Vm_e, for one nodal lead field z (N,)."""
        volumes, gradients = elementGradients(self.positions, self.elements)
        field = -np.einsum("eij,ej->ei", gradients, nodalLeadField[self.elements])       # -grad z per element
        return np.einsum("ec,ecd,edj,e->ej", field, self.sigmaIntra, gradients, volumes)


def _haveAmg():
    try:
        import pyamg  # noqa: F401
        return True
    except ImportError:
        return False


def latticeInterpolator(lattice, points):
    """Fast P1 interpolation matrix for a structuredBox mesh: the containing cube, then its six tetrahedra.

    lattice: dict(positions, elements, lower, spacing, shape) for a mesh made by structuredBox
    (element id = permutation * cells + cell index, as constructed there).
    """
    from scipy import sparse
    positions, elements = lattice["positions"], lattice["elements"]
    shape = np.asarray(lattice["shape"]); cellsShape = shape - 1
    cellCount = int(np.prod(cellsShape))
    points = np.asarray(points, float)
    cell = np.floor((points - np.asarray(lattice["lower"])) / lattice["spacing"] + 1e-12).astype(int)
    cell = np.clip(cell, 0, cellsShape - 1)
    cellIndex = np.ravel_multi_index(cell.T, cellsShape)
    rows, cols, values = [], [], []
    for index, (point, base) in enumerate(zip(points, cellIndex)):
        for permutation in range(6):
            element = elements[permutation * cellCount + base]
            matrix = np.vstack([np.ones(4), positions[element].T])
            weights = np.linalg.solve(matrix, np.concatenate([[1.0], point]))
            if weights.min() >= -1e-9:
                rows += [index] * 4; cols += list(element); values += list(weights)
                break
        else:
            raise ValueError(f"Point {point} is outside the lattice")
    return sparse.csr_matrix((values, (rows, cols)), shape=(len(points), len(positions)))


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
