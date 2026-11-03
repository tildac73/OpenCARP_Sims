"""Independent finite-element model of the extracellular field (tissue + lesion + bath).

Used for two things, both on exactly the discrete problem openCARP solves:
  1. checking every run: openCARP's pseudo-bidomain phi_e must equal this solve (electrograms stage);
  2. the ground truth: lead fields by reciprocity give each element's exact contribution.

Strong form (P1 tetrahedra, no flux through the outer boundary, phi = 0 on the ground nodes):
    div((sigma_i + sigma_e) grad phi) = -div(sigma_src grad Vm)   in active tissue
    div(sigma_blood grad phi)        = 0                          in the bath
Weak form: K phi = -S Vm. openCARP pseudo-bidomain uses sigma_src = sigma_m = sigma_i sigma_e /
(sigma_i + sigma_e) (verified on openCARP 19.0.0: residual 0.0, VL-2026-11-01-02).

Reciprocity: for an electrode row w (interpolation weights), w^T phi = -z^T S Vm with K z = w (K
symmetric), so element e contributes c_e(t) = coeff_e . Vm_e(t).
"""
import numpy as np

TISSUE_TAG, LESION_TAG, BATH_TAG = 1, 5, 0


# ----------------------------------------------------------------- finite elements
def elementGradients(positions, elements):
    """Per element: volume (E,) and gradients of the four P1 basis functions (E, 3, 4)."""
    matrices = np.concatenate([np.ones((len(elements), 4, 1)), positions[elements]], axis=2)
    volumes = np.abs(np.linalg.det(matrices)) / 6
    if np.any(volumes <= 0):
        raise ValueError("Degenerate tetrahedron")
    return volumes, np.linalg.inv(matrices)[:, 1:, :]


def assemble(positions, elements, conductivity, rows=None):
    """Sparse sum_e vol_e G_e^T sigma_e G_e (N x N); sigma (E, 3, 3); rows = element subset."""
    from scipy import sparse
    if rows is not None:
        elements, conductivity = elements[rows], conductivity[rows]
    volumes, gradients = elementGradients(positions, elements)
    local = np.einsum("e,eia,eij,ejb->eab", volumes, gradients, conductivity, gradients)
    i = np.repeat(elements, 4, axis=1).ravel()
    j = np.tile(elements, (1, 4)).ravel()
    return sparse.csr_matrix((local.ravel(), (i, j)), shape=(len(positions), len(positions)))


def localFrames(fibres):
    """(E, 3, 3) rows = unit fibre, sheet, normal (right-handed completion; t = n so the sheet choice is free)."""
    f = np.asarray(fibres, float)
    length = np.linalg.norm(f, axis=1)
    if np.any(length < 1e-12):
        raise ValueError("Zero fibre vector in tissue")
    f = f / length[:, None]
    helper = np.where(np.abs(f[:, 2:3]) < 0.9, [[0.0, 0.0, 1.0]], [[1.0, 0.0, 0.0]])
    s = np.cross(helper, f)
    s /= np.linalg.norm(s, axis=1, keepdims=True)
    return np.stack([f, s, np.cross(f, s)], axis=1)


def tensors(principal, frames):
    """sum_k principal[:, k] a_k a_k^T in mesh axes: (E, 3, 3)."""
    return np.einsum("eki,ek,ekj->eij", frames, np.asarray(principal, float), frames)


def haveAmg():
    try:
        import pyamg  # noqa: F401
        return True
    except ImportError:
        return False


class ExtracellularSolver:
    """K phi = -S Vm, assembled once; solved for many Vm frames or many lead-field loads.

    method "auto": sparse LU up to 150,000 free nodes, else pyamg-preconditioned CG (pip install pyamg),
    else ILU-preconditioned GMRES. Iterative solves use relative tolerance `tolerance`.
    """

    def __init__(self, positions, elements, sigmaTotal, sigmaSource, sourceElements, groundNodes,
                 method="auto", tolerance=1e-10):
        self.positions, self.elements = positions, elements
        self.sigmaSource = np.where(sourceElements[:, None, None], sigmaSource, 0.0)
        self.stiffness = assemble(positions, elements, sigmaTotal)
        self.source = assemble(positions, elements, self.sigmaSource, rows=sourceElements)
        self.ground = np.unique(np.asarray(groundNodes, int))
        if not len(self.ground):
            raise ValueError("At least one grounded node is required")
        self.free = np.setdiff1d(np.arange(len(positions)), self.ground)
        system = self.stiffness[self.free][:, self.free].tocsr()
        if method == "auto":
            method = "lu" if len(self.free) <= 150_000 else ("amg" if haveAmg() else "ilu")
        self.method = method
        if method == "lu":
            from scipy.sparse.linalg import splu
            self._solve = splu(system.tocsc()).solve
            return
        from scipy.sparse.linalg import LinearOperator, cg, gmres, spilu
        if method == "amg":
            import pyamg
            preconditioner, krylov = pyamg.smoothed_aggregation_solver(system).aspreconditioner(), cg
        elif method == "ilu":
            ilu = spilu(system.tocsc(), drop_tol=1e-4, fill_factor=10)
            preconditioner = LinearOperator(system.shape, ilu.solve)
            krylov = lambda A, b, **kw: gmres(A, b, restart=100, **kw)   # ILU is not symmetric
        else:
            raise ValueError(f"Unknown solver method {method}")

        def solveIterative(rhs):
            columns = np.asarray(rhs, float).reshape(len(rhs), -1)
            out = np.zeros_like(columns)
            for k in range(columns.shape[1]):
                if np.linalg.norm(columns[:, k]) == 0:
                    continue
                out[:, k], info = krylov(system, columns[:, k], rtol=tolerance, maxiter=5000, M=preconditioner)
                if info != 0:
                    raise RuntimeError(f"Iterative extracellular solve did not converge (info {info})")
            return out.reshape(np.shape(rhs))
        self._solve = solveIterative

    def solveLoad(self, load):
        phi = np.zeros(np.shape(load))
        phi[self.free] = self._solve(np.asarray(load, float)[self.free])
        return phi

    def solve(self, vmAllNodes):
        """phi (N,) or (N, T) for Vm on all nodes (bath values ignored)."""
        return self.solveLoad(-(self.source @ vmAllNodes))

    def leadField(self, interpolationRows):
        """Nodal z with K z = w for each row of a (P x N) interpolation matrix: (N, P)."""
        rows = interpolationRows.toarray() if hasattr(interpolationRows, "toarray") else np.asarray(interpolationRows)
        return self.solveLoad(np.atleast_2d(rows).T)

    def elementCoefficients(self, nodalLeadField):
        """(E, 4): phi_p = sum_e coeff_e . Vm_e for one lead field z (N,)."""
        volumes, gradients = elementGradients(self.positions, self.elements)
        minusGradZ = -np.einsum("eij,ej->ei", gradients, nodalLeadField[self.elements])
        return np.einsum("ec,ecd,edj,e->ej", minusGradZ, self.sigmaSource, gradients, volumes)


class BathModel:
    """The openCARP run's mesh with its conductivities, sigma_m source and ground: the pseudo-bidomain field."""

    def __init__(self, tissue, conductivities, lesionConductivity, bloodSPerM, groundNodes, method="auto"):
        positions, elements, tags, fibres = tissue["positions"], tissue["elements"], tissue["tags"], tissue["fibres"]
        isTissue = tags != BATH_TAG
        gi = np.array([conductivities[f"g_i{a}"] for a in "ltn"])
        ge = np.array([conductivities[f"g_e{a}"] for a in "ltn"])
        principalI = np.zeros((len(elements), 3)); principalE = np.zeros((len(elements), 3))
        principalI[isTissue], principalE[isTissue] = gi, ge
        lesion = tags == LESION_TAG
        principalI[lesion] = principalE[lesion] = lesionConductivity              # insulating scar
        frames = np.zeros((len(elements), 3, 3))
        frames[isTissue] = localFrames(fibres[isTissue])
        sigmaI, sigmaE = tensors(principalI, frames), tensors(principalE, frames)
        sigmaM = tensors(np.divide(principalI * principalE, principalI + principalE,
                                   out=np.zeros_like(principalI), where=isTissue[:, None]), frames)
        total = np.where(isTissue[:, None, None], sigmaI + sigmaE, bloodSPerM * np.eye(3))
        self.positions, self.elements, self.tags = positions, elements, tags
        self.tissueNodes = tissue["tissueNodes"]
        self.centres = positions[elements].mean(axis=1)
        self.tissueTopMm = float(positions[self.tissueNodes, 2].max())
        self.solver = ExtracellularSolver(positions, elements, total, sigmaM, isTissue, groundNodes, method)

    def fullVm(self, tissueVm):
        """(tissue nodes, ...) in vm.igb order -> (all nodes, ...), zero in the bath."""
        tissueVm = np.asarray(tissueVm, float)
        if tissueVm.shape[0] != len(self.tissueNodes):
            raise ValueError("Vm node count differs from the mesh's tissue nodes")
        out = np.zeros((len(self.positions),) + tissueVm.shape[1:])
        out[self.tissueNodes] = tissueVm
        return out


def pointInterpolator(positions, elements, points, neighbours=24):
    """Sparse (P x N) P1 interpolation at points, from the containing tetrahedron (k-d tree of element centres)."""
    from scipy import sparse
    from scipy.spatial import cKDTree
    points = np.asarray(points, float)
    _, candidates = cKDTree(positions[elements].mean(axis=1)).query(points, k=min(neighbours, len(elements)))
    rows, cols, values = [], [], []
    for index, (point, near) in enumerate(zip(points, np.atleast_2d(candidates))):
        for element in near:
            matrix = np.vstack([np.ones(4), positions[elements[element]].T])
            weights = np.linalg.solve(matrix, np.concatenate([[1.0], point]))
            if weights.min() >= -1e-9:
                rows += [index] * 4; cols += list(elements[element]); values += list(weights)
                break
        else:
            raise ValueError(f"Point {point} lies in none of its {len(near)} nearest elements (outside the mesh?)")
    return sparse.csr_matrix((values, (rows, cols)), shape=(len(points), len(positions)))
