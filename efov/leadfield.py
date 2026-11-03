"""Lead fields of the electrodes by reciprocity, and each element's contribution coefficients (FM-14).

openCARP route (fixed.groundTruth.leadFieldMethod = "openCARP"), one Laplace solve per electrode:
  * config/leadfield.par (experiment 2) plus run lines: the run mesh, the run's ground vertices,
    the electrode's node set as an extracellular current stimulus with a total current of 1 uA, and
    conductivity regions whose extracellular values are g_i + g_e of the run (tissue and lesion) and
    g_bath, so the Laplace operator is div((sigma_i + sigma_e) grad) - the operator of openCARP's
    pseudo-bidomain phi_e.
  * The potential at every node is the lead field z (mV per uA).
Python route ("python", the pre-declared fallback of FM-14): the same discrete problem, assembled
here with P1 elements and solved with sparse LU or pyamg-preconditioned CG (verified earlier against
closed-form solutions, VL-2026-10-28-07, and against openCARP, VL-2026-11-02-01).

Reciprocity (both routes): an electrode with node set S records phi = mean over S of phi_e. With
K phi = -S_m Vm (S_m assembled with sigma_m) and K z = w (w = 1/|S| on S), phi = -z^T S_m Vm, so
element e contributes c_e(t) = coeff_e . Vm_e(t), coeff_e = -vol_e (grad N)^T sigma_m,e grad z.
Units: z in mV/uA, positions in mm, sigma in S/m and Vm in mV give c in mV (the factors cancel:
1e3 [V/A per mV/uA] x 1e6 [two gradients] x 1e-9 [mm^3] x 1e-3 [V per mV] x 1e3 [mV per V] = 1).
"""
import os
from pathlib import Path
import subprocess

import numpy as np

from .config import CONDUCTIVITY_NAMES, par
from .io import igbHeader, readIgb, saveJson, writeVertices
from .parfile import writeResolved

BATH_TAG, TISSUE_TAG, LESION_TAG = 0, 1, 5


# ----------------------------------------------------------------- conductivities
def principalConductivities(config):
    """Principal (fibre, sheet, normal) values per region: {'tissue'|'lesion': (gi, ge)}, and g_bath (S/m)."""
    out = {}
    for name, index in (("tissue", 0), ("lesion", 1)):
        gi = np.array([par(config, f"gregion[{index}].g_i{a}") for a in "ltn"])
        ge = np.array([par(config, f"gregion[{index}].g_e{a}") for a in "ltn"])
        out[name] = (gi, ge)
    return out, par(config, "gregion[2].g_bath")


def laplaceRegionLines(config):
    """Conductivity regions of the Laplace run: extracellular = g_i + g_e (tissue, lesion), g_bath."""
    regions, bath = principalConductivities(config)
    lines = {"num_gregions": 3}
    for index, (name, tag) in enumerate((("tissue", TISSUE_TAG), ("lesion", LESION_TAG))):
        gi, ge = regions[name]
        total = gi + ge
        lines.update({f"gregion[{index}].num_IDs": 1, f"gregion[{index}].ID[0]": tag})
        for axis, value in zip("ltn", total):
            lines[f"gregion[{index}].g_i{axis}"] = float(value)     # not used by a Laplace solve
            lines[f"gregion[{index}].g_e{axis}"] = float(value)
    lines.update({"gregion[2].num_IDs": 1, "gregion[2].ID[0]": BATH_TAG, "gregion[2].g_bath": float(bath)})
    return lines


# ----------------------------------------------------------------- openCARP route
def laplaceOutput(folder, nodeCount):
    """The potential written by the Laplace run (last frame); the file name is checked, not assumed."""
    candidates = [Path(folder) / name for name in ("phie.igb", "phi.igb", "laplace.igb")]
    candidates += sorted(p for p in Path(folder).glob("*.igb") if p not in candidates and p.name != "vm.igb")
    for path in candidates:
        if path.exists() and igbHeader(path)[1] == nodeCount:
            return readIgb(path, slice(-1, None))[0].astype(float), path.name
    found = sorted(p.name for p in Path(folder).glob("*"))
    raise RuntimeError(f"No whole-mesh potential found after the Laplace run in {folder}: {found}")


def openCarpLeadField(config, meshBase, groundVertices, electrodeNodes, nodeCount, folder, processCount=1):
    """Lead field (mV per uA, all nodes) of one electrode from an openCARP Laplace solve."""
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    writeVertices(folder / "electrode.vtx", electrodeNodes)
    writeVertices(folder / "ground.vtx", groundVertices)
    lines = {"meshname": str(meshBase), "simID": str(folder),
             "stim[0].elec.vtx_file": str(folder / "ground"), "stim[1].elec.vtx_file": str(folder / "electrode"),
             **laplaceRegionLines(config)}
    parPath = writeResolved(config["leadFieldPar"], {}, lines, folder / "leadfield.par",
                            header="# Resolved Laplace lead-field parameters (efov; FM-14)")
    from .simulation import findExecutable, runLogged
    command = [findExecutable("openCARP", "OPENCARP_BIN"), "+F", str(parPath)]
    if processCount > 1:
        command = [os.environ.get("MPIEXEC_BIN", "mpiexec"), "-n", str(processCount)] + command
    if runLogged(command, folder / "laplace.log") != 0:
        raise RuntimeError(f"openCARP Laplace run failed; see {folder / 'laplace.log'}")
    z, name = laplaceOutput(folder, nodeCount)
    saveJson(folder / "leadfield.json", dict(output=name, electrodeNodes=len(electrodeNodes), currentUA=1.0))
    return z


# ----------------------------------------------------------------- finite elements (shared)
def elementGradients(positions, elements):
    """Per element: volume (E,) and gradients of the four P1 basis functions (E, 3, 4)."""
    matrices = np.concatenate([np.ones((len(elements), 4, 1)), positions[elements]], axis=2)
    volumes = np.abs(np.linalg.det(matrices)) / 6
    if np.any(volumes <= 0):
        raise ValueError("Degenerate tetrahedron")
    return volumes, np.linalg.inv(matrices)[:, 1:, :]


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


def tissueTensors(config, tags, fibres, which):
    """Per-element tensors on tissue and lesion elements: which = 'monodomain' (sigma_m) or 'bulk' (sigma_i + sigma_e)."""
    regions, _ = principalConductivities(config)
    principal = np.zeros((len(tags), 3))
    for name, tag in (("tissue", TISSUE_TAG), ("lesion", LESION_TAG)):
        gi, ge = regions[name]
        principal[tags == tag] = gi * ge / (gi + ge) if which == "monodomain" else gi + ge
    return tensors(principal, localFrames(fibres))


class ContributionModel:
    """Tissue elements of a run and their sigma_m tensors: turns a lead field into contribution coefficients."""

    def __init__(self, config, tissue):
        tissueElements = tissue["tags"] != BATH_TAG
        self.elements = tissue["elements"][tissueElements]
        self.tags = tissue["tags"][tissueElements]
        self.positions = tissue["positions"]
        self.centres = self.positions[self.elements].mean(axis=1)
        self.volumes, self.gradients = elementGradients(self.positions, self.elements)
        self.sigmaM = tissueTensors(config, self.tags, tissue["fibres"][tissueElements], "monodomain")
        self.tissueNodes = tissue["tissueNodes"]
        local = np.full(len(self.positions), -1, np.int64)
        local[self.tissueNodes] = np.arange(len(self.tissueNodes))
        self.localElements = local[self.elements]                 # element corners as indices into vm.igb columns

    def coefficients(self, leadFieldOnNodes):
        """(E_tissue, 4): phi = sum_e coeff_e . Vm_e for a lead field given on all mesh nodes (mV/uA)."""
        minusGradZ = -np.einsum("eij,ej->ei", self.gradients, leadFieldOnNodes[self.elements])
        return np.einsum("ec,ecd,edj,e->ej", minusGradZ, self.sigmaM, self.gradients, self.volumes)


# ----------------------------------------------------------------- Python route (fallback)
def assemble(positions, elements, conductivity):
    """Sparse sum_e vol_e G_e^T sigma_e G_e (N x N); sigma (E, 3, 3)."""
    from scipy import sparse
    volumes, gradients = elementGradients(positions, elements)
    local = np.einsum("e,eia,eij,ejb->eab", volumes, gradients, conductivity, gradients)
    i = np.repeat(elements, 4, axis=1).ravel()
    j = np.tile(elements, (1, 4)).ravel()
    return sparse.csr_matrix((local.ravel(), (i, j)), shape=(len(positions), len(positions)))


class PythonLeadFieldSolver:
    """K z = w on the run mesh: (sigma_i + sigma_e) in tissue and lesion, g_bath in the bath, z = 0 on the ground."""

    def __init__(self, config, tissue, groundVertices, tolerance=1e-10):
        from scipy.sparse.linalg import splu
        tags, fibres = tissue["tags"], tissue["fibres"]
        _, bath = principalConductivities(config)
        sigma = np.repeat((bath * np.eye(3))[None], len(tags), axis=0)
        inTissue = tags != BATH_TAG
        sigma[inTissue] = tissueTensors(config, tags[inTissue], fibres[inTissue], "bulk")
        stiffness = assemble(tissue["positions"], tissue["elements"], sigma)
        self.count = len(tissue["positions"])
        self.free = np.setdiff1d(np.arange(self.count), np.unique(groundVertices))
        system = stiffness[self.free][:, self.free].tocsr()
        self.tolerance = tolerance
        if len(self.free) <= 150_000:
            self._solve = splu(system.tocsc()).solve
        else:
            import pyamg
            from scipy.sparse.linalg import cg
            preconditioner = pyamg.smoothed_aggregation_solver(system).aspreconditioner()

            def solve(rhs):
                out, info = cg(system, rhs, rtol=tolerance, maxiter=5000, M=preconditioner)
                if info != 0:
                    raise RuntimeError(f"Lead-field CG did not converge (info {info})")
                return out
            self._solve = solve

    def leadField(self, electrodeNodes):
        load = np.zeros(self.count)
        load[electrodeNodes] = 1.0 / len(electrodeNodes)          # 1 uA shared equally over the electrode's nodes
        z = np.zeros(self.count)
        z[self.free] = self._solve(load[self.free])
        return z
