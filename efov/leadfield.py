"""Lead fields of the electrodes by reciprocity, and the electrograms and contributions they give (FM-14, FM-23).

Forward model (FM-23). Vm is openCARP's monodomain solution. The extracellular potential of the
bidomain with a bath obeys

    div((sigma_i + sigma_e) grad phi_e) = -div(sigma_i grad Vm)   in tissue,   div(g_bath grad phi_e) = 0 in the bath,

discretised with P1 elements as K phi = -S_i Vm (K with sigma_i + sigma_e in tissue and lesion and
g_bath in the bath, S_i with sigma_i on tissue and lesion elements, phi = 0 on the corner ground).
This is the classical recovery of phi_e from a monodomain Vm. openCARP's pseudo-bidomain uses
sigma_m as the source instead (VL-2026-11-01-02), which weights fibre-parallel and cross-fibre
sources differently whenever the intra- and extracellular anisotropy ratios differ.

Reciprocity: an electrode with node set S records phi = mean over S of phi_e. With K z = w
(w = 1/|S| on S; a 1 uA total current injected over S), phi = -z^T S_i Vm. So
  * the electrogram is a fixed weighting of Vm on the tissue nodes: phi(t) = w_tissue . Vm(t),
    w_tissue = -S_i z (nodalWeights);
  * element e contributes c_e(t) = coeff_e . Vm_e(t), coeff_e = -vol_e (grad N)^T sigma_i,e grad z
    (coefficients); the coefficients of an element sum to zero, so a uniform Vm contributes nothing.
Units: z in mV/uA, positions in mm, sigma in S/m and Vm in mV give phi in mV (the factors cancel:
1e3 [V/A per mV/uA] x 1e6 [two gradients] x 1e-9 [mm^3] x 1e-3 [V per mV] x 1e3 [mV per V] = 1).

Lead fields come from openCARP Laplace solves (config/leadfield.par, one per electrode node set;
fixed.groundTruth.leadFieldMethod = "openCARP"). "python" assembles and solves the same discrete
problem here; it serves the unit tests and is the pre-declared fallback of FM-14.
Only z on the tissue nodes is kept (float32): that is all the source term reads. Lead fields are
cached by node set under a folder named by the conductor fingerprint (efov.config.conductor), so
variants that change only the wave, beta, electrodes, noise or analysis reuse them.
"""
import hashlib
import time
import os
from pathlib import Path

import numpy as np

from .config import par
from .io import igbHeader, loadJson, readIgb, saveJson, writeVertices
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
    (folder / name).unlink()                       # the whole-mesh potential is large; the tissue part is kept by the caller
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
    """Per-element tensors on tissue and lesion elements.

    which = 'intracellular' (sigma_i, the source), 'bulk' (sigma_i + sigma_e, the Laplace operator)
    or 'monodomain' (sigma_m = sigma_i sigma_e / (sigma_i + sigma_e), openCARP's propagation tensor).
    """
    regions, _ = principalConductivities(config)
    principal = np.zeros((len(tags), 3))
    for name, tag in (("tissue", TISSUE_TAG), ("lesion", LESION_TAG)):
        gi, ge = regions[name]
        principal[tags == tag] = {"intracellular": gi, "bulk": gi + ge, "monodomain": gi * ge / (gi + ge)}[which]
    return tensors(principal, localFrames(fibres))


class ContributionModel:
    """Tissue elements of a run with their sigma_i tensors: turns a lead field into electrogram weights and contributions.

    Lead fields are given on the tissue nodes only, in vm.igb column order (tissue["tissueNodes"]).
    """

    def __init__(self, config, tissue):
        tissueElements = tissue["tags"] != BATH_TAG
        self.elements = tissue["elements"][tissueElements]
        self.tags = tissue["tags"][tissueElements]
        self.positions = tissue["positions"]
        self.centres = self.positions[self.elements].mean(axis=1)
        self.volumes, self.gradients = elementGradients(self.positions, self.elements)
        self.sigmaI = tissueTensors(config, self.tags, tissue["fibres"][tissueElements], "intracellular")
        self.tissueNodes = tissue["tissueNodes"]
        local = np.full(len(self.positions), -1, np.int64)
        local[self.tissueNodes] = np.arange(len(self.tissueNodes))
        self.localElements = local[self.elements]                 # element corners as indices into vm.igb columns
        self.sourceStiffness = assemble(np.asarray(self.positions)[self.tissueNodes], self.localElements, self.sigmaI)

    def nodalWeights(self, leadFieldOnTissue):
        """(tissue nodes,): phi(t) = weights . Vm(t), weights = -S_i z."""
        return -(self.sourceStiffness @ np.asarray(leadFieldOnTissue, float))

    def coefficients(self, leadFieldOnTissue):
        """(E_tissue, 4): phi = sum_e coeff_e . Vm_e."""
        z = np.asarray(leadFieldOnTissue, float)[self.localElements]
        minusGradZ = -np.einsum("eij,ej->ei", self.gradients, z)
        return np.einsum("ec,ecd,edj,e->ej", minusGradZ, self.sigmaI, self.gradients, self.volumes)


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


# ----------------------------------------------------------------- cached lead fields of electrode node sets
def nodeSetKey(nodes):
    return hashlib.sha256(np.asarray(nodes, np.int64).tobytes()).hexdigest()[:20]


def electrodeLeadFields(config, tissue, groundVertices, nodeSets, folder, processCount=1, log=print):
    """Lead field on the tissue nodes (float32 .npy, mV/uA) of every node set; computed once per conductor and node set.

    folder is <lead-field cache>/<conductor fingerprint>/<condition>. Its manifest records the mesh it
    belongs to; a cache built on a different mesh is refused.
    """
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    manifest = dict(meshBase=str(tissue["meshBase"]), nodes=int(len(tissue["positions"])), tissueNodes=int(len(tissue["tissueNodes"])),
                    method=config["fixed"]["groundTruth"]["leadFieldMethod"], groundVertices=int(len(groundVertices)))
    if (folder / "manifest.json").exists():
        recorded = loadJson(folder / "manifest.json")
        if {k: recorded[k] for k in ("nodes", "tissueNodes", "method", "groundVertices")} != \
                {k: manifest[k] for k in ("nodes", "tissueNodes", "method", "groundVertices")}:
            raise ValueError(f"Lead-field cache {folder} belongs to a different mesh or method")
    else:
        saveJson(folder / "manifest.json", manifest)
    method = manifest["method"]
    paths = [folder / f"{nodeSetKey(nodes)}.npy" for nodes in nodeSets]
    missing = [(nodes, path) for nodes, path in zip(nodeSets, paths) if not path.exists()]
    missing = list({str(path): (nodes, path) for nodes, path in missing}.values())   # overlapping grids repeat node sets
    if method == "openCARP":
        for count, (nodes, path) in enumerate(missing, 1):
            z = openCarpLeadField(config, tissue["meshBase"], groundVertices, nodes, len(tissue["positions"]),
                                  folder / "runs" / path.stem, processCount)
            np.save(path, z[tissue["tissueNodes"]].astype(np.float32))
            if count % 50 == 0:
                log(f"  lead fields: {count} of {len(missing)} computed in {folder}")
    elif missing:
        _pythonLeadFields(config, tissue, groundVertices, missing, processCount, log, folder)
    return paths, len(missing)


# One factorised/preconditioned system per process group: built once, then shared with forked workers
# copy-on-write. Each worker solves whole lead fields and writes its own files (no results pass through pipes).
_SHARED = {}


def _solveAndSave(task):
    nodes, path, tissueNodes = task
    z = _SHARED["solver"].leadField(nodes)
    temporary = path.with_suffix(".partial.npy")
    np.save(temporary, z[tissueNodes].astype(np.float32))
    os.replace(temporary, path)                     # a file exists only once complete, so an interrupted run resumes
    return str(path)


def _pythonLeadFields(config, tissue, groundVertices, missing, processCount, log, folder):
    """FM-14 route: assemble K once (sigma_i + sigma_e in tissue, g_bath in bath), solve every missing lead field.

    Solves are independent; with processCount > 1 they run in forked worker processes sharing the one
    solver (Linux fork, copy-on-write). The result does not depend on the number of processes.
    """
    import multiprocessing
    started = time.time()
    _SHARED["solver"] = PythonLeadFieldSolver(config, tissue, groundVertices)
    log(f"  lead fields: system of {len(_SHARED['solver'].free):,} unknowns set up in {time.time() - started:.0f} s; "
        f"{len(missing)} to solve on {processCount} process(es) in {folder}")
    tasks = [(np.asarray(nodes), path, tissue["tissueNodes"]) for nodes, path in missing]
    started, done = time.time(), 0
    try:
        if processCount > 1 and len(tasks) > 1:
            with multiprocessing.get_context("fork").Pool(processCount) as pool:
                for _ in pool.imap_unordered(_solveAndSave, tasks):
                    done += 1
                    if done % 50 == 0:
                        log(f"  lead fields: {done} of {len(tasks)} in {time.time() - started:.0f} s")
        else:
            for task in tasks:
                _solveAndSave(task)
                done += 1
                if done % 50 == 0:
                    log(f"  lead fields: {done} of {len(tasks)} in {time.time() - started:.0f} s")
    finally:
        _SHARED.clear()
    log(f"  lead fields: {len(tasks)} solved in {time.time() - started:.0f} s")
