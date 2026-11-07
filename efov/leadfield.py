"""Lead fields of the electrodes by reciprocity, from openCARP Laplace solves, and the electrograms and
contributions they give (FM-14, FM-23, FM-34).

Forward model (FM-23). Vm is openCARP's monodomain solution. The extracellular potential of the
bidomain with a bath obeys

    div((sigma_i + sigma_e) grad phi_e) = -div(sigma_i grad Vm)   in tissue,   div(g_bath grad phi_e) = 0 in the bath,

discretised with P1 elements as K phi = -S_i Vm (K with sigma_i + sigma_e in tissue and lesion and
g_bath in the bath, S_i with sigma_i on tissue and lesion elements, phi = 0 on the corner ground).
This is the classical recovery of phi_e from a monodomain Vm. openCARP's pseudo-bidomain uses
sigma_m as the source instead (VL-2026-11-01-02).

Electrode (FM-34): a conductor. Its mesh nodes S share one potential and it draws no net current
while recording. openCARP's Laplace mode (experiment 2) solves K u = 0 with u = 1 mV on S and u = 0
on the ground; its operator is g_i + g_e in tissue and lesion and g_bath in the bath (checked
against an independent P1 solve to 2e-8, VL-2026-11-07-01), so the run receives the model's own
g_i and g_e. The current the electrode draws is I = u^T K u (uA for u in mV). Reciprocity for a
floating equipotential electrode then gives, exactly for the discrete problem,

    phi_S(t) = -z^T S_i Vm(t),   z = u / I   (mV per uA).

openCARP 19.0.0 ignores extracellular current stimuli in Laplace and bidomain runs (VL-2026-11-04-01,
VL-2026-11-07-01), so the electrode is driven by a potential and normalised by its current. Only the
normalising current uses K, assembled here once per condition from openCARP's mesh.
  * the electrogram is a fixed weighting of Vm on the tissue nodes: phi(t) = w_tissue . Vm(t),
    w_tissue = -S_i z (nodalWeights);
  * element e contributes c_e(t) = coeff_e . Vm_e(t), coeff_e = -vol_e (grad N)^T sigma_i,e grad z
    (coefficients); the coefficients of an element sum to zero, so a uniform Vm contributes nothing.
Units: z in mV/uA, positions in mm, sigma in S/m and Vm in mV give phi in mV (the factors cancel:
1e3 [V/A per mV/uA] x 1e6 [two gradients] x 1e-9 [mm^3] x 1e-3 [V per mV] x 1e3 [mV per V] = 1).

Only z on the tissue nodes is kept (float32): that is all the source term reads. Lead fields are
cached by node set under a folder named by the conductor fingerprint (efov.config.conductor), so
variants that change only the wave, beta, electrodes, noise or analysis reuse them.
"""
import hashlib
import os
import time
from concurrent.futures import ThreadPoolExecutor
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
    """Conductivity regions of the Laplace run: the model's g_i and g_e (tissue, lesion) and g_bath.

    openCARP's Laplace operator is g_i + g_e in tissue (VL-2026-11-07-01), the bidomain extracellular operator.
    """
    regions, bath = principalConductivities(config)
    lines = {"num_gregions": 3}
    for index, (name, tag) in enumerate((("tissue", TISSUE_TAG), ("lesion", LESION_TAG))):
        gi, ge = regions[name]
        lines.update({f"gregion[{index}].num_IDs": 1, f"gregion[{index}].ID[0]": tag})
        for axis, i, e in zip("ltn", gi, ge):
            lines[f"gregion[{index}].g_i{axis}"] = float(i)
            lines[f"gregion[{index}].g_e{axis}"] = float(e)
    lines.update({"gregion[2].num_IDs": 1, "gregion[2].ID[0]": BATH_TAG, "gregion[2].g_bath": float(bath)})
    return lines


# ----------------------------------------------------------------- openCARP Laplace route
def electrodePotentialMv(config):
    """The potential the Laplace run holds the electrode at (config/leadfield.par stim[1].pulse.strength)."""
    from .parfile import readPar
    return float(readPar(config["leadFieldPar"])["stim[1].pulse.strength"])


def laplaceOutput(folder, nodeCount):
    """The potential written by the Laplace run (last frame); the file name is checked, not assumed."""
    candidates = [Path(folder) / name for name in ("phie.igb", "phi.igb", "laplace.igb")]
    candidates += sorted(p for p in Path(folder).glob("*.igb") if p not in candidates and p.name != "vm.igb")
    for path in candidates:
        if path.exists() and igbHeader(path)[1] == nodeCount:
            return readIgb(path, slice(-1, None))[0].astype(float), path.name
    found = sorted(p.name for p in Path(folder).glob("*"))
    raise RuntimeError(f"No whole-mesh potential found after the Laplace run in {folder}: {found}")


def laplacePotential(config, meshBase, groundVertices, electrodeNodes, nodeCount, folder, ranks=1):
    """u (mV, all nodes): openCARP Laplace solve with the electrode at 1 mV and the corner ground at 0 mV.

    ranks MPI processes per solve (efov runs several solves at once). The run folder keeps its resolved
    parameter file and log; the large outputs are deleted once read.
    """
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    writeVertices(folder / "electrode.vtx", electrodeNodes)
    writeVertices(folder / "ground.vtx", groundVertices)
    lines = {"meshname": str(meshBase), "simID": str(folder),
             "stim[0].elec.vtx_file": str(folder / "ground"), "stim[1].elec.vtx_file": str(folder / "electrode"),
             **laplaceRegionLines(config)}
    parPath = writeResolved(config["leadFieldPar"], {}, lines, folder / "leadfield.par",
                            header="# Resolved Laplace lead-field parameters (efov; FM-14, FM-34)")
    from .simulation import findExecutable, runLogged
    command = [findExecutable("openCARP", "OPENCARP_BIN"), "+F", str(parPath)]
    if ranks > 1:
        command = [os.environ.get("MPIEXEC_BIN", "mpiexec"), "-n", str(ranks)] + command
    if runLogged(command, folder / "laplace.log") != 0:
        raise RuntimeError(f"openCARP Laplace run failed; see {folder / 'laplace.log'}")
    u, name = laplaceOutput(folder, nodeCount)
    for output in folder.glob("*.igb"):
        output.unlink()
    return u


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


# ----------------------------------------------------------------- P1 matrices (source term and the normalising current)
def assemble(positions, elements, conductivity):
    """Sparse sum_e vol_e G_e^T sigma_e G_e (N x N); sigma (E, 3, 3)."""
    from scipy import sparse
    volumes, gradients = elementGradients(positions, elements)
    local = np.einsum("e,eia,eij,ejb->eab", volumes, gradients, conductivity, gradients)
    i = np.repeat(elements, 4, axis=1).ravel()
    j = np.tile(elements, (1, 4)).ravel()
    return sparse.csr_matrix((local.ravel(), (i, j)), shape=(len(positions), len(positions)))


def bulkStiffness(config, tissue, block=2_000_000):
    """K of the whole run mesh: sigma_i + sigma_e on tissue and lesion elements, g_bath on bath elements.

    Used only for the electrode current I = u^T K u; assembled in element blocks to bound memory.
    """
    tags, fibres, elements = tissue["tags"], tissue["fibres"], tissue["elements"]
    _, bath = principalConductivities(config)
    total = None
    for first in range(0, len(elements), block):
        rows = slice(first, first + block)
        sigma = np.repeat((bath * np.eye(3))[None], len(elements[rows]), axis=0)
        inTissue = tags[rows] != BATH_TAG
        sigma[inTissue] = tissueTensors(config, tags[rows][inTissue], fibres[rows][inTissue], "bulk")
        part = assemble(tissue["positions"], elements[rows], sigma)
        total = part if total is None else total + part
    return total


def leadFieldFromPotential(u, stiffness, electrodeNodes, groundVertices, potentialMv):
    """z = u / I (mV/uA), with I = u^T K u / V (uA per mV of drive), after checking the boundary values openCARP applied."""
    atElectrode, atGround = u[np.asarray(electrodeNodes)], u[np.asarray(groundVertices)]
    if np.abs(atElectrode - potentialMv).max() > 1e-6 * potentialMv or np.abs(atGround).max() > 1e-6 * potentialMv:
        raise RuntimeError(f"Laplace boundary values not applied: electrode {atElectrode.min():.6g}-{atElectrode.max():.6g} mV, "
                           f"ground up to {np.abs(atGround).max():.3g} mV")
    interior = np.ones(len(u), bool)
    interior[np.asarray(electrodeNodes)] = interior[np.asarray(groundVertices)] = False
    if np.mean(np.abs(u[interior]) > 1e-9 * potentialMv) < 0.5:
        # A solve that did not iterate (absolute tolerance met at the start, mis-oriented elements) leaves u = 0
        # away from the boundary values, and u^T K u would still be positive.
        raise RuntimeError("Laplace potential is zero on most of the mesh: the openCARP solve did not run; see its laplace.log")
    if u.min() < -1e-6 * potentialMv or u.max() > potentialMv * (1 + 1e-6):
        raise RuntimeError(f"Laplace potential outside [0, {potentialMv}] mV (maximum principle): {u.min():.3g} to {u.max():.3g}")
    current = float(u @ (stiffness @ u)) / potentialMv            # uA drawn at potentialMv
    if not np.isfinite(current) or current <= 0:
        raise RuntimeError(f"Non-positive electrode current {current} uA")
    return u / current, current                                    # u / I: the field of a 1 uA injection


# ----------------------------------------------------------------- cached lead fields of electrode node sets
def nodeSetKey(nodes):
    return hashlib.sha256(np.asarray(nodes, np.int64).tobytes()).hexdigest()[:20]


# Peak memory of one openCARP Laplace run, GB = LAPLACE_MEMORY_GB[0] + LAPLACE_MEMORY_GB[1] x nodes / 1e6: a
# conservative fit to runs at the production spacing (2.28 GB at 0.51 M nodes, 1.67 GB at 0.31 M; VL-2026-11-07-02).
LAPLACE_MEMORY_GB = (0.75, 4.5)


def laplaceSchedule(nodeCount, processCount, availableGB=None):
    """(simultaneous runs, MPI ranks per run): as many runs as memory allows (80 % of what is available), up to
    processCount, and the remaining cores shared out as ranks."""
    if availableGB is None:
        try:
            with open("/proc/meminfo") as stream:
                fields = dict(line.split(":", 1) for line in stream)
            availableGB = int(fields["MemAvailable"].split()[0]) / 1e6
        except (OSError, KeyError, ValueError):
            availableGB = float("inf")
    perRun = LAPLACE_MEMORY_GB[0] + LAPLACE_MEMORY_GB[1] * nodeCount / 1e6
    runs = int(max(1, min(processCount, 0.8 * availableGB // perRun)))
    return runs, max(1, processCount // runs), perRun, availableGB


def electrodeLeadFields(config, tissue, groundVertices, nodeSets, folder, processCount=1, log=print):
    """Lead field on the tissue nodes (float32 .npy, mV/uA) of every node set; computed once per conductor and node set.

    folder is <lead-field cache>/<conductor fingerprint>/<condition>. Its manifest records the mesh it
    belongs to; a cache built on a different mesh is refused. The openCARP Laplace runs are independent:
    as many run at once as memory allows (laplaceSchedule), sharing processCount cores. Every lead field
    is written atomically with a record of its electrode current, so an interrupted stage resumes.
    """
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    manifest = dict(meshBase=str(tissue["meshBase"]), nodes=int(len(tissue["positions"])), tissueNodes=int(len(tissue["tissueNodes"])),
                    method="openCARP Laplace, equipotential electrode", groundVertices=int(len(groundVertices)))
    if (folder / "manifest.json").exists():
        recorded = loadJson(folder / "manifest.json")
        if {k: recorded.get(k) for k in ("nodes", "tissueNodes", "method", "groundVertices")} != \
                {k: manifest[k] for k in ("nodes", "tissueNodes", "method", "groundVertices")}:
            raise ValueError(f"Lead-field cache {folder} belongs to a different mesh or method")
    else:
        saveJson(folder / "manifest.json", manifest)
    paths = [folder / f"{nodeSetKey(nodes)}.npy" for nodes in nodeSets]
    missing = [(np.asarray(nodes), path) for nodes, path in zip(nodeSets, paths) if not path.exists()]
    missing = list({str(path): (nodes, path) for nodes, path in missing}.values())   # overlapping grids repeat node sets
    if not missing:
        return paths, 0
    started = time.time()
    stiffness, potential = bulkStiffness(config, tissue), electrodePotentialMv(config)
    runs, ranks, perRun, available = laplaceSchedule(len(tissue["positions"]), int(processCount))
    log(f"  lead fields: K assembled in {time.time() - started:.0f} s; {len(missing)} openCARP Laplace runs, {runs} at a time "
        f"x {ranks} MPI rank(s) (about {perRun:.1f} GB each, {available:.0f} GB available), in {folder}")

    def solveOne(task):
        nodes, path = task
        u = laplacePotential(config, tissue["meshBase"], groundVertices, nodes, len(tissue["positions"]), folder / "runs" / path.stem, ranks)
        z, current = leadFieldFromPotential(u, stiffness, nodes, groundVertices, potential)
        temporary = path.with_suffix(".partial.npy")
        np.save(temporary, z[tissue["tissueNodes"]].astype(np.float32))
        os.replace(temporary, path)
        saveJson(folder / "runs" / path.stem / "leadfield.json",
                 dict(electrodeNodes=int(len(nodes)), electrodePotentialMv=potential, currentUA=current,
                      leadFieldAtElectrodeMvPerUA=potential / current))
        return current

    started, done = time.time(), 0
    with ThreadPoolExecutor(max_workers=runs) as pool:
        for _ in pool.map(solveOne, missing):
            done += 1
            if done % 50 == 0 or done == len(missing):
                log(f"  lead fields: {done} of {len(missing)} in {time.time() - started:.0f} s")
    return paths, len(missing)
