"""Lead fields of the electrodes and their electrograms from openCARP's lead-field module (openCARP >= 20),
and the contributions they give (FM-14, FM-23, FM-35).

Forward model (FM-23). Vm is openCARP's monodomain solution. The extracellular potential of the
bidomain with a bath obeys

    div((sigma_i + sigma_e) grad phi_e) = -div(sigma_i grad Vm)   in tissue,   div(g_bath grad phi_e) = 0 in the bath,

with phi_e = 0 on the 1 mm corner ground. Reciprocity: the electrode reading equals Z^T K_i Vm, where Z
is the potential of a unit current injected at the electrode (openCARP manual, Eqs. 34-36).

openCARP's lead-field module (FM-35; post_processing_opts = 64, experiment = 4, config/leadfield.par)
does this in one run per condition: every electrode is a lead-field stimulus (crct.type 13, 1 uA spread
over the electrode volume, weighted by the mass matrix), every lead field Z solves the bulk system once
(LF_<name>.igb on the tissue nodes), and every electrogram V_L(t) = Z_L . (K_i Vm(t)) is streamed from the
monodomain vm.igb (ECG_<name>.dat). The run uses bidomain = 1 so that K_i is the sigma_i stiffness (in
monodomain and pseudo-bidomain mode openCARP builds that matrix with the monodomain tensor sigma_m;
VL-2026-11-01-02). The electrode reading is therefore the volume-weighted mean of phi_e over its nodes.

The ground truth (efov.groundtruth) needs the element contributions c_e(t) = coeff_e . Vm_e(t),
coeff_e = -vol_e (grad N)^T sigma_i,e grad Z, which openCARP does not output: ContributionModel forms
them from openCARP's lead fields. efov.electrograms checks that openCARP's traces equal -Z^T S_i Vm from
the same lead fields, so electrograms and ground truth cannot drift apart.
Units: Z in mV/uA, positions in mm, sigma in S/m and Vm in mV give phi in mV (the factors cancel:
1e3 [V/A per mV/uA] x 1e6 [two gradients] x 1e-9 [mm^3] x 1e-3 [V per mV] x 1e3 [mV per V] = 1).

Lead fields (float32, tissue nodes in vm.igb order) are cached per conductor fingerprint and electrode
node set (efov.config.conductor), so variants that change only the wave, beta, noise or analysis reuse them.
"""
import hashlib
import os
import subprocess
import time
from pathlib import Path

import numpy as np

from .config import par
from .io import igbHeader, igbTimes, loadJson, readIgb, saveJson, writeVertices
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


def regionLines(config):
    """Conductivity regions of the lead-field run: the model's g_i and g_e (tissue, lesion) and g_bath.

    openCARP assembles the bulk operator g_i + g_e in tissue (VL-2026-11-07-01) and K_i from g_i.
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


# ----------------------------------------------------------------- openCARP lead-field module (v20)
LEAD_FIELD_STIMULUS = 13          # crct.type "leadfield stim (extracellular current)", openCARP >= 20


def leadName(nodes):
    return "E" + nodeSetKey(nodes)


def supportsLeadfield(executable=None):
    """True if this openCARP build has the lead-field post-processing module (openCARP >= 20)."""
    from .simulation import findExecutable
    executable = executable or findExecutable("openCARP", "OPENCARP_BIN")
    text = subprocess.run([executable, "+Help", "post_processing_opts"], capture_output=True, text=True).stdout
    return "Leadfield" in text


def runOpenCarpLeadfield(config, meshBase, groundVertices, leads, vmFile, folder, ecgStepMs, ranks=1, leadFieldDir=None, overrides=None):
    """One openCARP lead-field run. leads: [(name, nodes)]. Returns {name: LF path or None}, {name: ECG path}.

    With leadFieldDir the lead fields LF_<name>.igb are loaded from it; otherwise openCARP constructs them.
    Electrograms are computed from vmFile at ecgStepMs. The run folder keeps the resolved parameter file and log.
    """
    from .parfile import writeResolved
    from .simulation import findExecutable, runLogged
    import shutil
    folder = Path(folder)
    if folder.exists():
        shutil.rmtree(folder)                      # no stale LF_/ECG_ files from an interrupted run
    (folder / "electrodes").mkdir(parents=True)
    writeVertices(folder / "ground.vtx", groundVertices)
    lines = {"meshname": str(meshBase), "simID": str(folder), "lf_vmfile": str(vmFile), "ecg_timedt": float(ecgStepMs),
             **regionLines(config), "num_stim": len(leads) + 1,
             "stim[0].crct.type": 3, "stim[0].elec.vtx_file": str(folder / "ground")}
    for index, (name, nodes) in enumerate(leads, start=1):
        writeVertices(folder / "electrodes" / f"{name}.vtx", nodes)
        lines.update({f"stim[{index}].name": name, f"stim[{index}].crct.type": LEAD_FIELD_STIMULUS,
                      f"stim[{index}].elec.vtx_file": str(folder / "electrodes" / name), f"stim[{index}].pulse.strength": 1.0})
    if leadFieldDir is not None:
        lines["lf_dir"] = str(leadFieldDir)
    parPath = writeResolved(config["leadFieldPar"], dict(overrides or {}), lines, folder / "leadfield.par",
                            header="# Resolved openCARP lead-field run (efov; FM-35)")
    command = [findExecutable("openCARP", "OPENCARP_BIN"), "+F", str(parPath)]
    if ranks > 1:
        command = [os.environ.get("MPIEXEC_BIN", "mpiexec"), "-n", str(ranks)] + command
    if runLogged(command, folder / "leadfield.log", cwd=folder) != 0:
        raise RuntimeError(f"openCARP lead-field run failed; see {folder / 'leadfield.log'}")
    found = {p.name: p for p in folder.rglob("*") if p.is_file()}
    fields = {name: found.get(f"LF_{name}.igb") for name, _ in leads}
    traces = {name: found.get(f"ECG_{name}.dat") for name, _ in leads}
    missing = [n for n, p in traces.items() if p is None] + ([n for n, p in fields.items() if p is None] if leadFieldDir is None else [])
    if missing:
        raise RuntimeError(f"openCARP lead-field run wrote no output for {len(missing)} lead(s), e.g. {missing[:3]}; "
                           f"see {folder / 'leadfield.log'} (openCARP >= 20 is required)")
    return fields, traces


def readEcg(path):
    """(times ms, values mV) of an ECG_<lead>.dat file (two columns, no header)."""
    data = np.loadtxt(path, ndmin=2)
    return data[:, 0], data[:, 1]


def loadLeadField(path):
    """Lead field on the tissue nodes (float64) from a cached LF igb file."""
    return readIgb(path)[0].astype(float)


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


# ----------------------------------------------------------------- P1 matrices (source term of the contributions)
def assemble(positions, elements, conductivity):
    """Sparse sum_e vol_e G_e^T sigma_e G_e (N x N); sigma (E, 3, 3)."""
    from scipy import sparse
    volumes, gradients = elementGradients(positions, elements)
    local = np.einsum("e,eia,eij,ejb->eab", volumes, gradients, conductivity, gradients)
    i = np.repeat(elements, 4, axis=1).ravel()
    j = np.tile(elements, (1, 4)).ravel()
    return sparse.csr_matrix((local.ravel(), (i, j)), shape=(len(positions), len(positions)))


# ----------------------------------------------------------------- cached lead fields of electrode node sets
def nodeSetKey(nodes):
    return hashlib.sha256(np.asarray(nodes, np.int64).tobytes()).hexdigest()[:20]


def leadFieldsAndTraces(config, tissue, groundVertices, nodeSets, vmFile, cacheFolder, runFolder, processCount=1, log=print):
    """Lead field (cached LF igb, tissue nodes) and electrogram trace of every node set, from openCARP's lead-field module.

    One openCARP run: if every lead field of this run is cached they are loaded (lf_dir); otherwise all of
    this run's lead fields are constructed and the new ones are added to the cache. Electrograms are computed
    from vmFile at the Vm output step. Returns (cache paths, number constructed, times ms, traces (sets, frames)).
    cacheFolder is <lead-field cache>/<conductor fingerprint>/<condition>; its manifest records the mesh.
    """
    cacheFolder, runFolder = Path(cacheFolder), Path(runFolder)
    cacheFolder.mkdir(parents=True, exist_ok=True)
    manifest = dict(meshBase=str(tissue["meshBase"]), nodes=int(len(tissue["positions"])), tissueNodes=int(len(tissue["tissueNodes"])),
                    method="openCARP lead-field module (crct.type 13)", groundVertices=int(len(groundVertices)))
    if (cacheFolder / "manifest.json").exists():
        recorded = loadJson(cacheFolder / "manifest.json")
        if {k: recorded.get(k) for k in ("nodes", "tissueNodes", "method", "groundVertices")} != \
                {k: manifest[k] for k in ("nodes", "tissueNodes", "method", "groundVertices")}:
            raise ValueError(f"Lead-field cache {cacheFolder} belongs to a different mesh or method")
    else:
        saveJson(cacheFolder / "manifest.json", manifest)
    leads = list({leadName(n): (leadName(n), np.asarray(n)) for n in nodeSets}.values())   # overlapping grids repeat node sets
    paths = {name: cacheFolder / f"{name}.igb" for name, _ in leads}
    cached = all(p.exists() for p in paths.values())
    step = float(igbTimes(vmFile)[1] - igbTimes(vmFile)[0])
    if cached:
        loadDir = runFolder / "leadFields"
        loadDir.mkdir(parents=True, exist_ok=True)
        for name, path in paths.items():
            link = loadDir / f"LF_{name}.igb"
            if not link.exists():
                link.symlink_to(path)
    started = time.time()
    log(f"  lead fields: one openCARP run, {len(leads)} leads ({'loaded from cache' if cached else 'constructed'}), "
        f"{processCount} MPI rank(s), in {runFolder}")
    fields, traceFiles = runOpenCarpLeadfield(config, tissue["meshBase"], groundVertices, leads, vmFile, runFolder / "openCARP",
                                              step, processCount, loadDir if cached else None)
    log(f"  lead fields: openCARP run finished in {time.time() - started:.0f} s")
    constructed = 0
    if not cached:
        for name, path in paths.items():
            if not path.exists():
                values = readIgb(fields[name])
                if values.shape[1] != len(tissue["tissueNodes"]):
                    raise ValueError(f"LF_{name}.igb has {values.shape[1]} nodes, the tissue {len(tissue['tissueNodes'])}")
                temporary = path.with_name(path.name + ".partial")
                os.replace(fields[name], temporary)
                os.replace(temporary, path)
                constructed += 1
    times, traces = None, {}
    for name, _ in leads:
        t, values = readEcg(traceFiles[name])
        times = t if times is None else times
        traces[name] = values
    frameTimes = igbTimes(vmFile)
    if len(times) != len(frameTimes) or np.abs(times - (frameTimes - frameTimes[0])).max() > 1e-6 * max(1.0, frameTimes[-1]):
        raise ValueError(f"openCARP electrogram times ({len(times)} samples) do not match the Vm frames ({len(frameTimes)})")
    order = [leadName(n) for n in nodeSets]
    return [paths[name] for name in order], constructed, frameTimes, np.stack([traces[name] for name in order])
