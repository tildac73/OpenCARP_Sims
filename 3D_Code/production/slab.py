#!/usr/bin/env python3
"""Production slab model as a standard carputils experiment (one condition per run).

This script is the single definition of the simulated model used for the final settings run
and, once frozen, for every parameter-sweep run. It follows the carputils experiment template
(openCARP example 02_EP_tissue/00_simple): standard parser, @tools.carpexample, mesh.Block,
tools.carp_cmd and job.carp. Every scientific value comes from the JSON run spec (--spec),
which finalise.py writes; nothing scientific is set here.

Model
  * Tissue slab x in [-Lx/2, Lx/2], y in [-Ly/2, Ly/2], z in [-t/2, t/2] (mm); fibres along +x.
  * Blood bath on the catheter side only, z in [t/2, t/2 + B], built by mesher through
    Block.set_bath. mesher gives bath elements tag 0 and zero fibre vectors, which is how
    openCARP recognises bath and applies gregion[].g_bath. (Our earlier hand-built bath mesh gave
    bath elements fibres, so openCARP used default tissue conductivities there: 1 Nov 2026.)
  * Lesion (condition "lesion"): tissue elements with y < boundary position, tag 5, near-zero
    conductivities as in the pilot (scar "insulating"). Tag 1 = viable tissue.
  * Forward model (spec "bidomain"): 2 = openCARP pseudo-bidomain (monodomain Vm; phi_e from an
    elliptic solve whose source uses the monodomain conductivity), 1 = full bidomain. Verified on
    openCARP 19.0.0 against the FEM solver: pseudo-bidomain phi_e = FEM with source sigma_m
    (exact); bidomain phi_e = FEM with source sigma_i (within 1.5 %).
  * Ground: extracellular ground (stimulus type 3) on the bath face farthest from the tissue.
  * Stimulus: transmembrane, strip of nodes with x <= xmin + stripWidth, viable tissue only
    (no node touching a lesion element), as in the pilot.
  * Ionic model: Courtemanche (openCARP), initial state from bench limit-cycle pacing (spec).
  * phi_e output (spec "phieOutputHeightMm"): openCARP still solves the whole tissue + bath, but writes
    phi_e only on the bath layer within that height of the tissue surface, where the electrodes are
    (dataout_e = 3; phieNodes.npy lists the nodes). Nothing downstream reads phi_e elsewhere.
  * Elliptic solver: carputils --flavor (finalise.py uses boomeramg; same tolerance as the default,
    results equal to float precision, about 8x faster bath solves; checked 1 Nov 2026).
  * Prepacing with checkpoint (spec "restartMs"): openCARP runs monodomain up to restartMs (Vm and
    activation times written at the full output rate; no extracellular solve), saves its state, and
    the run restarts from that state with the forward model switched on until the end. Vm and
    activation times are merged into one record from t = 0; phie.igb starts at restartMs
    (restart.json). Pseudo-bidomain Vm is the monodomain Vm, so the result equals a continuous run
    (checked 1 Nov 2026: Vm within 2e-6 mV, phi_e within 1e-6 mV). The tissue-only copy of phi_e
    (phie_i.igb, identical to phie.igb at the tissue nodes) is removed to save disk.

Usage (normally called by finalise.py):
    python3 slab.py --spec <run>.json --np 24 --ID <output folder>
    python3 slab.py --spec <run>.json --dry-run            # print the openCARP command only
"""
import json
import os
from pathlib import Path
import re
import shutil

import numpy as np
from carputils import mesh, tools

HERE = Path(__file__).resolve().parent
TISSUE_TAG, LESION_TAG, BATH_TAG = 1, 5, 0


def parser():
    result = tools.standard_parser()
    result.add_argument("--spec", required=True, help="JSON run specification written by finalise.py")
    return result


def jobID(args):
    return str(Path(args.spec).with_suffix(""))


def readSpec(path):
    spec = json.loads(Path(path).read_text())
    required = ["condition", "tissueSizeMm", "spacingMm", "bathThicknessMm", "bidomain", "conductivitiesSPerM",
                "lesionConductivitySPerM", "bloodSPerM", "boundaryYmm", "timeStepUs", "sampleStepMs", "pacing",
                "initialState", "meshRoot"]
    missing = [key for key in required if key not in spec]
    if missing:
        raise ValueError(f"{path}: missing {missing}")
    if spec["condition"] not in ("intact", "lesion"):
        raise ValueError("condition must be intact or lesion")
    if int(spec["bidomain"]) not in (1, 2):
        raise ValueError("bidomain must be 1 (bidomain) or 2 (pseudo-bidomain); the production model has a bath")
    if float(spec["bathThicknessMm"]) <= 0:
        raise ValueError("bathThicknessMm must be positive")
    return spec


def geometry(spec):
    size = np.asarray(spec["tissueSizeMm"], float)
    block = mesh.Block(size=tuple(size), resolution=float(spec["spacingMm"]))
    block.set_bath(thickness=(0.0, 0.0, float(spec["bathThicknessMm"])), both_sides=False)
    if spec["condition"] == "lesion":
        margin = float(spec["spacingMm"])
        lower = (-size[0] / 2 - margin, -size[1] / 2 - margin, -size[2] / 2 - margin)
        upper = (size[0] / 2 + margin, float(spec["boundaryYmm"]), size[2] / 2)   # tissue only: z <= t/2
        block.add_region(mesh.BoxRegion(lower, upper, tag=LESION_TAG))
    return block


def readMesh(meshBase):
    with open(f"{meshBase}.pts") as stream:
        count = int(stream.readline())
        positions = np.loadtxt(stream, ndmin=2)[:count] / 1000.0             # um -> mm
    with open(f"{meshBase}.elem") as stream:
        count = int(stream.readline())
        rows = np.loadtxt(stream, dtype=str, ndmin=2)[:count]
    if not np.all(rows[:, 0] == "Tt"):
        raise ValueError("Expected a tetrahedral mesh")
    return positions, rows[:, 1:5].astype(int), rows[:, 5].astype(int)


def writeVertices(path, nodes, kind):
    nodes = np.asarray(nodes, int)
    if not len(nodes):
        raise ValueError(f"{path}: no vertices selected")
    with open(path, "w") as stream:
        stream.write(f"{len(nodes)}\n{kind}\n")
        np.savetxt(stream, nodes, fmt="%d")


def boundarySets(meshBase, spec, folder):
    positions, elements, tags = readMesh(meshBase)
    tissueNodes = np.unique(elements[tags != BATH_TAG])
    isTissue = np.zeros(len(positions), bool); isTissue[tissueNodes] = True
    lesionNodes = np.unique(elements[tags == LESION_TAG]) if np.any(tags == LESION_TAG) else np.array([], int)
    strip = float(spec["pacing"]["stripWidthMm"])
    xmin = positions[isTissue, 0].min()
    stimulus = isTissue & (positions[:, 0] <= xmin + strip + 1e-9)
    stimulus[lesionNodes] = False
    ground = np.isclose(positions[:, 2], positions[:, 2].max())
    expectedTop = spec["tissueSizeMm"][2] / 2 + spec["bathThicknessMm"]
    if not np.isclose(positions[:, 2].max(), expectedTop):
        raise ValueError(f"Bath top at {positions[:, 2].max()} mm, expected {expectedTop} mm")
    writeVertices(folder / "stimulus.vtx", np.flatnonzero(stimulus), "extra")
    writeVertices(folder / "ground.vtx", np.flatnonzero(ground), "extra")
    height = spec.get("phieOutputHeightMm")
    if height is not None:
        # phi_e is written only where electrodes can be: the bath layer from the tissue surface up to
        # `height` mm (openCARP dataout_e = 3; output is in ascending node order, checked 1 Nov 2026).
        top = spec["tissueSizeMm"][2] / 2
        output = np.flatnonzero((positions[:, 2] >= top - 1e-9) & (positions[:, 2] <= top + float(height) + 1e-9))
        writeVertices(folder / "phieOutput.vtx", output, "extra")
        np.save(folder / "phieNodes.npy", output)
    return dict(nodes=len(positions), tissueNodes=int(isTissue.sum()), stimulusNodes=int(stimulus.sum()),
                groundNodes=int(ground.sum()), lesionElements=int((tags == LESION_TAG).sum()),
                bathElements=int((tags == BATH_TAG).sum()))


def conductivityOptions(spec):
    g, lesion = spec["conductivitiesSPerM"], float(spec["lesionConductivitySPerM"])
    options = ["-num_gregions", 3,
               "-gregion[0].num_IDs", 1, "-gregion[0].ID[0]", TISSUE_TAG,
               "-gregion[1].num_IDs", 1, "-gregion[1].ID[0]", LESION_TAG,
               "-gregion[2].num_IDs", 1, "-gregion[2].ID[0]", BATH_TAG,
               "-gregion[2].g_bath", float(spec["bloodSPerM"])]
    for name in ["g_il", "g_it", "g_in", "g_el", "g_et", "g_en"]:
        options += [f"-gregion[0].{name}", float(g[name]), f"-gregion[1].{name}", lesion]
    # Intracellular domain: viable tissue and lesion (lesion at near-zero conductivity, as the pilot).
    # Extracellular domain: tissue, lesion and bath.
    options += ["-num_phys_regions", 2,
                "-phys_region[0].ptype", 0, "-phys_region[0].name", "intracellular",
                "-phys_region[0].num_IDs", 2, "-phys_region[0].ID[0]", TISSUE_TAG, "-phys_region[0].ID[1]", LESION_TAG,
                "-phys_region[1].ptype", 1, "-phys_region[1].name", "extracellular",
                "-phys_region[1].num_IDs", 3, "-phys_region[1].ID[0]", TISSUE_TAG, "-phys_region[1].ID[1]", LESION_TAG,
                "-phys_region[1].ID[2]", BATH_TAG]
    return options


def igbHeader(path):
    with open(path, "rb") as stream:
        raw = stream.read(1024)
    fields = dict(token.split(":", 1) for token in raw.decode("latin-1").split() if ":" in token)
    nodes = int(fields.get("x", 1)) * int(fields.get("y", 1)) * int(fields.get("z", 1))
    size = {"float": 4, "double": 8}[fields["type"]]
    return raw, fields, nodes, int(fields["t"]), size


def mergeIgb(first, second, target):
    """target = every frame of first, then second without its first frame (which repeats first's last time)."""
    raw, fields, nodes, framesA, size = igbHeader(first)
    _, fieldsB, nodesB, framesB, sizeB = igbHeader(second)
    if (nodes, size, fields.get("inc_t")) != (nodesB, sizeB, fieldsB.get("inc_t")):
        raise ValueError(f"Cannot merge {first} and {second}: different nodes, type or time step")
    total = framesA + framesB - 1
    text = re.sub(r"(?<![\w])t:\d+", f"t:{total}", raw.decode("latin-1"), count=1)
    text = re.sub(r"dim_t:\S+", f"dim_t:{(total - 1) * float(fields['inc_t']):g}", text, count=1)
    body = text.rstrip(" \r\n\x0c\x00")
    if len(body) > 1022:
        raise ValueError("IGB header too long")
    header = (body + " " * (1023 - len(body)) + "\x0c").encode("latin-1")
    frameBytes = nodes * size
    with open(target, "wb") as out:
        out.write(header)
        with open(first, "rb") as stream:
            stream.seek(1024); shutil.copyfileobj(stream, out, 64 << 20)
        with open(second, "rb") as stream:
            stream.seek(1024 + frameBytes); shutil.copyfileobj(stream, out, 64 << 20)


def mergeActivations(first, second, target, restartMs):
    """Activation times of both phases (restored runs restart their LAT file); events at t >= restartMs from the second."""
    rows = []
    for path, keep in [(first, lambda t: t < restartMs), (second, lambda t: t >= restartMs)]:
        if Path(path).stat().st_size:
            values = np.loadtxt(path, ndmin=2)
            rows += [row for row in values if keep(row[1])]
    rows = np.array(rows) if rows else np.zeros((0, 2))
    np.savetxt(target, rows, fmt=["%d", "%.6f"], delimiter="\t")


@tools.carpexample(parser, jobID)
def run(args, job):
    spec = readSpec(args.spec)
    folder = Path(job.ID).resolve()
    folder.mkdir(parents=True, exist_ok=True)
    meshBase = mesh.generate(geometry(spec), rootdir=str(spec["meshRoot"]))
    sets = boundarySets(meshBase, spec, folder)
    pacing = spec["pacing"]
    durationMs = float(pacing["cycleLengthMs"]) * int(pacing["beatCount"])
    if "durationMs" in spec:                                          # short verification runs only
        durationMs = float(spec["durationMs"])
    restartMs = spec.get("restartMs")
    if restartMs is not None and not 0 < float(restartMs) < durationMs:
        raise ValueError("restartMs must lie inside the run")

    def command(simID, bidomain, tend):
        cycle = float(pacing["cycleLengthMs"])
        cmd = tools.carp_cmd(str(HERE / "production.par"))
        cmd += ["-simID", str(simID), "-meshname", meshBase, "-bidomain", int(bidomain),
                "-dt", float(spec["timeStepUs"]), "-spacedt", float(spec["sampleStepMs"]), "-tend", float(tend),
                "-imp_region[0].im_sv_init", str(spec["initialState"])]
        cmd += conductivityOptions(spec)
        cmd += ["-num_stim", 2,
                "-stim[0].crct.type", 0, "-stim[0].pulse.shape", 0, "-stim[0].ptcl.start", 0,
                "-stim[0].elec.vtx_file", str(folder / "stimulus"),
                "-stim[0].pulse.strength", float(pacing["strength"]),
                "-stim[0].ptcl.duration", float(pacing["durationMs"]),
                "-stim[0].ptcl.bcl", min(cycle, float(tend)),
                "-stim[0].ptcl.npls", int(np.ceil(float(tend) / cycle)),
                "-stim[1].crct.type", 3, "-stim[1].elec.vtx_file", str(folder / "ground")]
        return cmd

    commands = {}
    if restartMs is None:
        commands["run"] = command(folder, spec["bidomain"], durationMs)
    else:
        restartMs = float(restartMs)
        prepace = folder / "prepace"
        commands["prepace"] = command(prepace, 0, restartMs) + ["-num_tsav", 1, "-tsav[0]", restartMs,
                                                                "-write_statef", "state"]
        commands["run"] = command(folder, spec["bidomain"], durationMs) + [
            "-start_statef", str(prepace / f"state.{restartMs:.1f}")]
    if spec.get("phieOutputHeightMm") is not None:
        commands["run"] += ["-dataout_e", 3, "-dataout_e_vtx", str(folder / "phieOutput")]
    (folder / "slabRun.json").write_text(json.dumps(dict(spec=spec, meshBase=str(meshBase), sets=sets,
                                                         commands={k: [str(c) for c in v] for k, v in commands.items()}),
                                                    indent=2) + "\n")
    if "prepace" in commands:
        job.carp(commands["prepace"])
    job.carp(commands["run"])
    if restartMs is not None and not args.dry:
        os.replace(folder / "vm.igb", folder / "vm_fromRestart.igb")
        mergeIgb(prepace / "vm.igb", folder / "vm_fromRestart.igb", folder / "vm.igb")
        os.replace(folder / "act-thresh.dat", folder / "act_fromRestart.dat")
        mergeActivations(prepace / "act-thresh.dat", folder / "act_fromRestart.dat", folder / "act-thresh.dat", restartMs)
        for redundant in [folder / "vm_fromRestart.igb", prepace / "vm.igb", folder / "phie_i.igb"]:
            if redundant.exists():
                redundant.unlink()
        (folder / "restart.json").write_text(json.dumps(dict(
            restartMs=restartMs, phieStartMs=restartMs, vm="merged from t = 0", activations="merged from t = 0",
            removed=["vm_fromRestart.igb (merged)", "prepace/vm.igb (merged)", "phie_i.igb (= phie.igb at tissue nodes)"]),
            indent=2) + "\n")


if __name__ == "__main__":
    run()
