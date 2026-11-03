#!/usr/bin/env python3
"""One openCARP run of the production slab, as a standard carputils experiment.

Follows the carputils template (openCARP example 02_EP_tissue/00_simple): standard parser,
@tools.carpexample, mesh.Block, tools.carp_cmd, job.carp. Every value comes from the JSON run
spec written by efov.simulation.writeRunSpec; nothing scientific is set here.

Model
  * Tissue block size (Lx, Ly, wall) mm centred on the origin, fibres along +x (mesher default).
  * Bath: B mm of blood on every side (Block.set_bath both sides). mesher gives bath elements tag 0
    and zero fibres; openCARP recognises bath that way and applies gregion[].g_bath.
  * Lesion (condition "lesion"): tissue elements with y < boundary, tag 5, all six conductivities
    near zero (insulating scar). Viable tissue: tag 1.
  * Forward model: bidomain = 2 (pseudo-bidomain).
  * Ground: extracellular ground (stim type 3) on the nodes of a cube of side groundCubeMm at the
    (max x, max y, max z) bath corner; other outer walls no-flux (openCARP default).
  * Stimulus: transmembrane current (stim type 0) on viable tissue nodes with x <= xmin + strip
    width; nodes touching a lesion element are excluded.
  * Ionic model Courtemanche, initial state from bench conditioning; beta = cellSurfVolRatio.
  * Prepacing (restartMs): monodomain (bidomain = 0, no bath solve) to restartMs, state saved,
    then restarted with the forward model on. Vm and activation times are merged from t = 0;
    phie.igb starts at restartMs.
  * phi_e written only on the bath layer within phieOutputHeightMm of the tissue (dataout_e = 3).

Usage (normally called by efov.simulation.runCondition):
    python3 slab_experiment.py --spec <run>.json --np 24 --ID <output folder>
    python3 slab_experiment.py --spec <run>.json --dry      # print the openCARP commands only
"""
import json
import os
from pathlib import Path
import sys

import numpy as np
from carputils import mesh, tools

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from efov.io import mergeActivations, mergeIgb, readMesh, writeVertices  # noqa: E402

HERE = Path(__file__).resolve().parent
TISSUE_TAG, LESION_TAG, BATH_TAG = 1, 5, 0


def parser():
    result = tools.standard_parser()
    result.add_argument("--spec", required=True, help="JSON run spec written by efov.simulation")
    return result


def jobID(args):
    return str(Path(args.spec).with_suffix(""))


def geometry(spec):
    size = np.asarray(spec["tissueSizeMm"], float)
    block = mesh.Block(size=tuple(size), resolution=float(spec["spacingMm"]))
    bath = float(spec["bathThicknessMm"])
    block.set_bath(thickness=(bath, bath, bath), both_sides=True)
    if spec["condition"] == "lesion":
        lower = tuple(-size / 2)
        upper = (size[0] / 2, float(spec["boundaryYmm"]), size[2] / 2)    # tissue only
        block.add_region(mesh.BoxRegion(lower, upper, tag=LESION_TAG))
    return block


def nodeSets(meshBase, spec, folder):
    """Stimulus, ground and phi_e-output node files; returns counts for the run record."""
    positions, elements, tags, _ = readMesh(meshBase)
    isTissue = np.zeros(len(positions), bool)
    isTissue[np.unique(elements[tags != BATH_TAG])] = True
    stimulus = isTissue & (positions[:, 0] <= positions[isTissue, 0].min() + float(spec["stripWidthMm"]) + 1e-9)
    if np.any(tags == LESION_TAG):
        stimulus[np.unique(elements[tags == LESION_TAG])] = False
    size, bath = np.asarray(spec["tissueSizeMm"], float), float(spec["bathThicknessMm"])
    if not np.allclose(positions.max(axis=0), size / 2 + bath):
        raise ValueError(f"Mesh upper corner {positions.max(axis=0)} mm, expected {size / 2 + bath} mm")
    ground = np.all(positions >= positions.max(axis=0) - float(spec["groundCubeMm"]) - 1e-9, axis=1)
    if np.any(ground & isTissue):
        raise ValueError("Ground cube reaches tissue nodes")
    top = size[2] / 2
    output = np.flatnonzero((positions[:, 2] >= top - 1e-9) & (positions[:, 2] <= top + float(spec["phieOutputHeightMm"]) + 1e-9))
    writeVertices(folder / "stimulus.vtx", np.flatnonzero(stimulus))
    writeVertices(folder / "ground.vtx", np.flatnonzero(ground))
    writeVertices(folder / "phieOutput.vtx", output)
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
    # Intracellular domain: tissue and lesion. Extracellular domain: tissue, lesion and bath.
    options += ["-num_phys_regions", 2,
                "-phys_region[0].ptype", 0, "-phys_region[0].name", "intracellular",
                "-phys_region[0].num_IDs", 2, "-phys_region[0].ID[0]", TISSUE_TAG, "-phys_region[0].ID[1]", LESION_TAG,
                "-phys_region[1].ptype", 1, "-phys_region[1].name", "extracellular",
                "-phys_region[1].num_IDs", 3, "-phys_region[1].ID[0]", TISSUE_TAG,
                "-phys_region[1].ID[1]", LESION_TAG, "-phys_region[1].ID[2]", BATH_TAG]
    return options


def openCarpCommand(spec, meshBase, folder, simID, bidomain, endMs):
    cycle = float(spec["cycleLengthMs"])
    cmd = tools.carp_cmd(str(HERE / "solver.par"))
    cmd += ["-simID", str(simID), "-meshname", meshBase, "-bidomain", int(bidomain),
            "-dt", float(spec["timeStepUs"]), "-spacedt", float(spec["sampleStepMs"]), "-tend", float(endMs),
            "-imp_region[0].im_sv_init", str(spec["initialState"]),
            "-imp_region[0].cellSurfVolRatio", float(spec["cellSurfVolRatioPerUm"])]
    cmd += conductivityOptions(spec)
    cmd += ["-num_stim", 2,
            "-stim[0].crct.type", 0, "-stim[0].pulse.shape", 0, "-stim[0].ptcl.start", 0,
            "-stim[0].elec.vtx_file", str(folder / "stimulus"),
            "-stim[0].pulse.strength", float(spec["stimulusStrengthUAPerCm2"]),
            "-stim[0].ptcl.duration", float(spec["stimulusDurationMs"]),
            "-stim[0].ptcl.bcl", min(cycle, float(endMs)),
            "-stim[0].ptcl.npls", int(np.ceil(float(endMs) / cycle)),
            "-stim[1].crct.type", 3, "-stim[1].elec.vtx_file", str(folder / "ground")]
    return cmd


@tools.carpexample(parser, jobID)
def run(args, job):
    spec = json.loads(Path(args.spec).read_text())
    folder = Path(job.ID).resolve()
    folder.mkdir(parents=True, exist_ok=True)
    meshBase = mesh.generate(geometry(spec), rootdir=str(spec["meshRoot"]))
    # --dry prints the mesher command without running it, so there is no mesh to select nodes from.
    sets = {} if args.dry else nodeSets(meshBase, spec, folder)
    endMs = float(spec.get("durationMs") or spec["cycleLengthMs"] * spec["beatCount"])
    restartMs = spec.get("restartMs")
    commands = {}
    if restartMs is None:
        commands["run"] = openCarpCommand(spec, meshBase, folder, folder, 2, endMs)
    else:
        restartMs = float(restartMs)
        if not 0 < restartMs < endMs:
            raise ValueError("restartMs must lie inside the run")
        prepace = folder / "prepace"
        commands["prepace"] = openCarpCommand(spec, meshBase, folder, prepace, 0, restartMs) + [
            "-num_tsav", 1, "-tsav[0]", restartMs, "-write_statef", "state"]
        commands["run"] = openCarpCommand(spec, meshBase, folder, folder, 2, endMs) + [
            "-start_statef", str(prepace / f"state.{restartMs:.1f}")]
    commands["run"] += ["-dataout_e", 3, "-dataout_e_vtx", str(folder / "phieOutput")]
    (folder / "runRecord.json").write_text(json.dumps(dict(
        spec=spec, meshBase=str(meshBase), nodeSets=sets,
        commands={k: [str(c) for c in v] for k, v in commands.items()}), indent=2) + "\n")
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
            restartMs=restartMs, phieStartMs=restartMs, vm="merged from t = 0", activations="merged from t = 0"), indent=2) + "\n")


if __name__ == "__main__":
    run()
