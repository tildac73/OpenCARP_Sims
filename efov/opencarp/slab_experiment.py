#!/usr/bin/env python3
"""One openCARP run of the slab (or the tuning strip), as a standard carputils experiment.

Follows the carputils template (openCARP example 02_EP_tissue/00_simple): standard parser,
@tools.carpexample, mesh.Block, mesh.generate, tools.carp_cmd, job.carp. Every value comes from the
JSON run spec written by efov.simulation and from config/model.par; nothing scientific is set here.

Model (plan FM-01, FM-04 to FM-08, FM-17, FM-20)
  * Block mesh, x along the reference wave, y across the lesion boundary, z through the wall,
    centred on the origin; 0.25 mm in-plane and 0.1 mm transmural resolution (the mesher's
    per-axis -resolution[2], set by AnisotropicBlock; carputils Block writes one value for all axes).
  * Bath on every side from carputils; with epicardialMedium 'none' the bath below the tissue is
    removed (efov.meshbuild.runMesh), leaving a no-flux epicardium.
  * Lesion (condition 'lesion'): tissue with y < boundary, tag 5; conductivities from model.par.
  * Fibres: one vector per tissue element from the fibre model (efov.fibres); bath vectors zero.
  * Stimulus: transmembrane current on the viable-tissue band at the trailing edge of the wave.
  * Ground: extracellular ground on the bath nodes of a 1 mm cube at the (max x, max y, max z) corner.
  * Prepacing (restartMs): beats before restartMs monodomain, state saved, restart in pseudo-bidomain.
  * phi_e written only on the nodes of the electrode envelope box (dataout_e = 3).
  * Each openCARP launch reads one resolved parameter file (<ID>/<step>.par): model.par with the
    run's overrides in place, then the run-specific paths.

The 'strip' kind is the conduction-velocity tuning strip (FM-18): same resolution and model.par,
no bath, no lesion, fibres along x, monodomain only.

Usage (normally called by efov.simulation):
    python3 slab_experiment.py --spec <run>.json --np 24 --ID <output folder>
    python3 slab_experiment.py --spec <run>.json --dry      # print the commands only
"""
import json
import math
import os
from pathlib import Path
import sys

import numpy as np
from carputils import mesh, tools

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from efov import meshbuild  # noqa: E402
from efov.io import mergeActivations, mergeIgb, readMesh, writeMesh, writeVertices  # noqa: E402
from efov.parfile import readPar, writeResolved  # noqa: E402


class AnisotropicBlock(mesh.Block):
    """carputils Block whose mesher call uses a separate transmural (z) resolution."""

    def __init__(self, transmuralResolutionMm, **kwargs):
        super().__init__(**kwargs)
        self._transmuralUm = float(transmuralResolutionMm) * 1000.0

    def mesher_opts(self, meshname):
        options = list(super().mesher_opts(meshname))
        options[options.index("-resolution[2]") + 1] = self._transmuralUm
        return options


def parser():
    result = tools.standard_parser()
    result.add_argument("--spec", required=True, help="JSON run spec written by efov.simulation")
    return result


def jobID(args):
    return str(Path(args.spec).with_suffix(""))


def blockGeometry(spec):
    size = np.asarray(spec["tissueSizeMm"], float)
    spacing = np.asarray(spec["spacingMm"], float)
    if not np.isclose(spacing[0], spacing[1]):
        raise ValueError("In-plane spacing must be equal in x and y")
    block = AnisotropicBlock(spacing[2], size=tuple(size), resolution=float(spacing[0]))
    if spec["kind"] == "slab":
        bath = float(spec["bathThicknessMm"])
        block.set_bath(thickness=(bath, bath, bath), both_sides=True)
        if spec["condition"] == "lesion":
            block.add_region(mesh.BoxRegion(tuple(-size / 2), (size[0] / 2, float(spec["boundaryYmm"]), size[2] / 2),
                                            tag=meshbuild.LESION_TAG))
    return block


def buildRunMesh(blockBase, spec, folder):
    """Run mesh (trimmed bath, fibres) in <folder>/mesh/slab; returns its base name and positions etc."""
    positions, elements, tags, _ = readMesh(blockBase)
    fibres = spec["fibres"] if spec["kind"] == "slab" else {"model": "uniform", "angleDeg": 0.0}
    medium = spec["epicardialMedium"] if spec["kind"] == "slab" else "blood"
    positions, elements, tags, vectors = meshbuild.runMesh(positions, elements, tags, float(spec["tissueSizeMm"][2]), medium, fibres)
    base = writeMesh(folder / "mesh" / "slab", positions, elements, tags, vectors)
    return base, positions, elements, tags


def nodeSets(positions, elements, tags, spec, folder):
    """Stimulus, ground and phi_e-output vertex files; counts for the run record."""
    stimulus = meshbuild.stimulusNodes(positions, elements, tags, float(spec["waveAngleDeg"]), float(spec["stripWidthMm"]))
    ground = meshbuild.groundNodes(positions, elements, tags, float(spec["groundCubeMm"]), allowTissue=spec["kind"] == "strip")
    writeVertices(folder / "stimulus.vtx", stimulus)
    writeVertices(folder / "ground.vtx", ground)
    counts = dict(nodes=len(positions), elements=len(elements), tissueNodes=int(meshbuild.tissueNodeMask(len(positions), elements, tags).sum()),
                  stimulusNodes=len(stimulus), groundNodes=len(ground), lesionElements=int((tags == meshbuild.LESION_TAG).sum()),
                  bathElements=int((tags == meshbuild.BATH_TAG).sum()))
    if spec["kind"] == "slab":
        lower, upper = (np.asarray(c, float) for c in spec["phieBoxMm"])
        output = meshbuild.boxNodes(positions, lower, upper)
        if not len(output):
            raise ValueError("phi_e output box holds no node")
        writeVertices(folder / "phieOutput.vtx", output)
        np.save(folder / "phieNodes.npy", output)
        counts["phieNodes"] = len(output)
    return counts


def runLines(spec, meshBase, folder, simID):
    """Run-specific openCARP parameters (paths and run controls), written after model.par in the resolved file."""
    lines = {"meshname": meshBase, "simID": simID, "imp_region[0].im_sv_init": spec["initialState"],
             "stim[0].elec.vtx_file": str(folder / "stimulus"), "stim[1].elec.vtx_file": str(folder / "ground")}
    if spec["kind"] == "slab":
        lines["dataout_e_vtx"] = str(folder / "phieOutput")
    return lines


def overridesFor(spec, bidomain, endMs):
    """model.par overrides of one launch: the sweep/tuning overrides plus duration and formulation."""
    model = readPar(spec["modelPar"])
    cycle = float(model["stim[0].ptcl.bcl"])
    result = dict(spec["parOverrides"])
    result.update({"bidomain": int(bidomain), "tend": float(endMs), "stim[0].ptcl.npls": int(math.ceil(endMs / cycle - 1e-9))})
    if spec["kind"] == "strip":
        result["dataout_e"] = 0                 # no bath and no phi_e on the tuning strip
    return result


def launch(job, spec, folder, name, overrides, lines):
    parPath = writeResolved(spec["modelPar"], overrides, lines, folder / f"{name}.par",
                            header=f"# Resolved openCARP parameters of '{name}' (efov; carputils adds only the solver-flavour options)")
    command = tools.carp_cmd(str(parPath))
    job.carp(command)
    return [str(c) for c in command]


@tools.carpexample(parser, jobID)
def run(args, job):
    spec = json.loads(Path(args.spec).read_text())
    folder = Path(job.ID).resolve()
    folder.mkdir(parents=True, exist_ok=True)
    blockBase = mesh.generate(blockGeometry(spec), rootdir=str(spec["meshRoot"]))
    model = readPar(spec["modelPar"])
    endMs = float(spec.get("durationMs") or model["tend"])
    restartMs = None if spec["kind"] == "strip" else spec.get("restartMs")
    if args.dry:                       # --dry prints the mesher command; there is no mesh to build node sets from
        print(json.dumps(dict(spec=spec, endMs=endMs, restartMs=restartMs), indent=2))
        return
    meshBase, positions, elements, tags = buildRunMesh(blockBase, spec, folder)
    sets = nodeSets(positions, elements, tags, spec, folder)
    commands = {}

    def record():
        (folder / "runRecord.json").write_text(json.dumps(dict(spec=spec, blockMesh=str(blockBase), meshBase=str(meshBase),
                                                               nodeSets=sets, commands=commands), indent=2) + "\n")
    record()                           # written before openCARP starts, so a failed run is still documented
    finalBidomain = 0 if spec["kind"] == "strip" else int(model["bidomain"])
    if restartMs is None:
        commands["run"] = launch(job, spec, folder, "run", overridesFor(spec, finalBidomain, endMs),
                                 runLines(spec, meshBase, folder, str(folder)))
    else:
        restartMs = float(restartMs)
        if not 0 < restartMs < endMs:
            raise ValueError("restartMs must lie inside the run")
        prepace = folder / "prepace"
        lines = runLines(spec, meshBase, folder, str(prepace))
        lines.update({"num_tsav": 1, "tsav[0]": restartMs, "write_statef": "state"})
        commands["prepace"] = launch(job, spec, folder, "prepace", overridesFor(spec, 0, restartMs), lines)
        lines = runLines(spec, meshBase, folder, str(folder))
        lines["start_statef"] = str(prepace / f"state.{restartMs:.1f}")
        commands["run"] = launch(job, spec, folder, "run", overridesFor(spec, finalBidomain, endMs), lines)
        os.replace(folder / "vm.igb", folder / "vm_fromRestart.igb")
        mergeIgb(prepace / "vm.igb", folder / "vm_fromRestart.igb", folder / "vm.igb")
        os.replace(folder / "act-thresh.dat", folder / "act_fromRestart.dat")
        mergeActivations(prepace / "act-thresh.dat", folder / "act_fromRestart.dat", folder / "act-thresh.dat", restartMs)
        for redundant in [folder / "vm_fromRestart.igb", prepace / "vm.igb", folder / "phie_i.igb"]:
            if redundant.exists():
                redundant.unlink()
        (folder / "restart.json").write_text(json.dumps(dict(restartMs=restartMs, phieStartMs=restartMs,
                                                             vm="merged from t = 0", activations="merged from t = 0"), indent=2) + "\n")
    record()


if __name__ == "__main__":
    run()
