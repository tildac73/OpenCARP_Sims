#!/usr/bin/env python3
"""One openCARP run of the slab (or the tuning strip), as a standard carputils experiment.

Follows the carputils template (openCARP example 02_EP_tissue/00_simple): standard parser,
@tools.carpexample, mesh.Block, mesh.generate, tools.carp_cmd, job.carp. Every value comes from the
JSON run spec written by efov.simulation and from config/model.par; nothing scientific is set here.

Model (plan FM-01, FM-04 to FM-08, FM-17, FM-23, FM-24)
  * Block mesh, x along the reference wave, y across the lesion boundary, z through the wall,
    centred on the origin; 0.25 mm in-plane and 0.1 mm transmural resolution (the mesher's
    per-axis -resolution[2], set by AnisotropicBlock; carputils Block writes one value for all axes).
  * Bath on every side from carputils; with epicardialMedium 'none' the bath below the tissue is
    removed (efov.meshbuild.runMesh), leaving a no-flux epicardium. The monodomain solve ignores
    the bath; the mesh carries it for the lead-field runs (FM-23).
  * Lesion (condition 'lesion'): transmural strip boundary - lesionWidth <= y < boundary, tag 5;
    tissue beyond it stays viable (tag 1) but is isolated and never stimulated (FM-24).
  * Fibres: one vector per tissue element from the fibre model (efov.fibres); bath vectors zero.
  * Stimulus: transmembrane current on the viable-tissue band at the trailing edge of the wave.
  * Ground: vertex file of the bath corner cube, written here and used by the lead-field runs.
  * Monodomain throughout (model.par bidomain = 0), one launch, Vm and activation times only.
  * The launch reads one resolved parameter file (<ID>/run.par): model.par with the run's
    overrides in place, then the run-specific paths.

The 'strip' kind is the conduction-velocity tuning strip (FM-18): same resolution and model.par,
no bath, no lesion, fibres along x.

Usage (normally called by efov.simulation):
    python3 slab_experiment.py --spec <run>.json --np 24 --ID <output folder>
    python3 slab_experiment.py --spec <run>.json --dry      # print the commands only
"""
import json
import math
from pathlib import Path
import sys

import numpy as np
from carputils import mesh, tools

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from efov import meshbuild  # noqa: E402
from efov.io import readMesh, writeMesh, writeVertices  # noqa: E402
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
            boundary, width = float(spec["boundaryYmm"]), float(spec["lesionWidthMm"])
            block.add_region(mesh.BoxRegion((-size[0] / 2, boundary - width, -size[2] / 2), (size[0] / 2, boundary, size[2] / 2),
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
    """Stimulus and ground vertex files; counts for the run record."""
    isolated = float(spec["boundaryYmm"]) if spec["kind"] == "slab" and spec["condition"] == "lesion" else None
    stimulus = meshbuild.stimulusNodes(positions, elements, tags, float(spec["waveAngleDeg"]), float(spec["stripWidthMm"]), isolated)
    ground = meshbuild.groundNodes(positions, elements, tags, float(spec["groundCubeMm"]), allowTissue=spec["kind"] == "strip")
    writeVertices(folder / "stimulus.vtx", stimulus)
    writeVertices(folder / "ground.vtx", ground)
    counts = dict(nodes=len(positions), elements=len(elements), tissueNodes=int(meshbuild.tissueNodeMask(len(positions), elements, tags).sum()),
                  stimulusNodes=len(stimulus), groundNodes=len(ground), lesionElements=int((tags == meshbuild.LESION_TAG).sum()),
                  bathElements=int((tags == meshbuild.BATH_TAG).sum()))
    return counts


def runLines(spec, meshBase, folder, simID):
    """Run-specific openCARP parameters (paths and run controls), written after model.par in the resolved file."""
    return {"meshname": meshBase, "simID": simID, "imp_region[0].im_sv_init": spec["initialState"],
            "stim[0].elec.vtx_file": str(folder / "stimulus"), "stim[1].elec.vtx_file": str(folder / "ground")}


def overridesFor(spec, endMs):
    """model.par overrides of the launch: the sweep/tuning overrides plus the duration."""
    model = readPar(spec["modelPar"])
    cycle = float(model["stim[0].ptcl.bcl"])
    result = dict(spec["parOverrides"])
    result.update({"tend": float(endMs), "stim[0].ptcl.npls": int(math.ceil(endMs / cycle - 1e-9))})
    if endMs < cycle:
        # Short verification runs (one pulse): openCARP rejects bcl > tend ("above the maximum").
        # bcl = tend with one pulse gives the same single stimulus at ptcl.start.
        result["stim[0].ptcl.bcl"] = float(endMs)
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
    if args.dry:                       # --dry prints the mesher command; there is no mesh to build node sets from
        print(json.dumps(dict(spec=spec, endMs=endMs), indent=2))
        return
    meshBase, positions, elements, tags = buildRunMesh(blockBase, spec, folder)
    sets = nodeSets(positions, elements, tags, spec, folder)
    commands = {}

    def record():
        (folder / "runRecord.json").write_text(json.dumps(dict(spec=spec, blockMesh=str(blockBase), meshBase=str(meshBase),
                                                               nodeSets=sets, commands=commands), indent=2) + "\n")
    record()                           # written before openCARP starts, so a failed run is still documented
    commands["run"] = launch(job, spec, folder, "run", overridesFor(spec, endMs), runLines(spec, meshBase, folder, str(folder)))
    record()


if __name__ == "__main__":
    run()
