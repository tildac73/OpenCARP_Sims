#!/usr/bin/env python3
"""Cost of the Python lead-field route (FM-14, FM-31) at the production mesh spacing, by domain size.

Builds the production slab geometry (mesher, 0.25 mm in-plane, 0.1 mm transmural, 5 mm bath, no-flux
epicardium, bilayer fibres) on reduced in-plane extents, sets up the solver exactly as the pipeline does
(pyamg smoothed aggregation, CG, rtol 1e-10) and times the set-up and a few lead-field solves at
electrode-sized node sets above the tissue. A diagnostic for planning, not a model check.

    python3 checks/leadfield_cost.py --out runs/leadfieldCost --sizes 10x15 20x30
"""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import time

import numpy as np

REPOSITORY = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY))
from efov import config as cfg, leadfield, meshbuild  # noqa: E402
from efov.io import readMesh  # noqa: E402

sys.path.insert(0, str(REPOSITORY / "efov" / "opencarp"))
from slab_experiment import AnisotropicBlock  # noqa: E402


def buildMesh(config, sizeMm, folder):
    spacing = cfg.spacingMm(config)
    wall = cfg.slabSizeMm(config)[2]
    bath = config["fixed"]["bath"]["thicknessMm"]
    block = AnisotropicBlock(spacing[2], size=(sizeMm[0], sizeMm[1], wall), resolution=float(spacing[0]))
    block.set_bath(thickness=(bath, bath, bath), both_sides=True)
    base = str(folder / "block")
    subprocess.run(["mesher"] + [str(o) for o in block.mesher_opts(base)], check=True, capture_output=True)
    positions, elements, tags, _ = readMesh(base)
    positions, elements, tags, fibres = meshbuild.runMesh(positions, elements, tags, wall, config["values"]["epicardialMedium"],
                                                          config["values"]["fibres"])
    nodes = meshbuild.tissueNodeMask(len(positions), elements, tags)
    tissue = dict(positions=positions, elements=elements, tags=tags, fibres=fibres, tissueNodes=np.flatnonzero(nodes))
    ground = meshbuild.groundNodes(positions, elements, tags, config["fixed"]["bath"]["groundCubeMm"])
    return tissue, ground


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", required=True)
    parser.add_argument("--sizes", nargs="+", default=["10x15", "20x30"], help="in-plane tissue sizes, mm")
    parser.add_argument("--solves", type=int, default=3)
    args = parser.parse_args()
    config = cfg.loadConfig()
    wall = cfg.slabSizeMm(config)[2]
    rows = []
    for text in args.sizes:
        size = [float(v) for v in text.split("x")]
        folder = Path(args.out).resolve() / text
        folder.mkdir(parents=True, exist_ok=True)
        tissue, ground = buildMesh(config, size, folder)
        p = tissue["positions"]
        started = time.time()
        solver = leadfield.PythonLeadFieldSolver(config, tissue, ground)
        setup = time.time() - started
        seconds = []
        for k in range(args.solves):
            centre = np.array([(k - 1) * 2.0, 0.0, wall / 2 + 0.25 + 0.5])
            nodes = np.flatnonzero((np.hypot(p[:, 0] - centre[0], p[:, 1] - centre[1]) <= 0.5) & (np.abs(p[:, 2] - centre[2]) <= 0.5))
            started = time.time()
            z = solver.leadField(nodes)
            seconds.append(time.time() - started)
            assert np.isfinite(z).all() and z[nodes].mean() > 0
        rows.append(dict(sizeMm=size, nodes=int(len(p)), unknowns=int(len(solver.free)), setupSeconds=setup,
                         secondsPerSolve=float(np.median(seconds)), allSolveSeconds=seconds))
        print(json.dumps(rows[-1]), flush=True)
        del solver
    production = cfg.meshNodeCount(config)
    if len(rows) >= 2:
        n = np.log([r["nodes"] for r in rows]); t = np.log([r["secondsPerSolve"] for r in rows])
        exponent = float(np.polyfit(n, t, 1)[0])
        perSolve = rows[-1]["secondsPerSolve"] * (production / rows[-1]["nodes"]) ** exponent
        summary = dict(productionNodes=production, timeExponent=exponent, secondsPerSolveProductionEstimate=float(perSolve),
                       coreHoursPerCondition712=float(perSolve * 712 / 3600))
        print(json.dumps(summary))
        rows.append(summary)
    (Path(args.out) / "leadfieldCost.json").write_text(json.dumps(rows, indent=2) + "\n")


if __name__ == "__main__":
    main()
