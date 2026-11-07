#!/usr/bin/env python3
"""Cost of openCARP's lead-field module (openCARP >= 20; FM-35) at the production mesh spacing, by domain size.

Builds the production slab geometry (mesher, 0.25 mm in-plane, 0.1 mm transmural, 5 mm bath, no-flux
epicardium, bilayer fibres) on reduced in-plane extents and times one openCARP lead-field run with --leads
electrode-sized node sets above the tissue (and a two-frame dummy Vm), at --np MPI ranks. The time per lead
(after subtracting a one-lead run's start-up) scales the production estimate. A diagnostic for planning.

    python3 checks/leadfield_cost.py --out runs/leadfieldCost --sizes 10x15 20x30 --leads 8 --np 24
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
from efov.io import readMesh, writeIgb, writeMesh  # noqa: E402

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
    import resource
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", required=True)
    parser.add_argument("--sizes", nargs="+", default=["10x15", "20x30"], help="in-plane tissue sizes, mm")
    parser.add_argument("--leads", type=int, default=8)
    parser.add_argument("--np", type=int, default=1)
    args = parser.parse_args()
    if not leadfield.supportsLeadfield():
        raise SystemExit("This openCARP has no lead-field module: openCARP >= 20 is required")
    config = cfg.loadConfig()
    wall = cfg.slabSizeMm(config)[2]
    rows = []
    for text in args.sizes:
        size = [float(v) for v in text.split("x")]
        folder = Path(args.out).resolve() / text
        folder.mkdir(parents=True, exist_ok=True)
        tissue, ground = buildMesh(config, size, folder)
        p = tissue["positions"]
        meshBase = writeMesh(folder / "openCARP" / "block", p, tissue["elements"], tissue["tags"], tissue["fibres"])
        writeIgb(folder / "vm.igb", np.full((2, len(tissue["tissueNodes"])), -80.0), 0.0, 0.2)
        leads = []
        for k in range(args.leads):
            centre = np.array([(k % 4 - 1.5) * 2.0, (k // 4) * 2.0, wall / 2 + 0.25 + 0.5])
            nodes = np.flatnonzero((np.hypot(p[:, 0] - centre[0], p[:, 1] - centre[1]) <= 0.5) & (np.abs(p[:, 2] - centre[2]) <= 0.5))
            leads.append((leadfield.leadName(nodes), nodes))
        timing = {}
        for label, subset in (("one", leads[:1]), ("all", leads)):
            started = time.time()
            leadfield.runOpenCarpLeadfield(config, meshBase, ground, subset, folder / "vm.igb", folder / f"run_{label}", 0.2, args.np)
            timing[label] = time.time() - started
        perLead = (timing["all"] - timing["one"]) / max(1, len(leads) - 1)
        rows.append(dict(sizeMm=size, nodes=int(len(p)), np=args.np, startupSeconds=timing["one"] - perLead,
                         secondsPerLead=perLead, peakMemoryGB=resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss / 1e6))
        print(json.dumps(rows[-1]), flush=True)
    production = cfg.meshNodeCount(config)
    if len(rows) >= 2:
        n = np.log([r["nodes"] for r in rows]); t = np.log([max(r["secondsPerLead"], 1e-3) for r in rows])
        exponent = float(np.polyfit(n, t, 1)[0])
        perLead = rows[-1]["secondsPerLead"] * (production / rows[-1]["nodes"]) ** exponent
        summary = dict(productionNodes=production, timeExponent=exponent, secondsPerLeadProductionEstimate=float(perLead),
                       hoursPerCondition712=float(perLead * 712 / 3600), peakMemoryGBProductionEstimate=float(rows[-1]["peakMemoryGB"] * production / rows[-1]["nodes"]),
                       note=f"At {args.np} MPI rank(s); start-up (mesh reading, assembly) is paid once per condition run.")
        print(json.dumps(summary))
        rows.append(summary)
    (Path(args.out) / "leadfieldCost.json").write_text(json.dumps(rows, indent=2) + "\n")


if __name__ == "__main__":
    main()
