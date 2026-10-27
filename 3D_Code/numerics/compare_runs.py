#!/usr/bin/env python3
"""Step 2 gate: does a parallel (MPI) openCARP run reproduce the serial run?

Compares two run folders produced by run_boundary.py from the same config:
  1. inputs   - mesh, stimulus, recovery points and conditioning state must be identical
                (otherwise any output difference is meaningless);
  2. activation times (act-thresh.dat, every activation event of every node);
  3. Vm (vm.igb, every node and frame);
  4. recovered extracellular potential (phie_recovery.igb) and its peak-to-peak per point.

Parallel PETSc solves can legitimately differ from serial in floating-point summation
order, so exact equality is not required. Tolerances below are PROPOSED (readiness plan,
Step 2) and are reported, never adjusted to make the check pass.

Usage (from the repository root):
    python3 3D_Code/numerics/compare_runs.py <serial run folder> <parallel run folder>
Writes <parallel>/mpiEquivalence.json and prints a table.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "Shared_Pipeline"))
import igb_io  # noqa: E402

TOLERANCES = {  # PROPOSED, readiness plan Step 2
    "activationMaxAbsMs": 0.02,        # one solver step (dt = 20 us)
    "vmMaxAbsMv": 0.1,
    "phiePeakToPeakMaxRelative": 1e-3,  # 0.1 %
}
INPUT_FILES = ["mesh/{c}.pts", "mesh/{c}.elem", "mesh/{c}.lon", "{c}Stimulus.vtx",
               "electrodes/recoveryPoints.pts", "conditioning/Courtemanche.sv"]


def fileHash(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def readActivations(path):
    """act-thresh.dat rows are (node index, activation time ms); sort to compare event sets."""
    rows = np.loadtxt(path, ndmin=2)
    order = np.lexsort((rows[:, 1], rows[:, 0]))
    return rows[order]


def maxAbsDifferenceIgb(pathA, pathB, block=500):
    """Stream both files in time blocks; return max |A-B|, max |A| and per-point peak-to-peak."""
    _, nodesA, framesA, _ = igb_io.header(pathA)
    _, nodesB, framesB, _ = igb_io.header(pathB)
    if (nodesA, framesA) != (nodesB, framesB):
        raise ValueError(f"{pathA.name}: shapes differ ({nodesA}x{framesA} vs {nodesB}x{framesB})")
    worst, scale = 0.0, 0.0
    lowA = np.full(nodesA, np.inf); highA = -lowA.copy()
    lowB = lowA.copy(); highB = highA.copy()
    for first in range(0, framesA, block):
        frames = slice(first, min(first + block, framesA))
        a = igb_io.read(pathA, frames); b = igb_io.read(pathB, frames)
        worst = max(worst, float(np.max(np.abs(a - b))))
        scale = max(scale, float(np.max(np.abs(a))))
        lowA = np.minimum(lowA, a.min(0)); highA = np.maximum(highA, a.max(0))
        lowB = np.minimum(lowB, b.min(0)); highB = np.maximum(highB, b.max(0))
    return worst, scale, highA - lowA, highB - lowB


def compareCondition(serial, parallel, condition):
    result = {"condition": condition}
    mismatched = [name.format(c=condition) for name in INPUT_FILES
                  if fileHash(serial / name.format(c=condition)) != fileHash(parallel / name.format(c=condition))]
    result["inputsIdentical"] = not mismatched
    result["mismatchedInputs"] = mismatched
    a = readActivations(serial / condition / "act-thresh.dat")
    b = readActivations(parallel / condition / "act-thresh.dat")
    result["activationEventsSerial"], result["activationEventsParallel"] = len(a), len(b)
    sameEvents = a.shape == b.shape and np.array_equal(a[:, 0], b[:, 0])
    result["activationMaxAbsMs"] = float(np.max(np.abs(a[:, 1] - b[:, 1]))) if sameEvents else None
    vmWorst, _, _, _ = maxAbsDifferenceIgb(serial / condition / "vm.igb", parallel / condition / "vm.igb")
    result["vmMaxAbsMv"] = vmWorst
    phieWorst, phieScale, ppA, ppB = maxAbsDifferenceIgb(serial / condition / "phie_recovery.igb",
                                                         parallel / condition / "phie_recovery.igb")
    usable = ppA > 1e-6
    result["phieMaxAbsMv"] = phieWorst
    result["phieMaxAbsRelativeToRange"] = phieWorst / max(phieScale, 1e-12)
    result["phiePeakToPeakMaxRelative"] = float(np.max(np.abs(ppB[usable] - ppA[usable]) / ppA[usable]))
    result["passed"] = bool(
        result["inputsIdentical"] and sameEvents
        and result["activationMaxAbsMs"] <= TOLERANCES["activationMaxAbsMs"]
        and vmWorst <= TOLERANCES["vmMaxAbsMv"]
        and result["phiePeakToPeakMaxRelative"] <= TOLERANCES["phiePeakToPeakMaxRelative"])
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("serial", type=Path)
    parser.add_argument("parallel", type=Path)
    arguments = parser.parse_args()
    serial, parallel = arguments.serial.resolve(), arguments.parallel.resolve()
    meta = {name: json.loads((folder / "intact" / "meta.json").read_text())["processCount"]
            for name, folder in [("serial", serial), ("parallel", parallel)]}
    if meta["serial"] != 1 or meta["parallel"] < 2:
        raise SystemExit(f"Expected a 1-process and a multi-process run, got {meta}")
    results = [compareCondition(serial, parallel, c) for c in ["intact", "lesion"]]
    summary = dict(serial=str(serial), parallel=str(parallel), processCounts=meta,
                   tolerancesProposed=TOLERANCES, conditions=results,
                   passed=all(r["passed"] for r in results))
    (parallel / "mpiEquivalence.json").write_text(json.dumps(summary, indent=2) + "\n")
    for r in results:
        print(f"{r['condition']:7s} inputs identical={r['inputsIdentical']}  "
              f"LAT max|d|={r['activationMaxAbsMs']} ms  Vm max|d|={r['vmMaxAbsMv']:.3g} mV  "
              f"phie PP max rel d={r['phiePeakToPeakMaxRelative']:.3g}  -> {'PASS' if r['passed'] else 'FAIL'}")
    print("Overall:", "PASS" if summary["passed"] else "FAIL", f"(tolerances proposed: {TOLERANCES})")
    return 0 if summary["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
