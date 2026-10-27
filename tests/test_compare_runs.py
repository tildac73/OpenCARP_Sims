"""Synthetic checks for 3D_Code/numerics/compare_runs.py (Step 2 MPI-equivalence gate).

Builds two miniature run folders with identical inputs and controlled output differences,
and confirms the comparison passes inside tolerance and fails outside it.
Run: python3 -m unittest tests.test_compare_runs -v
"""
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("compare_runs", ROOT / "3D_Code/numerics/compare_runs.py")
compare = importlib.util.module_from_spec(spec); spec.loader.exec_module(compare)


def writeIgb(path, values):
    values = np.asarray(values, dtype="<f4")
    frames, nodes = values.shape
    header = f"x:{nodes} y:1 z:1 t:{frames} type:float systeme:little_endian org_t:0 inc_t:0.2 "
    with open(path, "wb") as stream:
        stream.write(header.encode().ljust(1023, b" ") + b"\x0c")
        stream.write(values.tobytes())


def makeRun(folder, processCount, vmOffset=0.0, phieScale=1.0, latShift=0.0):
    rng = np.random.default_rng(0)
    for condition in ["intact", "lesion"]:
        (folder / condition).mkdir(parents=True)
        (folder / condition / "meta.json").write_text(json.dumps({"processCount": processCount}))
        vm = -80 + 100 * rng.random((50, 30)); vm[10:] += vmOffset
        writeIgb(folder / condition / "vm.igb", vm)
        phie = np.sin(np.linspace(0, 6, 50))[:, None] * (1 + np.arange(12))[None, :]
        writeIgb(folder / condition / "phie_recovery.igb", phie * phieScale)
        lat = np.column_stack([np.arange(30), 5 + 0.1 * np.arange(30) + latShift])
        np.savetxt(folder / condition / "act-thresh.dat", lat, fmt="%d\t%.6f")
        for name in compare.INPUT_FILES:
            path = folder / name.format(c=condition); path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(f"identical input {name}\n")


class CompareRunsTests(unittest.TestCase):
    def check(self, **perturbation):
        with tempfile.TemporaryDirectory() as root:
            serial, parallel = Path(root) / "s", Path(root) / "p"
            makeRun(serial, 1); makeRun(parallel, 8, **perturbation)
            return [compare.compareCondition(serial, parallel, c) for c in ["intact", "lesion"]]

    def testIdenticalRunsPass(self):
        results = self.check()
        self.assertTrue(all(r["passed"] for r in results))
        self.assertEqual(results[0]["vmMaxAbsMv"], 0.0)

    def testSmallDifferencesInsideTolerancePass(self):
        self.assertTrue(all(r["passed"] for r in self.check(vmOffset=0.05, latShift=0.01, phieScale=1.0005)))

    def testVmDifferenceFails(self):
        results = self.check(vmOffset=0.5)
        self.assertFalse(results[0]["passed"]); self.assertAlmostEqual(results[0]["vmMaxAbsMv"], 0.5, places=4)

    def testActivationShiftFails(self):
        self.assertFalse(self.check(latShift=0.1)[0]["passed"])

    def testPhieAmplitudeFails(self):
        self.assertFalse(self.check(phieScale=1.01)[0]["passed"])

    def testDifferentInputsFail(self):
        with tempfile.TemporaryDirectory() as root:
            serial, parallel = Path(root) / "s", Path(root) / "p"
            makeRun(serial, 1); makeRun(parallel, 8)
            (parallel / "mesh/intact.pts").write_text("different mesh\n")
            result = compare.compareCondition(serial, parallel, "intact")
            self.assertFalse(result["passed"]); self.assertEqual(result["mismatchedInputs"], ["mesh/intact.pts"])


if __name__ == "__main__":
    unittest.main()
