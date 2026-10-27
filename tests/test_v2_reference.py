"""Regression test: the v2-pilot results must reproduce from the raw simulations.

What is checked (against tests/reference/v2_pilot, see make_v2_reference.py):
  * peak-to-peak voltage and selection weights for all 4,160 recordings,
  * selected clinical-style change-point fits (180 per window),
  * FOV80 contribution radii for the 48 ground-truth recordings (slow; opt-in).
Both the original 0-400 ms window and the declared 25-100 ms window are checked.

Requires the raw run folder "boundaryPilotV2 2" (Vm, egm.npz, meshes); tests are
skipped where it is absent. No simulation is rerun; analysis runs through
_refinement_work/run_window.py into a temporary folder, so stored results are
never overwritten.

Run from the repository root:
    python3 -m unittest tests.test_v2_reference -v
    EFOV_RUN_SLOW=1 python3 -m unittest tests.test_v2_reference -v   # + contributions
"""
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW_RUN = ROOT / "boundaryPilotV2 2"
DRIVER = ROOT / "_refinement_work" / "run_window.py"
REFERENCE = Path(__file__).resolve().parent / "reference" / "v2_pilot"
sys.path.insert(0, str(REFERENCE.parent))
from make_v2_reference import CHANGE_POINT_KEYS, weightHash  # noqa: E402

RAW_AVAILABLE = (RAW_RUN / "intact" / "egm.npz").exists() and (RAW_RUN / "lesion" / "egm.npz").exists()
RUN_SLOW = os.environ.get("EFOV_RUN_SLOW") == "1"
# PP is stored as float32 in the trace archives; the original run reproduced to 0.0 mV.
PP_RTOL, PP_ATOL = 1e-6, 1e-9
CHANGE_POINT_ATOL_MM = 1e-9  # grid search on a 0.05 mm grid: any change is a real change


def runDriver(stage, out, window):
    command = [sys.executable, str(DRIVER), stage, "--out", str(out),
               "--window", str(window[0]), str(window[1])]
    result = subprocess.run(command, cwd=ROOT, capture_output=True, text=True)
    if result.returncode != 0:
        raise AssertionError(f"{' '.join(command)} failed:\n{result.stdout}\n{result.stderr}")
    return result


@unittest.skipUnless(RAW_AVAILABLE, "raw run folder 'boundaryPilotV2 2' not present")
class WindowReproduction:
    """Shared checks; subclasses set the window and reference name."""
    window = None
    name = None

    @classmethod
    def setUpClass(cls):
        cls.temporary = tempfile.TemporaryDirectory(prefix=f"efov_{cls.name}_")
        cls.out = Path(cls.temporary.name) / "run"
        runDriver("measure", cls.out, cls.window)
        runDriver("changepoints-only", cls.out, cls.window)

    @classmethod
    def tearDownClass(cls):
        cls.temporary.cleanup()

    def testPeakToPeakAndSelectionWeights(self):
        reference = pd.read_csv(REFERENCE / f"points_{self.name}.csv")
        produced = pd.concat([pd.read_csv(self.out / "results" / f"{c}Points.csv")
                              for c in ["intact", "lesion"]], ignore_index=True)
        produced["weightHash"] = produced.apply(weightHash, axis=1)
        merged = reference.merge(produced[["recordingId", "ppMv", "weightHash"]],
                                 on="recordingId", suffixes=("Ref", "New"), validate="one_to_one")
        self.assertEqual(len(merged), len(reference), "recording set changed")
        np.testing.assert_allclose(merged.ppMvNew, merged.ppMvRef, rtol=PP_RTOL, atol=PP_ATOL)
        changed = merged[merged.weightHashRef != merged.weightHashNew]
        self.assertTrue(changed.empty, f"selection weights changed for {len(changed)} recordings, "
                                       f"e.g. {changed.recordingId.head(3).tolist()}")

    def testSelectedChangePoints(self):
        reference = pd.read_csv(REFERENCE / f"changepoints_{self.name}.csv")
        produced = pd.read_csv(self.out / "results" / "clinical" / "changePointFits.csv")
        produced = produced[produced.selected == True]  # noqa: E712
        merged = reference.merge(produced[CHANGE_POINT_KEYS + ["changePointMm", "fitStatus"]],
                                 on=CHANGE_POINT_KEYS, suffixes=("Ref", "New"), validate="one_to_one")
        self.assertEqual(len(merged), len(reference), "set of fitted profiles changed")
        np.testing.assert_allclose(merged.changePointMmNew, merged.changePointMmRef,
                                   rtol=0, atol=CHANGE_POINT_ATOL_MM)
        statusChanged = merged[merged.fitStatusRef != merged.fitStatusNew]
        self.assertTrue(statusChanged.empty, f"fit status changed:\n{statusChanged}")


class OriginalWindow(WindowReproduction, unittest.TestCase):
    window, name = (0, 400), "window0_400"


class DeclaredWindow(WindowReproduction, unittest.TestCase):
    window, name = (25, 100), "window25_100"


@unittest.skipUnless(RAW_AVAILABLE and RUN_SLOW, "set EFOV_RUN_SLOW=1 with the raw run present")
class DeclaredWindowContributions(unittest.TestCase):
    """Recomputes all 48 FOV80 radii in the 25-100 ms window (reads Vm; minutes)."""

    def testFov80AndReconstructionAgreement(self):
        window = (25, 100)
        with tempfile.TemporaryDirectory(prefix="efov_contrib_") as folder:
            out = Path(folder) / "run"
            runDriver("measure", out, window)
            for stage in ["contrib-intact", "contrib-lesion"]:
                runDriver(stage, out, window)
            produced = pd.concat([pd.read_pickle(out / "results" / f"_contributions_{c}.pkl")
                                  for c in ["intact", "lesion"]], ignore_index=True)
        reference = pd.read_csv(REFERENCE / "contributions_window25_100.csv")
        merged = reference.merge(produced, on="recordingId", suffixes=("Ref", "New"),
                                 validate="one_to_one")
        self.assertEqual(len(merged), 48)
        np.testing.assert_array_equal(merged.rawFov80ReconstructionMmNew,
                                      merged.rawFov80ReconstructionMmRef)
        np.testing.assert_allclose(merged.signedCorrelationNew, merged.signedCorrelationRef,
                                   rtol=0, atol=1e-6)


if __name__ == "__main__":
    unittest.main()
