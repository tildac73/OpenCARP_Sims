"""Tests for the readiness kit (3D_Code/readiness) and the pipeline options it relies on.

Simulator-dependent parts run against test doubles in tests/fakes (mesher, bench, openCARP).
The fake openCARP is NOT a physics model: it writes a planar activation map whose speed depends
on conductivity, so the tuning logic, file handling and gates can be exercised without openCARP.
"""
import copy
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile
import unittest

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
FAKES = ROOT / "tests" / "fakes"
for folder in ["3D_Code/readiness", "3D_Code", "Shared_Pipeline"]:
    sys.path.insert(0, str(ROOT / folder))

import boundary_setup  # noqa: E402
import common  # noqa: E402

try:
    import scipy.sparse.linalg  # noqa: F401
    HAVE_SPARSE_SOLVER = True
except ImportError:          # the analysis VM uses numpy stand-ins without a sparse solver
    HAVE_SPARSE_SOLVER = False

RS = {"g_il": 0.344, "g_it": 0.060, "g_in": 0.060, "g_el": 0.117, "g_et": 0.080, "g_en": 0.080}


def writeIgb(path, data, stepMs):
    data = np.asarray(data, "<f4")
    header = (f"x:{data.shape[1]} y:1 z:1 t:{data.shape[0]} type:float systeme:little_endian "
              f"dim_t:{stepMs * (data.shape[0] - 1)} inc_t:{stepMs} unites:mV")
    with open(path, "wb") as stream:
        stream.write(header.encode().ljust(1023, b" ") + b"\n")
        stream.write(data.tobytes())


class FakeExecutables:
    """Point the kit at the test doubles via the documented environment variables."""

    def __enter__(self):
        self.saved = {k: os.environ.get(k) for k in ["OPENCARP_BIN", "MESHER_BIN", "BENCH_BIN", "MPIEXEC_BIN"]}
        os.environ.update(OPENCARP_BIN=str(FAKES / "openCARP"), MESHER_BIN=str(FAKES / "mesher"), BENCH_BIN=str(FAKES / "bench"))
        os.environ.pop("MPIEXEC_BIN", None)
        return self

    def __exit__(self, *exc):
        for key, value in self.saved.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


class PipelineOptions(unittest.TestCase):
    def test_conductivity_override_replaces_each_line_once(self):
        text = (ROOT / "3D_Code/3d_pilot.par").read_text()
        values = {k: v * 10 for k, v in RS.items()}
        changed = boundary_setup.applyTissueConductivities(text, values)
        self.assertIn("gregion[0].g_il = 3.44\n", changed)
        self.assertIn("gregion[0].g_et = 0.8\n", changed)
        self.assertIn("gregion[1].g_il = 1e-6", changed)                 # lesion untouched
        self.assertEqual(changed.count("gregion[0].g_il"), 1)
        with self.assertRaises(ValueError):
            boundary_setup.applyTissueConductivities("num_gregions = 1\n", values)

    def test_validate_new_keys(self):
        config = json.loads((ROOT / "3D_Code/configs/boundary_pilot.json").read_text())
        config["tissue"] = {"conductivitiesSPerM": dict(RS)}
        config["acceptance"]["responseWindowMs"] = [0.0, 400.0]
        config["analysis"]["beatWindowMs"] = [7.0, 60.0]
        boundary_setup.validateConfig(config)
        bad = copy.deepcopy(config); bad["tissue"]["conductivitiesSPerM"].pop("g_en")
        with self.assertRaises(ValueError):
            boundary_setup.validateConfig(bad)
        bad = copy.deepcopy(config); bad["acceptance"]["responseWindowMs"] = [0.0, 600.0]
        with self.assertRaises(ValueError):
            boundary_setup.validateConfig(bad)

    def test_pipeline_logs_lose_credentials(self):
        with tempfile.TemporaryDirectory() as folder:
            log = Path(folder) / "x.log"
            log.write_text("*** GIT repo: https://gitlab-ci-token:abc_123@git.opencarp.org/openCARP/openCARP.git\n")
            boundary_setup.redactCredentials(log)
            self.assertNotIn("abc_123", log.read_text())
            self.assertEqual(common.redact("https://user:pw@host/x"), "https://<credentials removed>@host/x")


class CommonHelpers(unittest.TestCase):
    def test_activation_formats(self):
        with tempfile.TemporaryDirectory() as folder:
            events = Path(folder) / "a.dat"
            events.write_text("0\t1.5\n2\t3.0\n0\t501.5\n")
            nodes, times = common.readActivationFile(events, 3)
            first = common.firstActivationInWindow(nodes, times, 3, 500, 1000)
            np.testing.assert_allclose(first, [501.5, np.nan, np.nan])
            perNode = Path(folder) / "b.dat"
            perNode.write_text("1.0\n-1\n2.0\n")
            nodes, times = common.readActivationFile(perNode, 3)
            np.testing.assert_array_equal(nodes, [0, 2])
            empty = Path(folder) / "c.dat"; empty.write_text("")
            self.assertEqual(len(common.readActivationFile(empty, 3)[0]), 0)

    def test_plane_fit_recovers_speed(self):
        rng = np.random.default_rng(0)
        positions = rng.uniform(-5, 5, (200, 3))
        times = 3.0 + positions[:, 1] / 0.9
        fit = common.fitPlanarSpeed(positions, times, axis=1)
        self.assertAlmostEqual(fit["speedMPerS"], 0.9, places=10)
        self.assertAlmostEqual(fit["directionErrorDeg"], 0.0, places=6)

    def test_smoke_copy_and_log_parsing(self):
        with tempfile.TemporaryDirectory() as folder:
            par = Path(folder) / "run.par"
            par.write_text('bidomain = 0\nsimID = "/x/sim"\ntend = 2000\n')
            smoke = common.smokeCopy(par, Path(folder) / "smoke")
            self.assertIn("tend = 1.0", smoke.read_text())
            self.assertIn(str(Path(folder) / "smoke"), smoke.read_text())
            log = Path(folder) / "o.log"
            log.write_text("*** GIT hash:           9accae\n1.00\t0.0\t7.8\t0:00:07\t17:20:47\n8000\t100\t1\t17:21:02\t0:00:00\n")
            self.assertEqual(common.lastProgressSeconds(log), 17 * 3600 + 21 * 60 + 2)
            self.assertEqual(common.openCarpBanner(log)["GIT hash"], "9accae")


class CvTuning(unittest.TestCase):
    def test_tuning_loop_with_capture_search(self):
        import strip_cv
        settings = copy.deepcopy(common.loadSettings(common.REPOSITORY / "3D_Code/configs/readiness_kit.json")["cvTuning"])
        settings.update(referenceSpacingMm=0.25, evaluationSpacingsMm=[0.5], pilotConductivitySpacingsMm=[0.5],
                        evaluationTimeStepsUs=[10.0], sensitivityTargetsMPerS=[1.25])
        with FakeExecutables(), tempfile.TemporaryDirectory() as folder:
            summary = strip_cv.runStage(Path(folder) / "cv", settings, 1)
            tuned = json.loads((Path(folder) / "cv" / "tuned.json").read_text())
            logText = next((Path(folder) / "cv").glob("tune_*/iter01*/openCARP.log")).read_text()
        self.assertTrue(summary["converged"])
        self.assertLessEqual(abs(summary["history"][-1]["relativeError"]), settings["tuningToleranceRelative"])
        self.assertEqual(summary["history"][0]["action"], "strength doubled to 100")   # fake: no capture at 50
        self.assertEqual(tuned["stimulusStrength"], 100.0)
        ratios = {k: tuned["conductivitiesSPerM"][k] / RS[k] for k in RS}
        self.assertLess(np.ptp(list(ratios.values())), 1e-9)                            # all ratios kept
        self.assertTrue(summary["sensitivity"]["1.25"]["converged"])
        self.assertEqual(summary["sensitivity"]["1.25"]["stimulusStrength"], 200.0)
        self.assertNotIn("SECRET123", logText)


class Benchmark(unittest.TestCase):
    def test_parameter_blocks_and_geometry(self):
        import niederer
        settings = common.loadSettings(common.REPOSITORY / "3D_Code/configs/readiness_kit.json")["niederer"]
        text = niederer.parameterText(settings, 1)
        for line in ['imp_region[0].im_param          = "flags=EPI"', "gregion[0].g_it = 0.019", "stim[0].pulse.strength  = 35.71",
                     "stim[0].elec.p0[0]      = -10000.0", "stim[0].elec.p1[2]      = 0.0", "lats[0].threshold    = 0.", "stim[0].ptcl.bcl        = 150.0"]:
            self.assertIn(line, text)
        fraction, points = niederer.diagonalPoints(settings["slabSizeMm"])
        np.testing.assert_allclose(points[0], [-10, -3.5, -1.5]); np.testing.assert_allclose(points[-1], [10, 3.5, 1.5])


class Ladder(unittest.TestCase):
    def setUp(self):
        import ladder
        self.ladder = ladder
        self.settings = common.loadSettings(common.REPOSITORY / "3D_Code/configs/readiness_kit.json")
        self.tuned = dict(conductivitiesSPerM={k: 10 * v for k, v in RS.items()}, stimulusStrength=100.0)

    def test_every_run_config_is_valid(self):
        base = json.loads((ROOT / "3D_Code/configs/boundary_pilot.json").read_text())
        for run in self.settings["ladder"]["runs"]:
            config = self.ladder.makeConfig(base, run, self.settings["ladder"], self.tuned)
            self.assertAlmostEqual(config["tissue"]["conductivitiesSPerM"]["g_il"], 3.44)
            self.assertEqual(config["analysis"]["leadFieldQuadrature"]["nearRadiusMm"], 3 * run["spacingMm"])

    def test_window_rule_reproduces_pilot_choice(self):
        """Pilot intact beat-16 grid activation 39.3-73.4 ms -> 24.2-103.4 ms (DL-2026-10-27-01 chose 25-100)."""
        with tempfile.TemporaryDirectory() as folder:
            folder = Path(folder)
            (folder / "mesh").mkdir(); (folder / "intact").mkdir()
            xs = np.linspace(-10, 10, 41)
            positions = np.array([[x, 0.0, z] for x in xs for z in (-2.0, 2.0)])
            common.writeMesh(folder / "mesh" / "intact", positions, np.array([[0, 1, 2, 3]]), [1])
            time = 39.3 + (positions[:, 0] + 6.5) * (73.4 - 39.3) / 13.0
            beatStart = 15 * 500.0
            rows = [f"{i}\t{t + beatStart}" for i, t in enumerate(time)] + [f"{i}\t{t}" for i, t in enumerate(time)]
            (folder / "intact" / "act-thresh.dat").write_text("\n".join(rows) + "\n")
            config = dict(pacing=dict(cycleLengthMs=500.0, beatCount=16, durationMs=2.0), sampleStepMs=0.2)
            window = self.ladder.deriveWindow(folder, config, self.settings["ladder"])
        self.assertEqual(window["windowMs"], [24.2, 103.4])

    def test_decimation_keeps_every_second_frame(self):
        import igb_io
        with tempfile.TemporaryDirectory() as folder:
            data = np.arange(21 * 3, dtype=float).reshape(21, 3)
            writeIgb(Path(folder) / "a.igb", data, 0.1)
            self.ladder.decimateIgb(Path(folder) / "a.igb", Path(folder) / "b.igb", 2)
            np.testing.assert_allclose(igb_io.read(Path(folder) / "b.igb"), data[::2])
            np.testing.assert_allclose(igb_io.timeAxis(Path(folder) / "b.igb"), np.arange(11) * 0.2)

    def _analysedRun(self, folder, ppScale=1.0, cpShift=0.0, speed=0.9, gain=0.5):
        results = folder / "results"; (results / "clinical").mkdir(parents=True)
        rows = []
        for condition in ["intact", "lesion"]:
            for index, centre in enumerate([-1.0, 0.0, 1.0]):
                rows.append(dict(recordingId=f"{condition}:0:{index}:bi:B", condition=condition, noiseRepeat=0, centreYmm=centre,
                                 gapMm=0.25, kind="bi", label="B", distanceInsideMm=centre,
                                 cleanAtSelectedWeightsPpMv=gain * ppScale * (2.0 + centre)))
        points = pd.DataFrame(rows)
        for condition in ["intact", "lesion"]:
            points[points.condition == condition].to_csv(results / f"{condition}Points.csv", index=False)
        pd.DataFrame(dict(recordingId=["intact:0:1:bi:B", "lesion:0:1:bi:B"], fov80ReconstructionMm=[3.5, 3.0])).to_csv(results / "contributions.csv", index=False)
        pd.DataFrame([dict(condition="lesion", gapMm=0.25, noiseRepeat=0, kind="bi", label="B", distanceConvention="recordingAnchor",
                           preprocessing="clinicalTrimmed", model="Exponential", changePointMm=5.0 + cpShift,
                           fitStatus="supportedDescriptiveChangePoint", selected=True)]).to_csv(results / "clinical" / "changePointFits.csv", index=False)
        (results / "calibration.json").write_text(json.dumps(dict(gain=gain)))
        pd.DataFrame([dict(condition="intact", beatNumber=4, speedMmPerMs=speed)]).to_csv(folder / "physiologyChecks.csv", index=False)

    def test_comparison_passes_and_fails_as_specified(self):
        with tempfile.TemporaryDirectory() as folder:
            a, b, c = Path(folder) / "a", Path(folder) / "b", Path(folder) / "c"
            self._analysedRun(a); self._analysedRun(b, gain=0.7)       # different gain, same raw signal
            self._analysedRun(c, ppScale=1.08, cpShift=0.6, speed=0.86)
            same = self.ladder.compare(a, b, "same")
            different = self.ladder.compare(a, c, "different")
        self.assertTrue(same["passedStrict"]); self.assertAlmostEqual(same["ppRelativeMaxAll"], 0.0)
        self.assertFalse(different["passedClarified"])
        self.assertAlmostEqual(different["ppRelativeMaxAll"], 0.08 / 1.08)
        self.assertAlmostEqual(different["changePointMaxAbsMmSupported"], 0.6)
        self.assertAlmostEqual(different["cvRelative"], 0.04 / 0.86)


@unittest.skipUnless(HAVE_SPARSE_SOLVER, "scipy.sparse.linalg not available")
class BathCheck(unittest.TestCase):
    def setUp(self):
        import bath_check
        self.bath = bath_check
        kit = common.loadSettings(common.REPOSITORY / "3D_Code/configs/readiness_kit.json")
        self.kit = kit
        self.settings = dict(kit["bath"], tissueSizeMm=[16.0, 16.0, 1.0], bathThicknessMm=4.0)
        self.tuned = dict(conductivitiesSPerM={k: 10 * v for k, v in RS.items()}, stimulusStrength=100.0)

    def _synthetic(self, folder, error=0.0):
        import fem_extracellular as fem
        with FakeExecutables():
            positions, elements, isTissue, tissueCount = self.bath.buildMesh(folder, self.settings)
        ground = np.flatnonzero(np.isclose(positions[:, 2], positions[:, 2].max()))
        np.savez_compressed(folder / "bathMesh.npz", positions=positions, elements=elements, isTissueElement=isTissue,
                            tissueNodeCount=tissueCount, groundNodes=ground)
        times = np.arange(0, 30.2, 0.2)
        x = positions[:tissueCount, 0]
        vm = -80 + 100 / (1 + np.exp(-((times[:, None] - 5.0) * 0.9 - (x[None] + 8.0)) / 0.3))
        c = self.tuned["conductivitiesSPerM"]
        solver = fem.ExtracellularSolver(positions, elements, isTissue,
                                         np.tile([c["g_il"], c["g_it"], c["g_in"]], (len(elements), 1)),
                                         np.tile([c["g_el"], c["g_et"], c["g_en"]], (len(elements), 1)),
                                         self.settings["bloodConductivitySPerM"], ground)
        vmAll = np.zeros((len(positions), len(times))); vmAll[:tissueCount] = vm.T
        phi = solver.solve(vmAll).T * (1 + error)
        (folder / "sim").mkdir()
        writeIgb(folder / "sim" / "vm.igb", vm, 0.2)
        writeIgb(folder / "sim" / "phie.igb", phi, 0.2)
        return positions, elements, tissueCount

    def test_mesh_numbers_tissue_first(self):
        with tempfile.TemporaryDirectory() as folder:
            positions, elements, tissueCount = self._synthetic(Path(folder))
        self.assertTrue(np.all(positions[:tissueCount, 2] <= 0.5 + 1e-9))
        self.assertAlmostEqual(positions[:, 2].min(), -0.5); self.assertAlmostEqual(positions[:, 2].max(), 4.5)

    def test_check_accepts_identical_and_rejects_scaled(self):
        with tempfile.TemporaryDirectory() as folder:
            self._synthetic(Path(folder))
            good = self.bath.check(Path(folder), self.settings, self.tuned)
        with tempfile.TemporaryDirectory() as folder:
            self._synthetic(Path(folder), error=0.05)
            bad = self.bath.check(Path(folder), self.settings, self.tuned)
        self.assertTrue(good["vmOrderingConfirmed"]); self.assertTrue(good["passed"])
        self.assertLess(good["sampleRelativeRmsMax"], 1e-5)
        self.assertFalse(bad["passed"]); self.assertAlmostEqual(bad["sampleRelativeRmsMax"], 0.05, places=4)

    def test_fem_exact_for_linear_vm(self):
        import fem_extracellular as fem
        with tempfile.TemporaryDirectory() as folder:
            with FakeExecutables():
                positions, elements = common.generateBox(folder, "box", [4.0, 2.0, 2.0], 0.5)
        sigmaI, sigmaE = np.array([0.3, 0.05, 0.05]), np.array([0.1, 0.08, 0.08])
        ground = np.flatnonzero(np.isclose(positions[:, 0], 2.0))
        solver = fem.ExtracellularSolver(positions, elements, np.ones(len(elements), bool), np.tile(sigmaI, (len(elements), 1)),
                                         np.tile(sigmaE, (len(elements), 1)), 0.74, ground)
        phi = solver.solve(2.0 * positions[:, 0])
        exact = -sigmaI[0] / (sigmaI[0] + sigmaE[0]) * 2.0 * (positions[:, 0] - 2.0)
        np.testing.assert_allclose(phi, exact, atol=1e-10)


class Plan(unittest.TestCase):
    def test_plan_sizes(self):
        import kit
        settings = common.loadSettings(common.REPOSITORY / "3D_Code/configs/readiness_kit.json")
        jobs = {job["name"]: job for job in kit.planJobs(settings, ("A", "B"))}
        self.assertEqual(jobs["D1_dx0.50_40x60"]["nodes"], 88209)          # the pilot mesh
        self.assertEqual(jobs["L2_dx0.25_dt20"]["nodes"], 161 * 121 * 17)
        self.assertEqual(jobs["dx0.1_dt5"]["nodes"], 201 * 71 * 31)


if __name__ == "__main__":
    unittest.main()
