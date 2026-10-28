"""Tests for the Step 6 bath analysis and the Step 7 fibre check, on small synthetic run folders.

Synthetic runs hold a regular tissue grid, a smooth Vm wave and the files the modules read; the
expected answers come from independent calculations (direct solves, or synthesis with the true fibres).
"""
import json
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
for folder in ["3D_Code/readiness", "3D_Code", "Shared_Pipeline"]:
    sys.path.insert(0, str(ROOT / folder))

try:
    import scipy.sparse.linalg  # noqa: F401
    HAVE = True
except ImportError:
    HAVE = False

GI = [3.44, 0.6, 0.6]
GE = [1.17, 0.8, 0.8]


def writeIgb(path, data, stepMs):
    data = np.asarray(data, "<f4")
    header = (f"x:{data.shape[1]} y:1 z:1 t:{data.shape[0]} type:float systeme:little_endian "
              f"dim_t:{stepMs * (data.shape[0] - 1)} inc_t:{stepMs} unites:mV")
    with open(path, "wb") as stream:
        stream.write(header.encode().ljust(1023, b" ") + b"\n")
        stream.write(data.tobytes())


def syntheticRun(folder, fibreVectors=None):
    """16 x 16 x 2 mm tissue grid at 0.5 mm, straight boundary at y = 0, a planar wave along x."""
    import common
    import fem_extracellular as fem
    from boundary_geometry import makeElectrodeLayout
    folder = Path(folder)
    config = json.loads((ROOT / "3D_Code/configs/boundary_pilot.json").read_text())
    config.update(slabSizeMm=[16.0, 16.0, 2.0], meshSpacingMm=0.5, sampleStepMs=0.2)
    config["electrodes"].update(centreYmm=[0.0], gapsMm=[0.25])
    config["pacing"].update(beatCount=1, cycleLengthMs=100.0)
    config["analysis"].update(beatWindowMs=[5.0, 30.0], filterHistoryMs=4.0, groundTruthCentreYmm=[0.0])
    positions, elements = fem.structuredBox([-8, -8, -1], [8, 8, 1], 0.5)
    tags = np.ones(len(elements), int)
    (folder / "mesh").mkdir(parents=True)
    for condition in ["intact", "lesion"]:
        t = tags.copy()
        if condition == "lesion":
            t[positions[elements].mean(axis=1)[:, 1] < 0] = 5
        common.writeMesh(folder / "mesh" / condition, positions, elements, t, fibreVectors)
    layout = makeElectrodeLayout(config)
    (folder / "electrodes").mkdir()
    np.savetxt(folder / "electrodes" / "recoveryPoints.pts", layout["recoveryPoints"] * 1000, fmt="%.6f",
               header=str(len(layout["recoveryPoints"])), comments="")
    times = np.arange(0, 60.2, 0.2)
    wave = -80 + 100 / (1 + np.exp(-((times[:, None] - 10) * 0.9 - (positions[None, :, 0] + 8)) / 0.4))
    for condition in ["intact", "lesion"]:
        sub = folder / condition; sub.mkdir()
        lines = ["num_gregions = 2", "gregion[0].num_IDs = 1", "gregion[0].ID[0] = 1",
                 "gregion[1].num_IDs = 1", "gregion[1].ID[0] = 5"]
        lines += [f"gregion[0].g_i{a} = {v}" for a, v in zip("ltn", GI)] + [f"gregion[0].g_e{a} = {v}" for a, v in zip("ltn", GE)]
        lines += [f"gregion[1].g_{d}{a} = 1e-6" for d in "ie" for a in "ltn"]
        (sub / "parameters.par").write_text("\n".join(lines) + "\n")
        writeIgb(sub / "vm.igb", wave, 0.2)
        np.savez_compressed(sub / "egm.npz", uni=np.zeros((1, 16, len(times)), np.float32), uni_sample_xyz=layout["samples"],
                            t_ms=times, centre_xy_mm=layout["placements"][:, :2], p_gap_mm=layout["placements"][:, 2])
    return config, positions, elements, wave, times, layout


@unittest.skipUnless(HAVE, "scipy.sparse.linalg not available")
class BathAnalysis(unittest.TestCase):
    def test_lattice_interpolator_equals_search(self):
        import fem_extracellular as fem
        positions, elements = fem.structuredBox([-2, -2, -1], [2, 2, 3], 0.5)
        lattice = dict(positions=positions, elements=elements, lower=np.array([-2, -2, -1.0]), spacing=0.5, shape=[9, 9, 9])
        points = np.random.default_rng(0).uniform([-1.9, -1.9, -0.9], [1.9, 1.9, 2.9], (40, 3))
        a = fem.latticeInterpolator(lattice, points).toarray()
        b = fem.barycentricInterpolator(positions, elements, points).toarray()
        np.testing.assert_allclose(a, b, atol=1e-12)

    def test_model_and_electrograms(self):
        import bath_analysis as bath
        import boundary_analysis
        import fem_extracellular as fem
        with tempfile.TemporaryDirectory() as folder:
            config, positions, elements, wave, times, layout = syntheticRun(folder)
            model = bath.latticeModel(folder, config, "lesion", 4.0, "conducting")
            tissueElements = model["positions"][model["elements"]].mean(axis=1)[:, 2] < 1.0
            lesion = tissueElements & (model["positions"][model["elements"]].mean(axis=1)[:, 1] < 0)
            self.assertEqual(int(model["active"].sum()), int((tissueElements & ~lesion).sum()))
            np.testing.assert_allclose(model["positions"][model["nodeMap"]], positions)
            insulating = bath.latticeModel(folder, config, "lesion", 4.0, "insulating")
            self.assertEqual(int(insulating["active"].sum()), int(tissueElements.sum()))
            solver = bath.solverFor(model, 0.74, "lu")
            rawSlice, _ = boundary_analysis.measurementSlices(times, config)
            source, signals = bath.bathElectrograms(folder, config, "lesion", model, solver, rawSlice)
            frame = rawSlice.start + 40
            lattice = np.zeros(len(model["positions"])); lattice[model["nodeMap"]] = wave[frame]
            rows = fem.barycentricInterpolator(model["positions"], model["elements"], layout["samples"].reshape(-1, 3))
            expected = (rows @ solver.solve(lattice)).reshape(1, 16, -1).mean(axis=2)
            np.testing.assert_allclose(signals[..., frame], expected, rtol=1e-5, atol=1e-7)
            self.assertTrue(np.isnan(signals[..., 0]).all())
            # Reciprocity ground truth reproduces a bipolar electrogram: sum of contributions = weighted electrode signal.
            weights = np.zeros(16); weights[5], weights[6] = 1.0, -1.0
            row = sum(weights[e] * fem.latticeInterpolator(model["lattice"], layout["samples"][0, e]).mean(axis=0) for e in (5, 6))
            z = solver.leadField(np.asarray(row))[:, 0]
            coefficients = solver.elementCoefficients(z)
            reconstructed = np.sum(coefficients * lattice[model["elements"]])
            self.assertAlmostEqual(reconstructed, float(weights @ signals[0, :, frame]), delta=1e-4 * abs(reconstructed) + 1e-7)

    def test_derived_run_passes_pipeline_checks(self):
        """writeDerivedRun must produce a folder that run_boundary --process-only accepts."""
        import shutil
        import bath_analysis as bath
        import boundary_setup
        with tempfile.TemporaryDirectory() as folder:
            source = Path(folder) / "source"
            config, positions, elements, wave, times, layout = syntheticRun(source)
            shutil.copy2(source / "mesh" / "intact.elem", source / "mesh" / "slabRaw.elem")
            (source / "conditioning").mkdir(); (source / "conditioning" / "Courtemanche.sv").write_text("Courtemanche\n")
            (source / "sourceSnapshot").mkdir()
            shutil.copy2(ROOT / "3D_Code/3d_pilot.par", source / "sourceSnapshot" / "3d_pilot.par")
            boundary_setup.saveJson(source / "config.json", config)
            for condition in ["intact", "lesion"]:
                (source / f"{condition}.log").write_text("*** Launching simulation ***\n")
                writeIgb(source / condition / "phie_recovery.igb", np.zeros((len(times), 4)), 0.2)
                boundary_setup.saveJson(source / condition / "meta.json", dict(complete=True, condition=condition))
            signals = {c: (dict(np.load(source / c / "egm.npz")), np.ones((1, 16, len(times)), np.float32)) for c in ["intact", "lesion"]}
            derived = Path(folder) / "derived"
            bath.writeDerivedRun(source, derived, config, signals)
            boundary_setup.checkCompletedSimulations(derived)
            boundary_setup.checkSimulationSettings(derived, config)
            stored = np.load(derived / "lesion" / "egm.npz")
            np.testing.assert_array_equal(stored["uni"], 1.0)
            np.testing.assert_array_equal(stored["uni_sample_xyz"], layout["samples"])
            self.assertTrue((derived / "intact" / "vm.igb").is_symlink())


@unittest.skipUnless(HAVE, "scipy.sparse.linalg not available")
class FibreCheck(unittest.TestCase):
    def test_compare_detects_correct_and_wrong_fibres(self):
        import fibre_check
        import electrode_synthesis as synthesis
        import ground_truth
        angle = np.radians(30)
        with tempfile.TemporaryDirectory() as folder:
            from fem_extracellular import structuredBox
            positions, elements = structuredBox([-8, -8, -1], [8, 8, 1], 0.5)
            fibres = np.tile([np.cos(angle), np.sin(angle), 0.0], (len(elements), 1))
            config, positions, elements, wave, times, layout = syntheticRun(folder, fibres)
            config["pacing"]["beatCount"] = 1
            geometry = ground_truth.readElementGeometry(folder, "intact")
            points = np.loadtxt(Path(folder) / "electrodes/recoveryPoints.pts", skiprows=1, ndmin=2) / 1000
            truth = (synthesis.pointOperator(geometry, points, nearRadiusMm=1.5, subdivisionLevel=2) @ wave.T.astype(float)).T
            writeIgb(Path(folder) / "intact" / "phie_recovery.igb", truth, 0.2)
            correct = fibre_check.compare(folder, "intact", config)
            wrong = fibre_check.compare(folder, "intact", config, assumeXFibres=True)
        self.assertGreater(correct["correlationMin"], 0.99999)
        self.assertLess(correct["relativeRmsMax"], 1e-5)
        self.assertGreater(wrong["relativeRmsMedian"], 100 * max(correct["relativeRmsMedian"], 1e-9))


if __name__ == "__main__":
    unittest.main()
