"""Tests for arbitrary fibre orientation (Shared_Pipeline/fibres.py and its use; Step 7).

Key checks:
  * fibres along x keep the v2-pilot arithmetic exactly (no tensors are formed);
  * rotating the whole problem (mesh, fibres, electrodes) about z leaves every electrogram unchanged,
    which tests the tensor algebra in the synthesis and ground-truth operators;
  * the FEM checker solves a rotated anisotropic problem exactly where the P1 solution is exact.
"""
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
for folder in ["Shared_Pipeline", "3D_Code", "3D_Code/numerics"]:
    sys.path.insert(0, str(ROOT / folder))
import fibres  # noqa: E402
import ground_truth  # noqa: E402
import electrode_synthesis as synthesis  # noqa: E402
from quadrature_study import boxMesh  # noqa: E402

try:
    import scipy.sparse.linalg  # noqa: F401
    HAVE_SPARSE_SOLVER = True
except ImportError:
    HAVE_SPARSE_SOLVER = False

GI = {1: [0.344, 0.060, 0.060]}
GE = {1: [0.117, 0.080, 0.080]}


def rotationZ(degrees):
    a = np.radians(degrees)
    return np.array([[np.cos(a), -np.sin(a), 0], [np.sin(a), np.cos(a), 0], [0, 0, 1.0]])


def smoothVm(positions):
    return (-80 + 100 / (1 + np.exp(-(positions[:, 0] - 0.3 * positions[:, 1]) / 0.7)))[:, None]


class FibreModels(unittest.TestCase):
    def test_angles_by_model(self):
        depth = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
        np.testing.assert_allclose(fibres.fibreAnglesDeg(depth, 4.0, {"model": "uniform", "angleDeg": 30}), 30)
        linear = {"model": "linearRotation", "endocardialAngleDeg": 0, "epicardialAngleDeg": 90}
        np.testing.assert_allclose(fibres.fibreAnglesDeg(depth, 4.0, linear), [0, 22.5, 45, 67.5, 90])
        abrupt = {"model": "bilayer", "endocardialAngleDeg": 0, "epicardialAngleDeg": 90, "transitionDepthMm": 2.5, "transitionWidthMm": 0}
        np.testing.assert_allclose(fibres.fibreAnglesDeg(depth, 4.0, abrupt), [0, 0, 0, 90, 90])
        smooth = dict(abrupt, transitionWidthMm=1.0)
        np.testing.assert_allclose(fibres.fibreAnglesDeg(np.array([2.0, 2.5, 3.0]), 4.0, smooth), [0, 45, 90])

    def test_vectors_follow_depth_from_catheter_surface(self):
        centres = np.array([[0, 0, 1.9], [0, 0, -1.9]])          # near top (catheter side), near bottom
        spec = {"model": "linearRotation", "endocardialAngleDeg": 0, "epicardialAngleDeg": 90}
        vectors = fibres.fibreVectors(centres, [10, 10, 4], spec)
        np.testing.assert_allclose(vectors[0], [np.cos(np.radians(2.25)), np.sin(np.radians(2.25)), 0], atol=1e-12)
        np.testing.assert_allclose(vectors[1], [np.cos(np.radians(87.75)), np.sin(np.radians(87.75)), 0], atol=1e-12)

    def test_invalid_specs(self):
        with self.assertRaises(ValueError):
            fibres.validateSpec({"model": "helix"}, 4)
        with self.assertRaises(ValueError):
            fibres.validateSpec({"model": "bilayer", "endocardialAngleDeg": 0, "epicardialAngleDeg": 90,
                                 "transitionDepthMm": 3.9, "transitionWidthMm": 1.0}, 4)


class Tensors(unittest.TestCase):
    def test_tensor_has_principal_values_on_fibre_axes(self):
        f = np.array([[np.cos(0.4), np.sin(0.4), 0.0]])
        tensor = fibres.tensors([[0.3, 0.05, 0.05]], fibres.localFrames(f))[0]
        np.testing.assert_allclose(tensor @ f[0], 0.3 * f[0], atol=1e-15)
        np.testing.assert_allclose(np.linalg.eigvalsh(tensor), [0.05, 0.05, 0.3], atol=1e-15)

    def test_monodomain_tensor_is_series_combination(self):
        frames = fibres.localFrames(np.array([[np.cos(1.1), np.sin(1.1), 0.0]]))
        gi, ge = np.array([[0.344, 0.06, 0.06]]), np.array([[0.117, 0.08, 0.08]])
        Si, Se = fibres.tensors(gi, frames)[0], fibres.tensors(ge, frames)[0]
        expected = Si @ np.linalg.inv(Si + Se) @ Se
        got = fibres.tensors(synthesis.monodomainConductivity(gi, ge), frames)[0]
        np.testing.assert_allclose(got, expected, atol=1e-14)


class Operators(unittest.TestCase):
    def setUp(self):
        self.positions, self.elements = boxMesh([-2, -2, -1], [2, 2, 1], 0.5)
        self.tags = np.ones(len(self.elements), int)
        self.points = np.array([[0.3, -0.2, 1.4], [1.5, 1.0, 2.2], [-1.0, 0.5, 1.25]])

    def geometry(self, positions, fibreVectors):
        return ground_truth.buildElementGeometry(positions, self.elements, self.tags, fibreVectors, GI, GE)

    def test_x_fibres_keep_pilot_arithmetic(self):
        g = self.geometry(self.positions, np.tile([1.0, 0, 0], (len(self.elements), 1)))
        self.assertIsNone(g.fibreFrames)
        self.assertEqual(synthesis.sourceConductivity(g).ndim, 2)
        forced = self.geometry(self.positions, np.tile([1.0, 0, 0], (len(self.elements), 1)))
        forced.fibreFrames = fibres.localFrames(np.tile([1.0, 0, 0], (len(self.elements), 1)))
        vm = smoothVm(self.positions)
        a = synthesis.pointOperator(g, self.points, nearRadiusMm=1.0, subdivisionLevel=1) @ vm
        b = synthesis.pointOperator(forced, self.points, nearRadiusMm=1.0, subdivisionLevel=1) @ vm
        np.testing.assert_allclose(a, b, rtol=1e-12)

    def test_rotating_everything_leaves_signals_unchanged(self):
        """Rotate mesh, fibres and electrodes by 35 deg about z: potentials must not change."""
        R = rotationZ(35.0)
        reference = self.geometry(self.positions, np.tile([1.0, 0, 0], (len(self.elements), 1)))
        rotated = self.geometry(self.positions @ R.T, np.tile(R[:, 0], (len(self.elements), 1)))
        self.assertIsNotNone(rotated.fibreFrames)
        vm = smoothVm(self.positions)                              # same nodal values: the field rotates with the mesh
        for near, level in [(0.0, 0), (1.0, 2)]:
            a = synthesis.pointOperator(reference, self.points, nearRadiusMm=near, subdivisionLevel=level) @ vm
            b = synthesis.pointOperator(rotated, self.points @ R.T, nearRadiusMm=near, subdivisionLevel=level) @ vm
            np.testing.assert_allclose(b, a, rtol=1e-10, atol=1e-14)
        # Ground-truth ring operator: same totals and ring radii after rotation (rings are lateral distance).
        samples = self.points[:, None, :]
        weights = np.array([1.0, -1.0, 0.0])
        opA, _ = ground_truth.ringOperator(reference, samples, weights, self.points[0], 0.5)
        opB, _ = ground_truth.ringOperator(rotated, samples @ R.T, weights, (R @ self.points[0]), 0.5)
        np.testing.assert_allclose((opB @ vm).ravel(), (opA @ vm).ravel(), rtol=1e-10, atol=1e-14)

    def test_fibre_angle_changes_signal(self):
        """Rotating fibres alone (not the mesh) must change the signal: anisotropy is applied."""
        base = self.geometry(self.positions, np.tile([1.0, 0, 0], (len(self.elements), 1)))
        turned = self.geometry(self.positions, np.tile([0.0, 1.0, 0], (len(self.elements), 1)))
        vm = smoothVm(self.positions)
        a = synthesis.pointOperator(base, self.points) @ vm
        b = synthesis.pointOperator(turned, self.points) @ vm
        self.assertGreater(np.abs(a - b).max(), 0.05 * np.abs(a).max())

    def test_unequal_sheet_values_need_sheet_vectors(self):
        with self.assertRaises(ValueError):
            ground_truth.buildElementGeometry(self.positions, self.elements, self.tags,
                                              np.tile([0.0, 1.0, 0], (len(self.elements), 1)),
                                              {1: [0.3, 0.1, 0.05]}, {1: [0.6, 0.2, 0.1]})


class Files(unittest.TestCase):
    def test_build_mesh_writes_and_reader_reads_fibres(self):
        import boundary_setup
        config = json.loads((ROOT / "3D_Code/configs/boundary_pilot.json").read_text())
        config.update(slabSizeMm=[4.0, 4.0, 2.0], meshSpacingMm=0.5)
        saved = os.environ.get("MESHER_BIN")
        os.environ["MESHER_BIN"] = str(ROOT / "tests/fakes/mesher")
        try:
            with tempfile.TemporaryDirectory() as folder:
                boundary_setup.buildMesh(folder, config)
                legacy = (Path(folder) / "mesh/intact.lon").read_text().splitlines()[1]
            with tempfile.TemporaryDirectory() as folder:
                config["fibres"] = {"model": "bilayer", "endocardialAngleDeg": 0, "epicardialAngleDeg": 90,
                                    "transitionDepthMm": 1.0, "transitionWidthMm": 0}
                fibres.validateSpec(config["fibres"], config["slabSizeMm"][2])
                boundary_setup.buildMesh(folder, config)
                (Path(folder) / "intact").mkdir()
                (Path(folder) / "intact/parameters.par").write_text(
                    "num_gregions = 1\ngregion[0].num_IDs = 1\ngregion[0].ID[0] = 1\n"
                    + "".join(f"gregion[0].g_i{a} = {v}\n" for a, v in zip("ltn", GI[1]))
                    + "".join(f"gregion[0].g_e{a} = {v}\n" for a, v in zip("ltn", GE[1])))
                geometry = ground_truth.readElementGeometry(folder, "intact")
        finally:
            if saved is None:
                os.environ.pop("MESHER_BIN", None)
            else:
                os.environ["MESHER_BIN"] = saved
        self.assertEqual(legacy, "1.0 0.0 0.0")                        # v2-pilot file unchanged
        fibreAxis = geometry.fibreFrames[:, 0, :]
        top = geometry.centres[:, 2] > 0
        np.testing.assert_allclose(np.abs(fibreAxis[top]), np.tile([1.0, 0, 0], (top.sum(), 1)), atol=1e-8)
        np.testing.assert_allclose(np.abs(fibreAxis[~top]), np.tile([0, 1.0, 0], ((~top).sum(), 1)), atol=1e-8)


@unittest.skipUnless(HAVE_SPARSE_SOLVER, "scipy.sparse.linalg not available")
class FemWithTensors(unittest.TestCase):
    def test_rotated_anisotropy_linear_vm_is_exact(self):
        import fem_extracellular as fem
        positions, elements = boxMesh([0, 0, 0], [3, 3, 1], 0.5)
        frames = fibres.localFrames(np.tile([np.cos(0.6), np.sin(0.6), 0.0], (len(elements), 1)))
        Si = fibres.tensors(np.tile(GI[1], (len(elements), 1)), frames)
        Se = fibres.tensors(np.tile(GE[1], (len(elements), 1)), frames)
        ground = [0]
        solver = fem.ExtracellularSolver(positions, elements, np.ones(len(elements), bool), Si, Se, 0.74, ground)
        gradient = np.array([1.5, -0.7, 0.3])
        phi = solver.solve(positions @ gradient)
        slope = -np.linalg.solve(Si[0] + Se[0], Si[0] @ gradient)       # total current zero everywhere
        exact = (positions - positions[0]) @ slope
        np.testing.assert_allclose(phi, exact, atol=1e-10)


if __name__ == "__main__":
    unittest.main()
