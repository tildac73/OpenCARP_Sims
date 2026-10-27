"""Lead-field quadrature tests against closed-form solutions (Step 4).

Run: python3 -m unittest tests.test_lead_field -v
"""
import importlib.util
from pathlib import Path
import sys
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "Shared_Pipeline"))
import analytic_fields  # noqa: E402
import ground_truth  # noqa: E402
import lead_field  # noqa: E402

spec = importlib.util.spec_from_file_location("quadrature_study", ROOT / "3D_Code/numerics/quadrature_study.py")
study = importlib.util.module_from_spec(spec); spec.loader.exec_module(study)


class ClosedForm(unittest.TestCase):
    def testRectangleIntegralMatchesBruteForce(self):
        n = 1200; xs = (np.arange(n) + 0.5) / n
        point = np.array([0.3, -0.2, 0.7])
        exact = analytic_fields.rectangleInverseDistance(point, 2, 0.0, ((-1.0, 2.0), (-0.5, 1.5)))
        X, Y = np.meshgrid(-1 + 3 * xs, -0.5 + 2 * xs, indexing="ij")
        brute = (1 / np.sqrt((X - point[0]) ** 2 + (Y - point[1]) ** 2 + point[2] ** 2)).mean() * 6
        self.assertLess(abs(exact - brute) / brute, 1e-6)

    def testInfiniteSlabLimit(self):
        # A wide uniformly polarised slab (J along z, thickness t) gives phi -> -J t / (2 sigma_b) above it.
        phi = analytic_fields.uniformBoxPotential([0, 0, 2.5], [-1e4, -1e4, -2], [1e4, 1e4, 2], [0, 0, 1.0])
        self.assertAlmostEqual(phi, -4 / 2, places=3)


class Subdivision(unittest.TestCase):
    def testEqualVolumeChildrenAndCentroids(self):
        vertices = np.array([[0, 0, 0], [1.3, 0.1, 0], [0.2, 0.9, 0.1], [0.1, 0.3, 1.1]])
        volume = lambda t: abs(np.linalg.det(np.c_[np.ones(4), t])) / 6
        for child in lead_field._children(np.eye(4)):
            self.assertAlmostEqual(volume(child @ vertices) / volume(vertices), 0.125, places=12)
        for level in range(4):
            np.testing.assert_allclose(lead_field.subdivisionBarycentres(level).mean(axis=0), 0.25)

    def testLevelZeroIsTheCentroidRule(self):
        rng = np.random.default_rng(2)
        centres = rng.normal(size=(50, 3)); vertices = centres[:, None, :] + rng.normal(scale=0.1, size=(50, 4, 3))
        centres = vertices.mean(axis=1)
        point = np.array([[3.0, 0.0, 0.0]])
        d = point - centres
        np.testing.assert_allclose(lead_field.averageLeadField(point, vertices, centres)[0],
                                   d / np.linalg.norm(d, axis=1)[:, None] ** 3)

    def testGroundTruthDefaultUnchanged(self):
        rng = np.random.default_rng(3)
        centres = rng.normal(size=(40, 3)); samples = rng.normal(size=(2, 8, 3)) + [0, 0, 5]
        weights = np.array([1.0, -1.0])
        expected = np.zeros_like(centres)
        for e in range(2):
            for s in samples[e]:
                d = s - centres
                expected += weights[e] * d / np.linalg.norm(d, axis=1)[:, None] ** 3 / 8
        np.testing.assert_allclose(ground_truth.finiteRecordingField(samples, weights, centres), expected)


class AccuracyAgainstAnalyticBox(unittest.TestCase):
    """Regression of the Step 4 quadrature study (0.5 mm mesh, near radius 1.5 mm)."""
    @classmethod
    def setUpClass(cls):
        cls.table = study.study(spacing=0.5, near=1.5)

    def testLevelTwoWithinHalfPercentAtAllHeights(self):
        level2 = self.table[self.table.level == 2]
        self.assertLess(level2.scaledError.max(), 0.005)

    def testCentroidRuleFineFarFromSurface(self):
        far = self.table[(self.table.level == 0) & (self.table.heightMm >= 1.0)]
        self.assertLess(far.scaledError.max(), 0.001)

    def testCentroidRuleDegradesNearSurface(self):
        near = self.table[(self.table.level == 0) & (self.table.heightMm == 0.05) & (self.table.case == "local")]
        self.assertGreater(near.scaledError.max(), 0.05)   # documents why subdivision is needed


if __name__ == "__main__":
    unittest.main()
