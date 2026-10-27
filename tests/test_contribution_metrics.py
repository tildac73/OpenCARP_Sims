"""Synthetic checks for Shared_Pipeline/contribution_metrics.py (Step 4)."""
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "Shared_Pipeline"))
import contribution_metrics as metrics  # noqa: E402


def pointGeometry(centres):
    """One 'element' per centre, each owning four private nodes at the centre, so that
    coefficients[e] . voltage[nodes_e] is a contribution we choose directly."""
    centres = np.asarray(centres, float)
    count = len(centres)
    return SimpleNamespace(centres=centres, nodePositions=np.repeat(centres, 4, axis=0),
                           elementNodes=np.arange(4 * count).reshape(count, 4), volumes=np.ones(count))


def unitCoefficients(count):
    coefficients = np.zeros((count, 4)); coefficients[:, 0] = 1.0
    return coefficients


def nodalVoltage(contributions):
    """Put each element's chosen contribution trace on its first node."""
    contributions = np.asarray(contributions, float)
    voltage = np.zeros((4 * len(contributions), contributions.shape[1]))
    voltage[0::4] = contributions
    return voltage


class CumulativeFraction(unittest.TestCase):
    def test_single_source_bin(self):
        trace = np.sin(np.linspace(0, 6, 50))
        fraction = metrics.cumulativeFraction(np.vstack([0 * trace, trace, 0 * trace]))
        np.testing.assert_allclose(fraction, [0.0, 1.0, 1.0], atol=1e-12)

    def test_cancelling_sources_are_not_monotone(self):
        trace = np.sin(np.linspace(0, 6, 50))
        fraction = metrics.cumulativeFraction(np.vstack([trace, 2 * trace, -2 * trace]))
        np.testing.assert_allclose(fraction, [1.0, -3.0, 1.0], atol=1e-12)
        self.assertEqual(metrics.persistentExtent(np.array([1.0, 2.0, 3.0]), fraction, 0.8), 3.0)

    def test_zero_signal_gives_nan(self):
        self.assertTrue(np.isnan(metrics.cumulativeFraction(np.zeros((3, 10)))).all())
        self.assertTrue(np.isnan(metrics.persistentExtent(np.arange(3.0), np.full(3, np.nan), 0.8)))


class Extents(unittest.TestCase):
    def test_known_radial_split(self):
        # Equal, in-phase contributions at radii 0.25, 1.25, 2.25, 3.25 mm: f(k) = 1 - ((4 - n)/4)^2.
        radii = np.array([0.25, 1.25, 2.25, 3.25])
        geometry = pointGeometry(np.c_[radii, np.zeros(4), np.zeros(4)])
        trace = np.cos(np.linspace(0, 3, 40))
        found, upper, fraction = metrics.extents(geometry, unitCoefficients(4), nodalVoltage([trace] * 4),
                                                 radii, 1.0, levels=(0.5, 0.8, 0.99))
        np.testing.assert_allclose(fraction[:4], [1 - 0.75 ** 2, 1 - 0.5 ** 2, 1 - 0.25 ** 2, 1.0])
        self.assertEqual(found, {0.5: 2.0, 0.8: 3.0, 0.99: 4.0})

    def test_binned_traces_sum_to_full_signal(self):
        rng = np.random.default_rng(1)
        geometry = pointGeometry(rng.uniform(-5, 5, (30, 3)))
        voltage = rng.normal(size=(120, 25))
        coefficients = rng.normal(size=(30, 4))
        keys = np.linalg.norm(geometry.centres, axis=1)
        binned = metrics.binnedTraces(geometry, coefficients, voltage, keys, np.arange(0, 12, 0.5))
        direct = np.einsum("ej,ejt->t", coefficients, voltage[geometry.elementNodes])
        np.testing.assert_allclose(binned.sum(axis=0), direct, atol=1e-12)


class Footprint(unittest.TestCase):
    def test_anisotropic_cloud_recovers_axes_and_angle(self):
        rng = np.random.default_rng(2)
        offsets = rng.normal(size=(20000, 2)) * [3.0, 1.0]
        angle = np.radians(30.0)
        rotation = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
        offsets = offsets @ rotation.T
        depths = rng.uniform(0, 2, len(offsets))
        geometry = pointGeometry(np.c_[offsets + [1.0, 2.0], 2.0 - depths])
        geometry.nodePositions[0, 2] = 2.0                  # surface at z = 2
        footprint, _ = metrics.energyFootprint(geometry, unitCoefficients(len(offsets)),
                                               nodalVoltage(np.ones((len(offsets), 3))), np.array([1.0, 2.0, 0.0]))
        self.assertAlmostEqual(footprint["sigmaMajorMm"], 3.0, delta=0.06)
        self.assertAlmostEqual(footprint["sigmaMinorMm"], 1.0, delta=0.03)
        self.assertAlmostEqual(footprint["majorAxisAngleToXDeg"], 30.0, delta=1.0)
        self.assertAlmostEqual(footprint["energyWeightedDepthMm"], 1.0, delta=0.02)
        self.assertAlmostEqual(footprint["centroidOffsetXMm"], 0.0, delta=0.05)

    def test_axis_angle_is_folded(self):
        geometry = pointGeometry([[-2, 0.01, 0], [2, -0.01, 0], [0, 0.5, 0], [0, -0.5, 0]])
        footprint, _ = metrics.energyFootprint(geometry, unitCoefficients(4), nodalVoltage(np.ones((4, 2))),
                                               np.zeros(3))
        self.assertTrue(-90.0 <= footprint["majorAxisAngleToXDeg"] < 90.0)
        self.assertAlmostEqual(abs(footprint["majorAxisAngleToXDeg"]), 0.0, delta=1.0)


class SphereVolume(unittest.TestCase):
    def test_counts_inside_only(self):
        geometry = pointGeometry([[0, 0, 0], [1, 0, 0], [3, 0, 0]])
        self.assertEqual(metrics.sphereVolume(geometry, np.zeros(3), 1.0), 2.0)
        self.assertTrue(np.isnan(metrics.sphereVolume(geometry, np.zeros(3), np.nan)))


if __name__ == "__main__":
    unittest.main()
