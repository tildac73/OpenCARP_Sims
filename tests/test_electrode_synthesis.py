"""Unit tests for Shared_Pipeline/electrode_synthesis.py.

The analytic check uses one tetrahedron with a uniform Vm gradient: in the far field its
potential must equal the closed-form dipole -(1/4 pi sigma_b) (G grad V vol) . r / |r|^3.
Run: python3 -m unittest tests.test_electrode_synthesis -v
"""
from pathlib import Path
import sys
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "Shared_Pipeline"))
import electrode_synthesis as synthesis  # noqa: E402


class OneTetrahedron:
    """Minimal geometry object with the ElementGeometry fields the module reads."""
    def __init__(self, gi=(0.3, 0.1, 0.1), ge=(0.6, 0.2, 0.2)):
        self.nodePositions = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]], float)
        self.elementNodes = np.array([[0, 1, 2, 3]])
        self.centres = self.nodePositions.mean(axis=0, keepdims=True)
        matrix = np.concatenate([np.ones((1, 4, 1)), self.nodePositions[None]], axis=2)
        self.gradients = np.linalg.inv(matrix)[:, 1:, :]
        self.volumes = np.abs(np.linalg.det(matrix)) / 6
        self.conductivity = np.array([gi], float)
        self.extracellularConductivity = np.array([ge], float)


class ElectrodeSynthesisTests(unittest.TestCase):
    def testMonodomainConductivityIsSeriesCombination(self):
        np.testing.assert_allclose(synthesis.monodomainConductivity([0.3], [0.6]), [0.2])
        with self.assertRaises(ValueError):
            synthesis.monodomainConductivity([0.3], [0.0])

    def testUniformGradientMatchesClosedFormDipole(self):
        geometry = OneTetrahedron()
        gradient = np.array([2.0, -1.0, 0.5])                       # mV/mm
        voltage = (geometry.nodePositions @ gradient)[:, None]       # (nodes, 1 frame)
        point = geometry.centres[0] + np.array([5.0, 3.0, 4.0])
        sigmaBath = 1.7
        result = synthesis.pointOperator(geometry, point, sigmaBathSPerM=sigmaBath) @ voltage
        conductivity = synthesis.monodomainConductivity(geometry.conductivity, geometry.extracellularConductivity)[0]
        r = point - geometry.centres[0]
        expected = -(conductivity * gradient * geometry.volumes[0]) @ r / np.linalg.norm(r) ** 3 / (4 * np.pi * sigmaBath)
        np.testing.assert_allclose(result[0, 0], expected, rtol=1e-12)

    def testIntracellularSourceScalesByConductivityRatio(self):
        geometry = OneTetrahedron(gi=(0.3, 0.3, 0.3), ge=(0.6, 0.6, 0.6))
        point = np.array([[4.0, 4.0, 4.0]])
        mono = synthesis.pointOperator(geometry, point)
        intra = synthesis.pointOperator(geometry, point, source="intracellular")
        np.testing.assert_allclose(intra, mono * (0.3 / 0.2), rtol=1e-12)

    def testElectrodeAverageEqualsMeanOfSamples(self):
        geometry = OneTetrahedron()
        voltage = np.random.default_rng(1).normal(size=(4, 7))
        samples = np.array([[[3.0, 0, 0], [0, 3.0, 0]], [[0, 0, 4.0], [4.0, 4.0, 0]]])
        averaged = synthesis.electrodePotentials(geometry, samples, voltage)
        direct = synthesis.pointOperator(geometry, samples.reshape(-1, 3)) @ voltage
        np.testing.assert_allclose(averaged, direct.reshape(2, 2, 7).mean(axis=1))

    def testRejectsPointAtCentroidAndBadSigma(self):
        geometry = OneTetrahedron()
        with self.assertRaises(ValueError):
            synthesis.pointOperator(geometry, geometry.centres)
        with self.assertRaises(ValueError):
            synthesis.pointOperator(geometry, [[5.0, 5.0, 5.0]], sigmaBathSPerM=0)


if __name__ == "__main__":
    unittest.main()
