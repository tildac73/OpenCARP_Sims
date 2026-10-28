"""Tests for bath lead fields (Shared_Pipeline/fem_extracellular.py, Step 6).

Analytic references (method of images, point current I in medium 1 at distance d from a plane):
  * insulating plane: phi on the plane is exactly twice the free-space value;
  * plane between conductivities s1 (source side) and s2: phi on the plane = I / (2 pi (s1 + s2) r),
    i.e. 2 s1 / (s1 + s2) times the free-space value in s1.
The finite boxes are grounded on far faces, so comparisons use potential differences between two
observation points (the grounding offset cancels); discretisation largely cancels in the ratio.
"""
from pathlib import Path
import sys
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
for folder in ["Shared_Pipeline", "3D_Code/numerics"]:
    sys.path.insert(0, str(ROOT / folder))

try:
    import scipy.sparse.linalg  # noqa: F401
    import fem_extracellular as fem
    HAVE = True
except ImportError:
    HAVE = False


def node(positions, point):
    index = int(np.argmin(np.linalg.norm(positions - point, axis=1)))
    assert np.allclose(positions[index], point), "test point must be a lattice node"
    return index


@unittest.skipUnless(HAVE, "scipy.sparse.linalg not available")
class LeadFields(unittest.TestCase):
    def test_structured_box_matches_reference_mesh(self):
        from quadrature_study import boxMesh
        a, ea = fem.structuredBox([0, 0, 0], [1, 1.5, 1], 0.5)
        b, eb = boxMesh([0, 0, 0], [1, 1.5, 1], 0.5)
        np.testing.assert_allclose(a, b)
        self.assertEqual(sorted(map(tuple, np.sort(ea, 1))), sorted(map(tuple, np.sort(eb, 1))))

    def _tissueBath(self):
        positions, elements = fem.structuredBox([-3, -3, -1], [3, 3, 3], 0.5)
        centres = positions[elements].mean(axis=1)
        tissue = centres[:, 2] < 1.0
        si = np.tile([0.344, 0.06, 0.06], (len(elements), 1)) * 10
        se = np.tile([0.117, 0.08, 0.08], (len(elements), 1)) * 10
        ground = np.flatnonzero(np.isclose(positions[:, 2], 3.0))
        return positions, elements, tissue, si, se, ground

    def test_reciprocity_equals_forward_solution(self):
        positions, elements, tissue, si, se, ground = self._tissueBath()
        solver = fem.ExtracellularSolver(positions, elements, tissue, si, se, 0.74, ground, method="lu")
        rng = np.random.default_rng(3)
        vm = np.where(positions[:, 2] <= 1.0, -80 + 100 / (1 + np.exp(-(positions[:, 0] - 0.4) / 0.6)), 0.0)
        vm += 0.5 * rng.normal(size=len(vm))
        points = np.array([[0.3, -0.2, 1.4], [1.1, 0.7, 2.1]])
        rows = fem.barycentricInterpolator(positions, elements, points)
        forward = rows @ solver.solve(vm)
        z = solver.leadField(rows)
        for k in range(len(points)):
            coefficients = solver.elementCoefficients(z[:, k])
            reciprocal = np.sum(coefficients * vm[elements])
            self.assertAlmostEqual(reciprocal, forward[k], delta=1e-10 * max(1.0, abs(forward[k])))

    def test_iterative_matches_direct(self):
        positions, elements, tissue, si, se, ground = self._tissueBath()
        vm = np.where(positions[:, 2] <= 1.0, positions[:, 0] ** 2, 0.0)
        direct = fem.ExtracellularSolver(positions, elements, tissue, si, se, 0.74, ground, method="lu").solve(vm)
        iterative = fem.ExtracellularSolver(positions, elements, tissue, si, se, 0.74, ground, method="ilu").solve(vm)
        np.testing.assert_allclose(iterative, direct, atol=1e-7 * np.abs(direct).max())

    def _surfaceDifference(self, positions, elements, sigma, source, a, b, ground, tissueMask=None):
        tissue = np.zeros(len(elements), bool) if tissueMask is None else tissueMask
        zero = np.zeros((len(elements), 3))
        solver = fem.ExtracellularSolver(positions, elements, tissue, zero, zero, 1.0, ground, method="lu",
                                         passiveConductivity=sigma)
        load = np.zeros(len(positions)); load[node(positions, source)] = 1.0
        phi = solver.solveLoad(load)
        return phi[node(positions, a)] - phi[node(positions, b)]

    def test_insulating_plane_doubles_surface_potential(self):
        h, L = 0.5, 8.0
        full, fe = fem.structuredBox([-L, -L, -L], [L, L, L], h)
        half, he = fem.structuredBox([-L, -L, -L], [L, L, 0.0], h)
        sigma = lambda e: np.ones((len(e), 3))
        far = lambda p: np.flatnonzero((np.abs(p[:, 0]) == L) | (np.abs(p[:, 1]) == L) | (p[:, 2] == -L))
        a, b = np.array([2.0, 0, 0]), np.array([4.0, 0, 0])                  # on the plane z = 0
        source = np.array([0, 0, -2.0])
        dHalf = self._surfaceDifference(half, he, sigma(he), source, a, b, far(half))
        dFull = self._surfaceDifference(full, fe, sigma(fe), source, a, b, far(full) if False else
                                        np.flatnonzero(np.abs(full).max(axis=1) == L))
        self.assertAlmostEqual(dHalf / dFull, 2.0, delta=0.08)

    def test_conductivity_jump_matches_image_factor(self):
        h, L = 0.5, 8.0
        positions, elements = fem.structuredBox([-L, -L, -L], [L, L, L], h)
        centres = positions[elements].mean(axis=1)
        s1, s2 = 1.0, 3.0
        jump = np.where(centres[:, 2:3] < 0, s1, s2) * np.ones((1, 3))
        uniform = np.full((len(elements), 3), s1)
        ground = np.flatnonzero(np.abs(positions).max(axis=1) == L)
        a, b, source = np.array([2.0, 0, 0]), np.array([4.0, 0, 0]), np.array([0, 0, -2.0])
        ratio = (self._surfaceDifference(positions, elements, jump, source, a, b, ground)
                 / self._surfaceDifference(positions, elements, uniform, source, a, b, ground))
        self.assertAlmostEqual(ratio, 2 * s1 / (s1 + s2), delta=0.04)


if __name__ == "__main__":
    unittest.main()
