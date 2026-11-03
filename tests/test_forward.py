"""Extracellular solver: reciprocity identity and method-of-images references.

Images (point current in medium 1 at distance d from a plane): an insulating plane doubles the
potential on the plane; a jump s1 | s2 multiplies it by 2 s1 / (s1 + s2). Finite grounded boxes,
so potential differences between two points on the plane are compared.
"""
import numpy as np

from efov import forward
from helpers import boxMesh


def node(positions, point):
    index = int(np.argmin(np.linalg.norm(positions - point, axis=1)))
    assert np.allclose(positions[index], point)
    return index


def tensorsOf(values):
    return np.einsum("ei,ij->eij", np.asarray(values, float), np.eye(3))


def tissueBath():
    positions, elements = boxMesh([-3, -3, -1], [3, 3, 3], 0.5)
    tissue = positions[elements].mean(axis=1)[:, 2] < 1.0
    gi, ge = np.array([0.344, 0.06, 0.06]), np.array([0.117, 0.08, 0.08])
    total = np.where(tissue[:, None, None], tensorsOf(np.tile(gi + ge, (len(elements), 1))), 0.74 * np.eye(3))
    source = tensorsOf(np.tile(gi * ge / (gi + ge), (len(elements), 1)))
    ground = np.flatnonzero(np.isclose(positions[:, 2], 3.0))
    return positions, elements, tissue, total, source, ground


def test_reciprocity_equals_forward_solution():
    positions, elements, tissue, total, source, ground = tissueBath()
    solver = forward.ExtracellularSolver(positions, elements, total, source, tissue, ground, method="lu")
    vm = np.where(positions[:, 2] <= 1.0, -80 + 100 / (1 + np.exp(-(positions[:, 0] - 0.4) / 0.6)), 0.0)
    vm += 0.5 * np.random.default_rng(3).normal(size=len(vm))
    rows = forward.pointInterpolator(positions, elements, np.array([[0.3, -0.2, 1.4], [1.1, 0.7, 2.1]]))
    direct = rows @ solver.solve(vm)
    z = solver.leadField(rows)
    for k in range(2):
        reciprocal = np.sum(solver.elementCoefficients(z[:, k]) * vm[elements])
        np.testing.assert_allclose(reciprocal, direct[k], rtol=1e-10, atol=1e-12)


def test_iterative_matches_direct():
    positions, elements, tissue, total, source, ground = tissueBath()
    vm = np.where(positions[:, 2] <= 1.0, positions[:, 0] ** 2, 0.0)
    direct = forward.ExtracellularSolver(positions, elements, total, source, tissue, ground, method="lu").solve(vm)
    for method in ["ilu"] + (["amg"] if forward.haveAmg() else []):
        other = forward.ExtracellularSolver(positions, elements, total, source, tissue, ground, method=method).solve(vm)
        np.testing.assert_allclose(other, direct, atol=1e-7 * np.abs(direct).max())


def surfaceDifference(positions, elements, sigma, sourcePoint, a, b, ground):
    solver = forward.ExtracellularSolver(positions, elements, tensorsOf(sigma), np.zeros((len(elements), 3, 3)),
                                         np.zeros(len(elements), bool), ground, method="lu")
    load = np.zeros(len(positions)); load[node(positions, sourcePoint)] = 1.0
    phi = solver.solveLoad(load)
    return phi[node(positions, a)] - phi[node(positions, b)]


def test_insulating_plane_doubles_surface_potential():
    h, L = 0.5, 8.0
    full, fe = boxMesh([-L] * 3, [L] * 3, h)
    half, he = boxMesh([-L] * 3, [L, L, 0.0], h)
    a, b, src = np.array([2.0, 0, 0]), np.array([4.0, 0, 0]), np.array([0, 0, -2.0])
    groundHalf = np.flatnonzero((np.abs(half[:, 0]) == L) | (np.abs(half[:, 1]) == L) | (half[:, 2] == -L))
    groundFull = np.flatnonzero(np.abs(full).max(axis=1) == L)
    ratio = (surfaceDifference(half, he, np.ones((len(he), 3)), src, a, b, groundHalf)
             / surfaceDifference(full, fe, np.ones((len(fe), 3)), src, a, b, groundFull))
    assert abs(ratio - 2.0) < 0.08


def test_conductivity_jump_matches_image_factor():
    h, L, s1, s2 = 0.5, 8.0, 1.0, 3.0
    positions, elements = boxMesh([-L] * 3, [L] * 3, h)
    jump = np.where(positions[elements].mean(axis=1)[:, 2:3] < 0, s1, s2) * np.ones((1, 3))
    ground = np.flatnonzero(np.abs(positions).max(axis=1) == L)
    a, b, src = np.array([2.0, 0, 0]), np.array([4.0, 0, 0]), np.array([0, 0, -2.0])
    ratio = (surfaceDifference(positions, elements, jump, src, a, b, ground)
             / surfaceDifference(positions, elements, np.full((len(elements), 3), s1), src, a, b, ground))
    assert abs(ratio - 2 * s1 / (s1 + s2)) < 0.04


def test_bath_model_uses_series_source_and_insulating_lesion():
    positions, elements = boxMesh([-2, -2, -0.5], [2, 2, 1.5], 0.5)
    centres = positions[elements].mean(axis=1)
    tags = np.where(centres[:, 2] < 0.5, np.where(centres[:, 1] < 0, 5, 1), 0)
    fibres = np.where(tags[:, None] == 0, 0.0, [1.0, 0.0, 0.0])
    tissueNodes = np.unique(elements[tags != 0])
    g = dict(g_il=0.344, g_it=0.06, g_in=0.06, g_el=0.117, g_et=0.08, g_en=0.08)
    model = forward.BathModel(dict(positions=positions, elements=elements, tags=tags, fibres=fibres, tissueNodes=tissueNodes),
                              g, 1e-6, 0.74, np.flatnonzero(positions[:, 2] == 1.5), method="lu")
    healthy = np.flatnonzero(tags == 1)[0]
    np.testing.assert_allclose(np.diag(model.solver.sigmaSource[healthy]), [0.344 * 0.117 / 0.461, 0.06 * 0.08 / 0.14, 0.06 * 0.08 / 0.14])
    lesion = np.flatnonzero(tags == 5)[0]
    np.testing.assert_allclose(np.diag(model.solver.sigmaSource[lesion]), 5e-7)
    assert not model.solver.sigmaSource[tags == 0].any()
