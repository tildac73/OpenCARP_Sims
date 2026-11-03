import numpy as np

from efov import forward, groundtruth
from helpers import boxMesh


def model():
    positions, elements = boxMesh([-3, -3, -1], [3, 3, 2], 0.5)
    centres = positions[elements].mean(axis=1)
    tags = np.where(centres[:, 2] < 0, 1, 0)
    fibres = np.where(tags[:, None] == 0, 0.0, [1.0, 0.0, 0.0])
    tissueNodes = np.unique(elements[tags != 0])
    g = dict(g_il=0.344, g_it=0.06, g_in=0.06, g_el=0.117, g_et=0.08, g_en=0.08)
    corner = np.flatnonzero(np.all(positions >= positions.max(axis=0) - 1.0, axis=1))
    return forward.BathModel(dict(positions=positions, elements=elements, tags=tags, fibres=fibres, tissueNodes=tissueNodes),
                             g, 1e-6, 0.74, corner, method="lu")


def test_shares_sum_to_one_and_reconstruct_the_forward_signal():
    m = model()
    p = m.positions[m.tissueNodes]
    t = np.linspace(0, 40, 81)
    vm = -80 + 100 / (1 + np.exp(-(t[None, :] - 10 - (p[:, :1] + 3) / 0.9) / 0.5))     # planar wave in +x
    interpolator = forward.pointInterpolator(m.positions, m.elements, np.array([[0.0, 0.0, 0.5], [2.0, 0.0, 0.5]]))
    row = groundtruth.recordingRow(interpolator, np.array([[0], [1]] + [[0]] * 14), np.r_[1.0, -1.0, np.zeros(14)])
    shares, phi, total = groundtruth.shares(m, row, vm)
    direct = np.array([row @ m.solver.solve(m.fullVm(vm[:, k])) for k in range(len(t))])
    np.testing.assert_allclose(phi, direct, rtol=1e-8, atol=1e-10 * np.abs(direct).max())
    assert abs(total - 1) < 1e-10
    grid = dict(lower=np.array([-3.0, -3.0, -1.0]), voxelMm=1.0, shape=(6, 6, 1))
    assert abs(groundtruth.voxelise(m, shares, grid).sum() - 1) < 1e-10
    curves = groundtruth.cumulativeCurves(m, shares, [0, 0], 0.5)
    assert abs(curves["lateral"][-1] - 1) < 1e-10 and abs(curves["depth"][-1] - 1) < 1e-10


def test_field_distance():
    a, b = np.array([1.0, 0.0, -1.0]), np.array([1.0, 1.0, -1.0])
    assert groundtruth.fieldDistance(a, b) == 1 / 3
