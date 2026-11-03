import numpy as np

from efov import fibres, meshbuild
from helpers import boxWithBath


def test_no_flux_epicardium_removes_only_the_bath_below_the_tissue():
    positions, elements, tags = boxWithBath([4, 4], 1.0, 1.0, 0.5)
    p, e, t, f = meshbuild.runMesh(positions, elements, tags, 1.0, "none", {"model": "uniform", "angleDeg": 0.0})
    centres = p[e].mean(axis=1)
    assert np.all(centres[t == 0][:, 2] > -0.5) and (t == 1).sum() == (tags == 1).sum()
    assert p[:, 2].min() == -0.5 and e.max() == len(p) - 1
    np.testing.assert_allclose(f[t == 1], [[1.0, 0.0, 0.0]] * (t == 1).sum())
    assert np.all(f[t == 0] == 0)
    _, _, t2, _ = meshbuild.runMesh(positions, elements, tags, 1.0, "blood", {"model": "uniform", "angleDeg": 0.0})
    assert len(t2) == len(tags)


def test_bilayer_fibres_switch_at_mid_wall():
    spec = {"model": "bilayer", "endocardialAngleDeg": 0.0, "epicardialAngleDeg": 90.0, "transitionDepthFraction": 0.5, "transitionWidthMm": 0.0}
    vectors = fibres.fibreVectors(np.array([[0, 0, 0.4], [0, 0, -0.4]]), 1.0, spec)   # depth 0.1 and 0.9 mm
    np.testing.assert_allclose(vectors, [[1, 0, 0], [0, 1, 0]], atol=1e-12)
    linear = fibres.fibreAnglesDeg(np.array([0.0, 0.5, 1.0]), 1.0, {"model": "linearRotation", "endocardialAngleDeg": 0, "epicardialAngleDeg": 90})
    np.testing.assert_allclose(linear, [0, 45, 90])


def test_stimulus_band_follows_the_wave_angle_and_avoids_the_lesion():
    positions, elements, tags = boxWithBath([8, 8], 1.0, 1.0, 0.5)
    centres = positions[elements].mean(axis=1)
    tags = np.where((tags == 1) & (centres[:, 1] < 0), 5, tags)
    along = meshbuild.stimulusNodes(positions, elements, tags, 0.0, 0.5)
    assert np.allclose(positions[along, 0].max(), -3.5) and positions[along, 1].min() >= 0
    toward = meshbuild.stimulusNodes(positions, elements, tags, 90.0, 0.5)
    assert positions[toward, 1].min() >= 3.5                                         # far viable edge, travelling to -y
    np.testing.assert_allclose(meshbuild.waveDirection(90.0), [0, -1], atol=1e-12)


def test_ground_cube_and_box_nodes():
    positions, elements, tags = boxWithBath([4, 4], 1.0, 1.0, 0.5)
    ground = meshbuild.groundNodes(positions, elements, tags, 0.5)
    assert np.all(positions[ground] >= positions.max(axis=0) - 0.5 - 1e-9)
    inside = meshbuild.boxNodes(positions, [-1, -1, 0.5], [1, 1, 1.0])
    assert np.all(np.abs(positions[inside, :2]) <= 1 + 1e-9) and len(inside) == 5 * 5 * 2
