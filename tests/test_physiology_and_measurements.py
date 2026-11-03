import numpy as np

from efov import measurements, physiology
from helpers import finalConfig


def test_activation_plane_speed_and_direction():
    rng = np.random.default_rng(0)
    p = rng.uniform(-5, 5, size=(200, 3))
    t = 10 + p[:, 0] / 0.9                       # 0.9 mm/ms along +x
    plane = physiology.activationPlane(p, t)
    assert abs(plane["speedMPerS"] - 0.9) < 1e-9 and plane["directionErrorDeg"] < 1e-6


def test_activation_times_interpolate_the_crossing():
    t = np.arange(0, 10, 1.0)
    v = np.where(t < 4, -80.0, 20.0)[:, None] * np.ones((1, 2))
    v[4, 1] = -30.0                              # second node crosses -20 between 4 and 5 ms
    times, count = physiology.activationTimes(t, v, -20.0)
    np.testing.assert_allclose(times, [3 + 60 / 100, 4 + 10 / 50])
    assert count.tolist() == [1, 1]


def test_measurement_slices_and_seeds():
    c = finalConfig()
    t = np.round(np.arange(1108.0, 1560.1, 0.2), 6)   # window start - 2 x history .. window end
    raw, measure = measurements.measurementSlices(t, c, [8.0, 60.0])
    assert np.isclose(t[raw][0], 1500 + 8 - 200) and np.isclose(t[raw][measure][0], 1508.0)
    assert np.isclose(t[raw][measure][-1], 1559.8)      # half-open window
    assert measurements.stableSeed(1701, "lesion", 2) == measurements.stableSeed(1701, "lesion", 2)
    assert measurements.stableSeed(1701, "lesion", 2) != measurements.stableSeed(1701, "intact", 2)
