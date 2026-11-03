import numpy as np
import pytest

from efov import recordings as rec


def test_saved_weights_reproduce_every_trace():
    rng = np.random.default_rng(1)
    uni, diff = rng.normal(size=(5, 16, 40)), rng.normal(size=(5, 16, 40))
    labels = {k: sorted(rec.availableLabels(k)) for k in ("uni", "bi", "omni")}
    for kind, label, traces, weights in rec.makeRecordings(uni, diff, labels, 2):
        source = uni if kind == "uni" else diff
        np.testing.assert_allclose(np.einsum("pe,pet->pt", weights, source), traces, atol=1e-12)
    assert len(labels["bi"]) == 9 and len(labels["omni"]) == 36


def test_bipole_selects_the_larger_pair():
    diff = np.zeros((2, 16, 10))
    diff[0, 6, 5] = 2.0          # x pair E5-E6 larger in placement 0
    diff[1, 9, 5] = 3.0          # y pair E5-E9 larger in placement 1
    traces, weights = rec.bipole(diff, "E5-E6|E9")
    assert weights[0, 6] == 1 and weights[0, 5] == -1 and weights[1, 9] == 1
    np.testing.assert_allclose(np.ptp(traces, axis=1), [2.0, 3.0])


def test_omnipole_finds_the_wavefront_direction():
    t = np.linspace(0, 1, 50)
    wave = np.sin(2 * np.pi * t)
    angle = np.deg2rad(30)
    diff = np.zeros((1, 16, 50))
    diff[0, 6] = np.cos(angle) * wave            # x difference E6 - E5
    diff[0, 9] = np.sin(angle) * wave            # y difference E9 - E5
    traces, weights = rec.omnipole(diff, "E5-E6-E9", 2)
    np.testing.assert_allclose(np.ptp(traces), np.ptp(wave), rtol=1e-3)


def test_filter_rejects_band_above_nyquist():
    with pytest.raises(ValueError):
        rec.bandPass([30, 3000], 5000, 2)
