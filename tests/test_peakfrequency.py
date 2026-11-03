import numpy as np

from efov.peakfrequency import peakFrequency

SETTINGS = dict(minimumHz=30.0, maximumHz=500.0, stepHz=1.0, morletOmega0=6.0, magnitudeGateFraction=0.1, minimumPpMv=0.04)


def test_pure_tones_and_bursts_are_recovered():
    fs = 2000.0
    t = np.arange(0, 0.5, 1 / fs)
    start, length = 400, 200
    for f in (100.0, 250.0, 450.0):
        assert peakFrequency(np.cos(2 * np.pi * f * t)[None], fs, start, length, SETTINGS)[0][0] == f
        burst = np.exp(-((t - 0.25) / (6 / f)) ** 2) * np.cos(2 * np.pi * f * (t - 0.25))
        pf, index, _ = peakFrequency(burst[None], fs, start, length, SETTINGS)
        assert abs(pf[0] - f) <= 2 and abs(t[index[0]] - 0.25) < 0.005


def test_small_signals_are_undetectable():
    fs = 2000.0
    t = np.arange(0, 0.5, 1 / fs)
    pf, index, alternative = peakFrequency(0.01 * np.cos(2 * np.pi * 200 * t)[None], fs, 400, 200, SETTINGS)
    assert np.isnan(pf[0]) and index[0] == -1 and np.isnan(alternative[0])


def test_rows_use_their_own_windows():
    fs = 2000.0
    t = np.arange(0, 0.5, 1 / fs)
    signal = np.where(t < 0.25, np.cos(2 * np.pi * 100 * t), np.cos(2 * np.pi * 300 * t))
    pf, _, _ = peakFrequency(np.stack([signal, signal]), fs, np.array([100, 700]), 200, SETTINGS)
    assert pf[0] == 100 and pf[1] == 300
