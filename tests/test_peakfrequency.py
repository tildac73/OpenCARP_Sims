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


def test_trailing_context_removes_the_window_end_bias():
    """FM-36: a deflection 2 ms before the window end is read correctly only with signal after the window.

    Band-passed biphasic deflection (1.5 ms width). Without trailing samples its wavelets are truncated and the
    peak frequency is biased; with 150 ms and with 300 ms of context the result is the same.
    """
    from scipy.signal import butter, sosfilt
    fs, start, length = 2000.0, 400, 121
    sections = butter(2, [30.0, 500.0], btype="bandpass", fs=fs, output="sos")

    def pf(contextSamples):
        t = np.arange(start + length + contextSamples) / fs * 1000.0
        centre = t[start + length - 1] - 2.0
        deflection = -(t - centre) / 1.5 * np.exp(-0.5 * ((t - centre) / 1.5) ** 2)
        return peakFrequency(sosfilt(sections, deflection)[None], fs, start, length, SETTINGS)[0][0]

    assert pf(300) == pf(600)                    # converged at 150 ms of context
    assert abs(pf(0) - pf(300)) > 20             # truncated: biased by tens of Hz
