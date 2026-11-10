"""Peak frequency: an approximation of EnSite OT Near Field (FM-10).

Published description (Europace 2024;26:euae154): proprietary and wavelet-derived; a peak-frequency
trace over time; each point is assigned the highest peak frequency and its time; bipolar and
omnipolar signals only; computed only when peak-to-peak >= 0.04 mV; voltage taken within +-30 ms of
the peak-frequency annotation. Mother wavelet, scales and bounds are not published.

This implementation (assumptions, fixed.analysis.peakFrequency):
  * complex Morlet continuous wavelet transform, omega0 = 6, by FFT; analytic wavelet with
    Psi_hat(s w) = 2 exp(-(s w - omega0)^2 / 2) for w > 0, scale s = omega0 / (2 pi f). This
    L1 normalisation makes |W| equal the amplitude of a sinusoid at its own frequency, so the
    frequency of the largest magnitude is not biased toward low frequencies;
  * frequencies minimumHz..maximumHz in stepHz steps;
  * PF trace: at each sample, the frequency of the largest |W|; samples whose largest |W| is below
    magnitudeGateFraction of the window maximum are ignored;
  * 'ensite' PF = highest remaining trace value, t_PF = its time (if several samples share it, the
    one with the largest |W|);
  * 'global' PF = frequency of the largest |W| in the window. Co-primary with 'ensite' (FM-27):
    with white noise the highest trace value is drawn toward high frequencies as the signal falls,
    the global maximum much less so.
The transform runs over the whole filtered segment (filter history + window + trailing context,
FM-36) and is read only inside the measurement window, so wavelets are cut at neither window edge.
The transform is not causal: a wavelet at frequency f has time width omega0 / (2 pi f) (31.8 ms at
30 Hz), and without signal after the window its support is truncated for annotations near the
window end. Four widths at 30 Hz are about 127 ms; trailingContextMs = 150.
"""
import numpy as np


def frequenciesHz(settings):
    return np.arange(settings["minimumHz"], settings["maximumHz"] + 1e-9, settings["stepHz"])


def scalogram(signals, sampleRateHz, frequencies, omega0):
    """|W| of real signals (N, T) -> (N, F, T), L1-normalised complex Morlet."""
    signals = np.atleast_2d(np.asarray(signals, float))
    length = signals.shape[-1]
    size = int(2 ** np.ceil(np.log2(2 * length)))
    spectrum = np.fft.fft(signals, size, axis=-1)                                  # (N, n)
    omega = 2 * np.pi * np.fft.fftfreq(size, d=1.0 / sampleRateHz)                # rad/s
    scales = omega0 / (2 * np.pi * np.asarray(frequencies, float))                 # s
    wavelet = 2.0 * np.exp(-0.5 * (scales[:, None] * omega[None, :] - omega0) ** 2) * (omega[None, :] > 0)
    return np.abs(np.fft.ifft(spectrum[:, None, :] * wavelet[None, :, :], axis=-1)[..., :length])


def windowed(array, starts, length):
    """array (N, ..., T) -> (N, ..., length), each row from its own start sample."""
    starts = np.broadcast_to(np.asarray(starts, int), (array.shape[0],))
    index = starts[:, None] + np.arange(length)[None, :]
    index = index.reshape((array.shape[0],) + (1,) * (array.ndim - 2) + (length,))
    return np.take_along_axis(array, index, axis=-1)


def peakFrequency(signals, sampleRateHz, starts, length, settings, block=32):
    """PF (Hz), t_PF (sample index in the segment) and the global PF for each signal (N, T).

    starts, length: each signal's measurement window inside its segment (first sample, number of
    samples). Signals with window peak-to-peak below settings['minimumPpMv'] get NaN (undetectable).
    The caller passes the band-passed segment, which should continue beyond each window (trailing context, FM-36).
    """
    signals = np.atleast_2d(np.asarray(signals, float))
    frequencies = frequenciesHz(settings)
    count = len(signals)
    starts = np.broadcast_to(np.asarray(starts, int), (count,))
    pf, index, pfGlobal = np.full(count, np.nan), np.full(count, -1), np.full(count, np.nan)
    detectable = np.ptp(windowed(signals, starts, length), axis=1) >= settings["minimumPpMv"]
    for first in range(0, count, block):
        rows = np.arange(first, min(first + block, count))
        rows = rows[detectable[rows]]
        if not len(rows):
            continue
        magnitude = windowed(scalogram(signals[rows], sampleRateHz, frequencies, settings["morletOmega0"]), starts[rows], length)  # (n, F, W)
        best = magnitude.max(axis=1)                                               # (n, W)
        trace = frequencies[magnitude.argmax(axis=1)]                              # (n, W)
        kept = best >= settings["magnitudeGateFraction"] * best.max(axis=1, keepdims=True)
        masked = np.where(kept, trace, -np.inf)
        highest = masked.max(axis=1)
        tieBreak = np.where(masked == highest[:, None], best, -np.inf)
        pf[rows], index[rows] = highest, starts[rows] + tieBreak.argmax(axis=1)
        flat = magnitude.reshape(len(rows), -1).argmax(axis=1)
        pfGlobal[rows] = frequencies[np.unravel_index(flat, magnitude.shape[1:])[0]]
    return pf, index, pfGlobal
