import numpy as np

# scipy.signal.butter(2, [30, 450], btype="bandpass", fs=5000, output="sos")
_SOS_2_30_450_5000 = np.array([
    [0.05017146207089531, 0.10034292414179062, 0.05017146207089531,
     1.0, -1.290948024021348, 0.5002871911080428],
    [1.0, -2.0, 1.0, 1.0, -1.9474979155570544, 0.9490564068347218],
])


def butter(order, band, btype="low", fs=None, output="ba"):
    # fs is compared with a 1e-9 relative tolerance: a sampling rate derived from a time axis that
    # starts far from zero (e.g. 7125 ms) carries ~4e-12 relative float error; scipy would accept it.
    if (order, tuple(float(b) for b in band), btype, output) != (2, (30.0, 450.0), "bandpass", "sos") \
            or abs(float(fs) / 5000.0 - 1.0) > 1e-9:
        raise NotImplementedError("shim only supports the pipeline's declared filter")
    return _SOS_2_30_450_5000.copy()


def sosfilt(sos, x, axis=-1):
    """Cascaded transposed direct-form II biquads, zero initial state (as scipy)."""
    if axis not in (-1, np.ndim(x) - 1):
        raise NotImplementedError("shim filters the last axis only")
    y = np.asarray(x, dtype=np.float64)
    for b0, b1, b2, a0, a1, a2 in np.asarray(sos, dtype=np.float64):
        if a0 != 1.0:
            b0, b1, b2, a1, a2 = b0 / a0, b1 / a0, b2 / a0, a1 / a0, a2 / a0
        out = np.empty_like(y)
        z1 = np.zeros(y.shape[:-1])
        z2 = np.zeros(y.shape[:-1])
        for n in range(y.shape[-1]):
            xn = y[..., n]
            yn = b0 * xn + z1
            z1 = b1 * xn - a1 * yn + z2
            z2 = b2 * xn - a2 * yn
            out[..., n] = yn
        y = out
    return y
