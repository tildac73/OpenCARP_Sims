"""Unipolar, bipolar and omnipolar recordings from the 4 x 4 grid.

Signals have shape (placement, electrode, time); electrodes are numbered row-major (E0..E15, x
fastest). Every recording returns traces (placement, time) and the 16 electrode weights that made
each trace (placement, 16). The weights are saved: with noise, a different bipole or omnipolar
direction can be selected, and the same weights are applied to the clean signals.

  unipole  E<k>                 the electrode itself (unfiltered)
  bipole   E<c>-E<c+1>|E<c+4>   per placement, the larger peak-to-peak of the x pair and the y pair
                                sharing corner c (ties choose x): an orientation-selected bipole, as EnSite
                                does with the HD Grid. Sweep variable bipoleOrientation (FM-39): 'x' or
                                'y' keeps that pair at every placement instead (fixed orientation)
  omnipole E<v>-E<x>-E<y>       right-angle clique at vertex v; the two orthogonal differences are
                                rotated in angle steps and the direction of largest peak-to-peak kept
                                (a voltage-direction search, not the full Masse/Deno activation method)

Differential signals (bipole, omnipole) are band-pass filtered (efov.measurements: anti-alias,
decimation to the acquisition rate, then the causal band-pass; FM-09). The caller supplies the
filter history.
"""
import numpy as np
from scipy.signal import butter, sosfilt


def bandPass(bandHz, sampleRateHz, order):
    low, high = bandHz
    if not 0 < low < high < sampleRateHz / 2:
        raise ValueError("Filter band must lie between zero and Nyquist")
    return butter(order, bandHz, btype="bandpass", fs=sampleRateHz, output="sos")


def applyFilter(sections, signals):
    """Causal filter along the last (time) axis; None = unfiltered."""
    return signals if sections is None else sosfilt(sections, signals, axis=-1)


def _bipoleCorners():
    return [4 * row + column for row in range(3) for column in range(3)]


def _cliques():
    """(vertex, x neighbour, y neighbour, x sign, y sign) for the four right-angle triples of each square."""
    for corner in _bipoleCorners():
        square = {(0, 0): corner, (1, 0): corner + 1, (0, 1): corner + 4, (1, 1): corner + 5}
        for (xs, ys), vertex in square.items():
            yield vertex, square[(1 - xs, ys)], square[(xs, 1 - ys)], 1 - 2 * xs, 1 - 2 * ys


def allLabels(kind):
    """Every recording of a kind in a fixed order: 16 unipoles, 9 orientation-selected bipoles, 36 omnipolar cliques."""
    if kind == "uni":
        return [f"E{k}" for k in range(16)]
    if kind == "bi":
        return [f"E{c}-E{c + 1}|E{c + 4}" for c in _bipoleCorners()]
    if kind == "omni":
        return [f"E{v}-E{x}-E{y}" for v, x, y, _, _ in _cliques()]
    raise ValueError(f"Unknown recording kind {kind}")


def recordingLabels(setting):
    """'all' or {'uni': [...], 'bi': [...], 'omni': [...]} -> the label dict."""
    if setting == "all":
        return {kind: allLabels(kind) for kind in ("uni", "bi", "omni")}
    for kind, labels in setting.items():
        if kind not in ("uni", "bi", "omni") or not set(labels) <= set(allLabels(kind)):
            raise ValueError(f"Invalid {kind} recording labels: {labels}")
    return setting


def availableLabels(kind):
    if kind == "uni":
        return {f"E{k}" for k in range(16)}
    if kind == "bi":
        return {f"E{c}-E{c + 1}|E{c + 4}" for c in _bipoleCorners()}
    if kind == "omni":
        return {f"E{v}-E{x}-E{y}" for v, x, y, _, _ in _cliques()}
    raise ValueError(f"Unknown recording kind {kind}")


def unipole(unipolar, label):
    k = int(label[1:])
    weights = np.zeros((unipolar.shape[0], 16)); weights[:, k] = 1
    return unipolar[:, k], weights


def bipole(differential, label, orientation="selected"):
    """Bipole of grid square c: the larger of the x and y pairs ('selected'), or always the x or the y pair."""
    corner = int(label.split("-")[0][1:])
    x, y = corner + 1, corner + 4
    xTrace, yTrace = differential[:, x] - differential[:, corner], differential[:, y] - differential[:, corner]
    if orientation == "selected":
        chooseX = np.ptp(xTrace, axis=-1) >= np.ptp(yTrace, axis=-1)
    elif orientation in ("x", "y"):
        chooseX = np.full(len(xTrace), orientation == "x")
    else:
        raise ValueError(f"Bipole orientation must be selected, x or y, not {orientation}")
    basis = np.eye(16)
    weights = np.where(chooseX[:, None], basis[x] - basis[corner], basis[y] - basis[corner])
    return np.where(chooseX[:, None], xTrace, yTrace), weights


def omnipole(differential, label, angleStepDeg):
    vertex, x, y = (int(v[1:]) for v in label.split("-"))
    _, _, _, xSign, ySign = next(c for c in _cliques() if c[:3] == (vertex, x, y))
    xTrace = xSign * (differential[:, x] - differential[:, vertex])
    yTrace = ySign * (differential[:, y] - differential[:, vertex])
    best, bestAngle = np.full(len(xTrace), -np.inf), np.zeros(len(xTrace))
    for angle in np.deg2rad(np.arange(0, 180, angleStepDeg)):
        amplitude = np.ptp(np.cos(angle) * xTrace + np.sin(angle) * yTrace, axis=-1)
        better = amplitude > best
        best[better], bestAngle[better] = amplitude[better], angle
    c, s = np.cos(bestAngle)[:, None], np.sin(bestAngle)[:, None]
    basis = np.eye(16)
    weights = c * xSign * (basis[x] - basis[vertex]) + s * ySign * (basis[y] - basis[vertex])
    return c * xTrace + s * yTrace, weights


def makeRecordings(unipolar, differential, labels, angleStepDeg, bipoleOrientation="selected"):
    """[(kind, label, traces, weights)] for the declared labels {'uni': [...], 'bi': [...], 'omni': [...]}."""
    out = []
    for label in labels.get("uni", []):
        out.append(("uni", label, *unipole(unipolar, label)))
    for label in labels.get("bi", []):
        out.append(("bi", label, *bipole(differential, label, bipoleOrientation)))
    for label in labels.get("omni", []):
        out.append(("omni", label, *omnipole(differential, label, angleStepDeg)))
    return out
