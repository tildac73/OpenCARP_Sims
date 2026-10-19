"""The three recording types used by the 4 × 4 boundary-pilot catheter.

Signals have shape (placement, electrode, time). Each Recording contains traces
(placement, time) and weights (placement, 16). Saving these weights is essential:
a noisy signal may select a different bipolar pair or omnipolar direction.
"""

from dataclasses import dataclass

import numpy as np
from scipy.signal import butter, sosfilt


@dataclass
class Recording:
    kind: str
    label: str
    traces: np.ndarray
    weights: np.ndarray


def makeFilter(bandHz, sampleRateHz, filterOrder):
    """Return a causal Butterworth band-pass filter, or None for unfiltered data."""
    if bandHz is None:
        return None
    lowHz, highHz = bandHz
    if not 0 < lowHz < highHz < sampleRateHz / 2:
        raise ValueError("Filter frequencies must lie between zero and Nyquist")
    return butter(filterOrder, bandHz, btype="bandpass", fs=sampleRateHz, output="sos")


def applyFilter(filterSections, signals):
    """Filter the last (time) axis. The caller supplies the preceding filter history."""
    if filterSections is None:
        return signals
    return sosfilt(filterSections, signals, axis=-1)


def makeUnipolarRecordings(unipolar):
    recordings = []
    placementCount = unipolar.shape[0]
    for electrodeIndex in range(16):
        weights = np.zeros((placementCount, 16))
        weights[:, electrodeIndex] = 1
        recordings.append(
            Recording("uni", f"E{electrodeIndex}", unipolar[:, electrodeIndex], weights)
        )
    return recordings


def makeBipolarRecordings(differential):
    """For each of nine squares, choose the larger of two orthogonal bipoles.

    The pairs share the lower-index corner. Equal amplitudes choose the x pair.
    The label names all three candidates; the saved weights identify the pair.
    """
    recordings = []
    placementCount = differential.shape[0]
    electrodeBasis = np.eye(16)
    for row in range(3):
        for column in range(3):
            corner = 4 * row + column
            xNeighbour = corner + 1
            yNeighbour = corner + 4
            xBipole = differential[:, xNeighbour] - differential[:, corner]
            yBipole = differential[:, yNeighbour] - differential[:, corner]
            chooseX = np.ptp(xBipole, axis=-1) >= np.ptp(yBipole, axis=-1)
            traces = np.where(chooseX[:, None], xBipole, yBipole)
            weights = np.empty((placementCount, 16))
            weights[chooseX] = electrodeBasis[xNeighbour] - electrodeBasis[corner]
            weights[~chooseX] = electrodeBasis[yNeighbour] - electrodeBasis[corner]
            label = f"E{corner}-E{xNeighbour}|E{yNeighbour}"
            recordings.append(Recording("bi", label, traces, weights))
    return recordings


def electrodeCliques():
    """Yield the four right-angle triples in each square, with their axis signs."""
    for row in range(3):
        for column in range(3):
            lowerLeft = 4 * row + column
            corners = {
                (0, 0): lowerLeft,
                (1, 0): lowerLeft + 1,
                (0, 1): lowerLeft + 4,
                (1, 1): lowerLeft + 5,
            }
            for (xSide, ySide), vertex in corners.items():
                xNeighbour = corners[(1 - xSide, ySide)]
                yNeighbour = corners[(xSide, 1 - ySide)]
                xSign = 1 - 2 * xSide
                ySign = 1 - 2 * ySide
                yield vertex, xNeighbour, yNeighbour, xSign, ySign


def makeOmnipolarRecordings(differential, angleStepDeg):
    """Rotate two orthogonal bipolar signals and keep the largest peak-to-peak.

    All 36 right-angle cliques are retained. We scan one angle at a time to avoid
    allocating an extra large (placement, angle, time) array.
    """
    if not 0 < angleStepDeg <= 180:
        raise ValueError("Omnipolar angle step must be in (0, 180] degrees")
    angles = np.deg2rad(np.arange(0, 180, angleStepDeg))
    placementCount = differential.shape[0]
    electrodeBasis = np.eye(16)
    recordings = []
    for vertex, xNeighbour, yNeighbour, xSign, ySign in electrodeCliques():
        xBipole = xSign * (differential[:, xNeighbour] - differential[:, vertex])
        yBipole = ySign * (differential[:, yNeighbour] - differential[:, vertex])
        bestAmplitude = np.full(placementCount, -np.inf)
        bestAngle = np.zeros(placementCount)
        for angle in angles:
            rotatedSignal = np.cos(angle) * xBipole + np.sin(angle) * yBipole
            amplitude = np.ptp(rotatedSignal, axis=-1)
            improved = amplitude > bestAmplitude
            bestAmplitude[improved] = amplitude[improved]
            bestAngle[improved] = angle
        cosine = np.cos(bestAngle)[:, None]
        sine = np.sin(bestAngle)[:, None]
        traces = cosine * xBipole + sine * yBipole
        weights = cosine * xSign * (
            electrodeBasis[xNeighbour] - electrodeBasis[vertex]
        ) + sine * ySign * (electrodeBasis[yNeighbour] - electrodeBasis[vertex])
        label = f"E{vertex}-E{xNeighbour}-E{yNeighbour}"
        recordings.append(Recording("omni", label, traces, weights))
    return recordings


def makeRecordings(unipolar, differential, angleStepDeg):
    """Build 16 unipoles, 9 selected bipoles, then 36 omnipolar recordings."""
    recordings = makeUnipolarRecordings(unipolar)
    recordings.extend(makeBipolarRecordings(differential))
    recordings.extend(makeOmnipolarRecordings(differential, angleStepDeg))
    return recordings
