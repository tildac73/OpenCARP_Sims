"""Geometry used by the boundary pilot. All coordinates in this module are mm."""

import re
import numpy as np


def signedDistance(pointsMm, boundary):
    """Positive on the lesion side; the intact condition uses the same virtual boundary."""
    pointsMm = np.asarray(pointsMm)
    if boundary["shape"] == "straight":
        return boundary["positionMm"] - pointsMm[..., 1]
    if boundary["shape"] == "circle":
        return boundary["radiusMm"] - np.linalg.norm(
            pointsMm[..., :2] - np.asarray(boundary["centreMm"]), axis=-1
        )
    raise ValueError("Boundary shape must be straight or circle")


def recordingGeometry(
    electrodePositions, samplePositions, kind, label, weights, boundary, condition
):
    """Keep the actual selected pair distinct from the three-electrode selection clique."""
    labelledIds = np.array([int(value) for value in re.findall(r"E(\d+)", label)])
    selectedIds = np.flatnonzero(np.abs(weights) > 1e-12)
    if kind == "uni":
        anchor = electrodePositions[labelledIds[0]]
    elif kind == "bi":
        if len(selectedIds) != 2:
            raise ValueError("A bipolar recording must use exactly two electrodes")
        anchor = electrodePositions[selectedIds].mean(axis=0)
    else:
        anchor = electrodePositions[labelledIds].mean(axis=0)
    distances = signedDistance(electrodePositions[labelledIds], boundary)
    sampleDistances = signedDistance(samplePositions[labelledIds], boundary)
    return dict(
        anchorXmm=float(anchor[0]),
        anchorYmm=float(anchor[1]),
        anchorZmm=float(anchor[2]),
        distanceInsideMm=float(signedDistance(anchor, boundary)),
        distanceViableMm=float(-signedDistance(anchor, boundary)),
        meanElectrodeDistanceInsideMm=float(distances.mean()),
        cliqueDistanceInsideMm=float(
            signedDistance(electrodePositions[labelledIds].mean(axis=0), boundary)
        ),
        selectedPairDistanceInsideMm=(
            float(signedDistance(anchor, boundary)) if kind == "bi" else np.nan
        ),
        electrodeIds=",".join(map(str, labelledIds)),
        selectedElectrodeIds=",".join(map(str, selectedIds)),
        electrodesOverLesion=int(np.count_nonzero(distances > 0)) if condition == "lesion" else 0,
        footprintCrossesBoundary=bool(sampleDistances.min() < 0 < sampleDistances.max()),
        anchorConvention=(
            "electrode"
            if kind == "uni"
            else "selectedPairMidpoint" if kind == "bi" else "cliqueCentroid"
        ),
    )


def makeElectrodeLayout(config):
    """Same row-major 4 x 4 grid and eight cylinder samples as the original notebook."""
    electrodeConfig = config["electrodes"]
    offsets = np.asarray(electrodeConfig["gridOffsetsMm"])
    sampleCount = electrodeConfig["axialSamples"] * electrodeConfig["angularSamples"]
    halfLength = electrodeConfig["lengthMm"] / 2
    # Midpoints of axial strips; equally spaced samples around the lateral surface.
    axialOffsets = -halfLength + (np.arange(electrodeConfig["axialSamples"]) + 0.5) * (
        2 * halfLength / electrodeConfig["axialSamples"]
    )
    angles = (
        np.arange(electrodeConfig["angularSamples"]) * 2 * np.pi / electrodeConfig["angularSamples"]
    )
    radius = electrodeConfig["radiusMm"]
    topSurface = config["slabSizeMm"][2] / 2
    centrePositions, samplePositions, placementRows = [], [], []
    for gap in electrodeConfig["gapsMm"]:
        for centreY in electrodeConfig["centreYmm"]:
            centreX = electrodeConfig["centreXmm"]
            placementRows.append([centreX, centreY, gap])
            positions, samples = [], []
            for rowOffset in offsets:
                for columnOffset in offsets:
                    centre = np.array(
                        [centreX + columnOffset, centreY + rowOffset, topSurface + gap + radius]
                    )
                    positions.append(centre)
                    samples.append(
                        [
                            centre + [axial, radius * np.cos(angle), radius * np.sin(angle)]
                            for axial in axialOffsets
                            for angle in angles
                        ]
                    )
            centrePositions.append(positions)
            samplePositions.append(samples)
    samplePositions = np.asarray(samplePositions)
    # Shared physical electrodes are recovered only once, then gathered into placements.
    uniqueSamples, inverse = np.unique(samplePositions.reshape(-1, 3), axis=0, return_inverse=True)
    return dict(
        centres=np.asarray(centrePositions),
        samples=samplePositions,
        placements=np.asarray(placementRows),
        recoveryPoints=uniqueSamples,
        recoveryIndex=inverse.reshape(len(placementRows), 16, sampleCount),
    )
