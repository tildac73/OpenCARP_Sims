"""Lesion distance and the catheter layout. All coordinates in mm.

Coordinates: x along propagation, y across the lesion boundary, z through the wall; the tissue
surface facing the catheter is z = +wall/2. Distance convention: positive inside the lesion
(y < boundary); the intact run uses the same virtual boundary.
"""
import re

import numpy as np

from .config import centreYmm, slabSizeMm


def signedDistance(pointsMm, boundaryYmm):
    """Positive on the lesion side (y < boundary)."""
    return boundaryYmm - np.asarray(pointsMm)[..., 1]


def electrodeLayout(config):
    """Every catheter placement: electrode centres, surface samples, and a de-duplicated sample list.

    Electrodes are numbered row-major (E0..E15, x fastest), as the original notebook. Each cylinder
    (axis along x) is sampled at the midpoints of `axialSamples` strips and `angularSamples` equally
    spaced angles on its lateral surface. Overlapping grids share physical electrodes, so samples are
    stored once (samplePoints) and gathered per placement (sampleIndex).
    """
    e = config["electrodes"]
    offsets = np.asarray(e["gridOffsetsMm"])
    radius, half = e["radiusMm"], e["lengthMm"] / 2
    axial = -half + (np.arange(e["axialSamples"]) + 0.5) * (2 * half / e["axialSamples"])
    angles = np.arange(e["angularSamples"]) * 2 * np.pi / e["angularSamples"]
    surface = slabSizeMm(config)[2] / 2
    centres, samples, placements = [], [], []
    for gap in e["gapsMm"]:
        for y in centreYmm(config):
            placements.append([e["centreXmm"], y, gap])
            gridCentres = [np.array([e["centreXmm"] + dx, y + dy, surface + gap + radius])
                           for dy in offsets for dx in offsets]
            centres.append(gridCentres)
            samples.append([[c + [a, radius * np.cos(t), radius * np.sin(t)] for a in axial for t in angles]
                            for c in gridCentres])
    samples = np.asarray(samples)                                     # (placements, 16, samples, 3)
    unique, inverse = np.unique(samples.reshape(-1, 3), axis=0, return_inverse=True)
    return dict(centres=np.asarray(centres), samples=samples, placements=np.asarray(placements),
                samplePoints=unique, sampleIndex=inverse.reshape(samples.shape[:3]))


def recordingGeometry(electrodeCentres, kind, label, weights, boundaryYmm):
    """Anchor of a recording and its distance from the boundary.

    Anchors: unipole = its electrode; bipole = midpoint of the pair actually selected (the saved
    weights); omnipole = centroid of its three-electrode clique.
    """
    labelled = np.array([int(v) for v in re.findall(r"E(\d+)", label)])
    selected = np.flatnonzero(np.abs(weights) > 1e-12)
    if kind == "uni":
        anchor = electrodeCentres[labelled[0]]
    elif kind == "bi":
        if len(selected) != 2:
            raise ValueError("A bipolar recording must use exactly two electrodes")
        anchor = electrodeCentres[selected].mean(axis=0)
    else:
        anchor = electrodeCentres[labelled].mean(axis=0)
    return dict(anchorXmm=float(anchor[0]), anchorYmm=float(anchor[1]), anchorZmm=float(anchor[2]),
                distanceInsideMm=float(signedDistance(anchor, boundaryYmm)),
                cliqueDistanceInsideMm=float(signedDistance(electrodeCentres[labelled].mean(axis=0), boundaryYmm)),
                selectedElectrodes=",".join(map(str, selected)))
