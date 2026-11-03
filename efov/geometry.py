"""Lesion distance, the catheter layout and the electrodes as mesh node sets. All coordinates in mm.

Coordinates: x along the reference wave (wave angle 0), y across the lesion boundary, z through the
wall; the catheter-facing tissue surface is z = +wall/2. Distance convention: positive inside the
lesion (y < boundary); the intact run uses the same virtual boundary.

Electrodes (FM-14): each electrode is a cylinder (axis along x, radius r, length L) whose centre sits
gap + r above the tissue. Its sensing set is every mesh node inside the cylinder; its recorded
potential is the mean phi_e over that set, and its lead field is the response to a total current
injected over the same set, so reciprocity holds for the discrete problem.
"""
import re

import numpy as np

from .config import centreYmm, gridOffsetsMm, slabSizeMm


def signedDistance(pointsMm, boundaryYmm):
    """Positive on the lesion side (y < boundary)."""
    return boundaryYmm - np.asarray(pointsMm)[..., 1]


def electrodeLayout(config):
    """Every catheter placement and its 16 electrode centres; electrodes shared by overlapping grids are stored once.

    Electrodes are numbered row-major (E0..E15, x fastest). Returns placements (P, 3) = (centre x, centre y,
    gap), centres (P, 16, 3), uniqueCentres (U, 3) and electrodeIndex (P, 16) into uniqueCentres.
    """
    v, e = config["values"], config["fixed"]["electrodes"]
    offsets = gridOffsetsMm(v["gridSpacingMm"])
    surface = slabSizeMm(config)[2] / 2
    placements, centres = [], []
    for gap in v["gapsMm"]:
        for y in centreYmm(config):
            placements.append([e["centreXmm"], y, gap])
            centres.append([[e["centreXmm"] + dx, y + dy, surface + gap + v["electrodeRadiusMm"]] for dy in offsets for dx in offsets])
    centres = np.round(np.asarray(centres, float), 9)
    unique, index = np.unique(centres.reshape(-1, 3), axis=0, return_inverse=True)
    return dict(placements=np.asarray(placements), centres=centres, uniqueCentres=unique,
                electrodeIndex=index.reshape(centres.shape[:2]))


def electrodeNodeSets(positionsMm, candidateNodes, uniqueCentres, radiusMm, lengthMm):
    """Mesh nodes inside each electrode cylinder (axis along x), from the candidate nodes; list of arrays."""
    from scipy.spatial import cKDTree
    candidates = np.asarray(candidateNodes, np.int64)
    points = positionsMm[candidates]
    tree = cKDTree(points)
    reach = float(np.hypot(radiusMm, lengthMm / 2)) + 1e-9
    sets = []
    for centre in uniqueCentres:
        near = np.asarray(tree.query_ball_point(centre, reach), np.int64)
        local = points[near] - centre
        inside = (np.abs(local[:, 0]) <= lengthMm / 2 + 1e-9) & (np.hypot(local[:, 1], local[:, 2]) <= radiusMm + 1e-9)
        nodes = np.sort(candidates[near[inside]])
        if not len(nodes):
            raise ValueError(f"Electrode at {centre} mm contains no mesh node of the phi_e output box")
        sets.append(nodes)
    return sets


def electrodeAveraging(nodeSets, columnOfNode):
    """Sparse (U x columns) matrix: mean over each electrode's nodes, columns = positions in the phi_e output."""
    from scipy import sparse
    rows, cols, values = [], [], []
    for k, nodes in enumerate(nodeSets):
        columns = columnOfNode[nodes]
        if np.any(columns < 0):
            raise ValueError("An electrode node is not in the phi_e output (enlarge the output box)")
        rows += [k] * len(nodes); cols += list(columns); values += [1.0 / len(nodes)] * len(nodes)
    return sparse.csr_matrix((values, (rows, cols)), shape=(len(nodeSets), int(columnOfNode.max()) + 1))


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
                selectedElectrodes=",".join(map(str, selected)))
