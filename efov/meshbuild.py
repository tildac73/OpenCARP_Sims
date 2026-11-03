"""Turn the carputils block mesh into the run mesh, and select its node sets. No openCARP calls here.

    runMesh(blockMesh, spec)        trim the epicardial bath (FM-05) and write the fibre field (FM-06)
    stimulusNodes(...)              viable-tissue band at the trailing edge of the wave direction (FM-17)
    groundNodes(...)                bath nodes of the corner cube (Leenknegt et al. 2023)
    boxNodes(...)                   nodes inside the phi_e output box (FM-20)

Tags: 0 = bath (zero fibre vector, so openCARP applies g_bath), 1 = viable tissue, 5 = lesion.
"""
import numpy as np

from .fibres import fibreVectors

BATH_TAG, TISSUE_TAG, LESION_TAG = 0, 1, 5


def runMesh(positions, elements, tags, wallMm, epicardialMedium, fibres):
    """Run mesh from the block mesh: drop the bath below the tissue when the epicardium is no-flux; set fibres.

    The block mesh has bath on every side. With epicardialMedium 'none' every bath element whose
    centroid lies below the tissue (z < -wall/2) is removed and the nodes renumbered, so the
    tissue's far surface becomes a no-flux boundary. Lateral bath beside the tissue and the bath
    above it are kept.
    """
    centres = positions[elements].mean(axis=1)
    keep = np.ones(len(elements), bool)
    if epicardialMedium == "none":
        keep = ~((tags == BATH_TAG) & (centres[:, 2] < -wallMm / 2))
    elif epicardialMedium != "blood":
        raise ValueError("epicardialMedium must be 'none' or 'blood'")
    elements, tags, centres = elements[keep], tags[keep], centres[keep]
    used = np.unique(elements)
    renumber = np.full(len(positions), -1, np.int64)
    renumber[used] = np.arange(len(used))
    elements = renumber[elements]
    positions = positions[used]
    vectors = np.zeros((len(elements), 3))
    tissue = tags != BATH_TAG
    vectors[tissue] = fibreVectors(centres[tissue], wallMm, fibres)
    return positions, elements, tags, vectors


def tissueNodeMask(nodeCount, elements, tags):
    mask = np.zeros(nodeCount, bool)
    mask[np.unique(elements[tags != BATH_TAG])] = True
    return mask


def stimulusNodes(positions, elements, tags, waveAngleDeg, stripWidthMm):
    """Viable-tissue nodes within stripWidth of the trailing edge, along d = (cos a, -sin a).

    a = 0: the strip at x min (wave along +x, parallel to the boundary). a = 90: the strip at y max
    (wave toward the lesion, -y). Oblique angles stimulate a corner band. Nodes touching a lesion
    element are excluded, so the lesion is never stimulated directly.
    """
    angle = np.radians(waveAngleDeg)
    viable = np.zeros(len(positions), bool)
    viable[np.unique(elements[tags == TISSUE_TAG])] = True
    viable[np.unique(elements[tags == LESION_TAG])] = False
    if not viable.any():
        raise ValueError("No viable tissue nodes")
    projection = positions[:, 0] * np.cos(angle) - positions[:, 1] * np.sin(angle)
    selected = viable & (projection <= projection[viable].min() + stripWidthMm + 1e-9)
    return np.flatnonzero(selected)


def waveDirection(waveAngleDeg):
    angle = np.radians(waveAngleDeg)
    return np.array([np.cos(angle), -np.sin(angle)])


def groundNodes(positions, elements, tags, cubeMm, allowTissue=False):
    """Nodes within cubeMm of the (max x, max y, max z) corner of the mesh."""
    corner = positions.max(axis=0)
    nodes = np.flatnonzero(np.all(positions >= corner - cubeMm - 1e-9, axis=1))
    if not allowTissue and np.any(tissueNodeMask(len(positions), elements, tags)[nodes]):
        raise ValueError("Ground cube reaches tissue nodes")
    if not len(nodes):
        raise ValueError("Ground cube holds no node")
    return nodes


def boxNodes(positions, lower, upper):
    """Nodes inside the axis-aligned box [lower, upper] (mm)."""
    lower, upper = np.asarray(lower) - 1e-9, np.asarray(upper) + 1e-9
    return np.flatnonzero(np.all((positions >= lower) & (positions <= upper), axis=1))
