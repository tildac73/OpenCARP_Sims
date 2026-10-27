"""Ground-truth descriptors of where a recording's signal comes from (readiness plan, Step 4).

Every electrogram s(t) is a sum of element contributions c_e(t) (ground_truth.py). For a family
of nested regions R(k) (all elements with key_e <= k), the partial reconstruction
S_k(t) = sum_{e in R(k)} c_e(t) captures a fraction of the signal

    f(k) = 1 - ||s - S_k||^2 / ||s||^2,

and the "X % extent" is the smallest k after which f stays >= X at every larger k (persistent;
contributions are signed and can cancel, so f need not be monotone). FOV80 in ground_truth.py is
this with cylindrical rings. Families provided here:

  cylinder   lateral distance from the recording anchor (reproduces FOV80)
  sphere     3D distance from the anchor; also the tissue volume inside the sphere
  depth      depth below the tissue surface facing the catheter (transmural profile)
  halfPlane  distance behind a plane parallel to a straight lesion boundary through the anchor;
             the signal from tissue beyond distance d predicts the lesion profile at distance d

and an energy-weighted lateral footprint (second moments of sum_t c_e(t)^2), reported as an
ellipse. Energy weights ignore cancellation between elements; the footprint describes where
strong sources lie, not a reconstruction extent.
"""
import numpy as np

LEVELS = (0.5, 0.8, 0.9, 0.95, 0.99)


def elementCoefficients(geometry, field, conductivity):
    """(E, 4) weights mapping nodal Vm to each element's contribution (common prefactor omitted)."""
    return np.einsum("ec,ec,ecj,e->ej", field, conductivity, geometry.gradients, geometry.volumes)


def binnedTraces(geometry, coefficients, voltage, keys, edges):
    """Sum of contributions per key bin: (bins, time). Elements outside [edges[0], edges[-1]) go to the last bin."""
    bins = np.clip(np.searchsorted(edges, keys, side="right") - 1, 0, len(edges) - 2)
    nodeCount = len(geometry.nodePositions)
    operator = np.zeros((len(edges) - 1, nodeCount))
    np.add.at(operator, (np.repeat(bins, 4), geometry.elementNodes.ravel()), coefficients.ravel())
    return operator @ voltage


def cumulativeFraction(binTraces):
    full = binTraces.sum(axis=0)
    energy = float(full @ full)
    if not np.isfinite(energy) or energy <= 1e-24:
        return np.full(len(binTraces), np.nan)
    cumulative = np.cumsum(binTraces, axis=0)
    return 1.0 - np.sum((full - cumulative) ** 2, axis=1) / energy


def persistentExtent(upperEdges, fraction, level):
    """Smallest upper edge after which fraction >= level at every larger bin; NaN if never."""
    if not np.isfinite(fraction).all():
        return np.nan
    failed = np.flatnonzero(fraction < level)
    index = int(failed[-1] + 1) if len(failed) else 0
    return float(upperEdges[index]) if index < len(upperEdges) else np.nan


def extents(geometry, coefficients, voltage, keys, stepMm, levels=LEVELS):
    edges = np.arange(0.0, keys.max() + 2 * stepMm, stepMm)
    fraction = cumulativeFraction(binnedTraces(geometry, coefficients, voltage, keys, edges))
    return {level: persistentExtent(edges[1:], fraction, level) for level in levels}, edges[1:], fraction


def sphereVolume(geometry, anchor, radiusMm):
    distance = np.linalg.norm(geometry.centres - anchor, axis=1)
    return float(geometry.volumes[distance <= radiusMm].sum()) if np.isfinite(radiusMm) else np.nan


def energyFootprint(geometry, coefficients, voltage, anchor, chunk=50000):
    """Energy-weighted lateral second moments about the anchor -> ellipse (sigma major/minor, axis angle to x in [-90, 90) deg)."""
    energy = np.empty(len(coefficients))
    for first in range(0, len(coefficients), chunk):
        nodes = geometry.elementNodes[first:first + chunk]
        traces = np.einsum("ej,ejt->et", coefficients[first:first + chunk], voltage[nodes])
        energy[first:first + chunk] = np.sum(traces ** 2, axis=1)
    offsets = geometry.centres[:, :2] - anchor[:2]
    weights = energy / energy.sum()
    mean = weights @ offsets
    covariance = (offsets - mean).T @ ((offsets - mean) * weights[:, None])
    values, vectors = np.linalg.eigh(covariance)
    major = vectors[:, 1]
    angle = float((np.degrees(np.arctan2(major[1], major[0])) + 90.0) % 180.0 - 90.0)   # axis, in [-90, 90)
    surfaceZ = geometry.nodePositions[:, 2].max()            # tissue surface facing the catheter
    return dict(sigmaMajorMm=float(np.sqrt(values[1])), sigmaMinorMm=float(np.sqrt(values[0])),
                aspectRatio=float(np.sqrt(values[1] / values[0])), majorAxisAngleToXDeg=angle,
                centroidOffsetXMm=float(mean[0]), centroidOffsetYMm=float(mean[1]),
                energyWeightedDepthMm=float(weights @ (surfaceZ - geometry.centres[:, 2]))), energy
