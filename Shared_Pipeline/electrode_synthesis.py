"""Extracellular potential at arbitrary points, computed from saved Vm (readiness plan, Step 2).

Purpose: once validated against openCARP's own recovery, catheter geometry and standoff can
be changed after a simulation without re-running propagation.

Forward model (intended to match openCARP phie recovery as run in boundaryPilotV2):
monodomain sources in an unbounded, homogeneous, isotropic medium of conductivity sigma_b,

    phi(x, t) = -1/(4 pi sigma_b) * sum_e  (G_m,e grad Vm_e(t)) . (x - c_e) / |x - c_e|^3 * vol_e

  G_m,e   monodomain conductivity of element e, per principal axis G_i G_e / (G_i + G_e)
  grad Vm linear-tetrahedron gradient of nodal Vm (constant per element)
  c_e     element centroid (one-point quadrature; least accurate close to the electrode)

ASSUMPTIONS TO VERIFY (not read from the openCARP manual, which could not be accessed):
  * the monodomain (not intracellular) conductivity is the source conductivity;
  * sigma_b = 1.0 S/m, the documented default of gregion g_bath, applies to recovery;
  * the sign and the 1/(4 pi) prefactor.
These are inferred from the Part 1 reconstruction scale (-0.0621 vs predicted -0.0622) and are
tested by 3D_Code/numerics/verify_synthesis.py against phie_recovery.igb. Until that check
passes, treat outputs as unverified.

Units: positions mm, Vm mV, conductivity S/m -> potential mV.
"""
import numpy as np

import lead_field


def monodomainConductivity(intracellular, extracellular):
    """Per-axis series combination G_i G_e / (G_i + G_e); valid for coaxial diagonal tensors."""
    intracellular = np.asarray(intracellular, float)
    extracellular = np.asarray(extracellular, float)
    if intracellular.shape != extracellular.shape:
        raise ValueError("Intracellular and extracellular conductivity arrays differ in shape")
    if np.any(intracellular <= 0) or np.any(extracellular <= 0):
        raise ValueError("Conductivities must be positive")
    return intracellular * extracellular / (intracellular + extracellular)


def sourceConductivity(geometry, source="monodomain"):
    """Source conductivity per element: (E, 3) diagonal values when fibres are along x
    (geometry.fibreFrames is None, the v2-pilot case), else (E, 3, 3) tensors in mesh axes."""
    if source == "monodomain":
        if geometry.extracellularConductivity is None:
            raise ValueError("Geometry lacks extracellular conductivity; read it with ground_truth.readElementGeometry")
        principal = monodomainConductivity(geometry.conductivity, geometry.extracellularConductivity)
    elif source == "intracellular":
        principal = np.asarray(geometry.conductivity, float)
    else:
        raise ValueError("source must be 'monodomain' or 'intracellular'")
    frames = getattr(geometry, "fibreFrames", None)
    if frames is None:
        return principal
    import fibres
    return fibres.tensors(principal, frames)


def conductivityTimesGradients(conductivity, gradients):
    """sigma . grad(N_j) per element: (E, 3, 4). Diagonal (E, 3) or tensor (E, 3, 3) conductivity."""
    conductivity = np.asarray(conductivity, float)
    if conductivity.ndim == 2:
        return conductivity[:, :, None] * gradients
    return np.einsum("ecd,edj->ecj", conductivity, gradients)


def elementCoefficients(field, conductivity, gradients, volumes):
    """(E, 4) weights: field . (sigma grad N_j) vol. The diagonal branch is the v2-pilot arithmetic."""
    conductivity = np.asarray(conductivity, float)
    if conductivity.ndim == 2:
        return np.einsum("ec,ec,ecj,e->ej", field, conductivity, gradients, volumes)
    return np.einsum("ec,ecd,edj,e->ej", field, conductivity, gradients, volumes)


def pointOperator(geometry, points, sigmaBathSPerM=1.0, source="monodomain", batch=8,
                  nearRadiusMm=0.0, subdivisionLevel=0):
    """Dense (points, nodes) matrix mapping nodal Vm (mV) to potential (mV) at each point.

    Fibres may have any orientation (geometry.fibreFrames); with fibres along x the
    conductivity is diagonal in mesh axes, as in the v2-pilot. nearRadiusMm / subdivisionLevel enable near-field subdivision
    (lead_field.py; Step 4 setting: radius = 3 mesh spacings, level 2). Defaults keep the
    one-point centroid rule verified in VL-2026-10-28-02.
    """
    points = np.atleast_2d(np.asarray(points, float))
    if points.shape[1] != 3 or not np.isfinite(points).all():
        raise ValueError("points must be finite (n, 3) positions in mm")
    if sigmaBathSPerM <= 0:
        raise ValueError("sigma_b must be positive")
    conductivity = sourceConductivity(geometry, source)
    # Element weight vector w_e,j such that sum_j w_e,j Vm_j = (G grad Vm_e) . L * vol, for unit L axis.
    # Combined per point below; computed in batches to bound memory.
    weightedGradients = conductivityTimesGradients(conductivity, geometry.gradients) * geometry.volumes[:, None, None]
    nodeCount = len(geometry.nodePositions)
    flatNodes = geometry.elementNodes.ravel()
    operator = np.empty((len(points), nodeCount))
    prefactor = -1.0 / (4.0 * np.pi * sigmaBathSPerM)
    vertices = geometry.nodePositions[geometry.elementNodes] if subdivisionLevel > 0 else None
    for first in range(0, len(points), batch):
        chunk = points[first:first + batch]
        leadField = lead_field.averageLeadField(chunk, vertices, geometry.centres,
                                                nearRadiusMm, subdivisionLevel)   # (p, e, 3) 1/mm^2
        elementWeights = np.einsum("pec,ecj->pej", leadField, weightedGradients)
        for row, weights in enumerate(elementWeights):
            operator[first + row] = np.bincount(flatNodes, weights=weights.ravel(), minlength=nodeCount)
    return prefactor * operator


def electrodePotentials(geometry, samplePositions, voltage, **options):
    """Mean potential over each electrode's surface samples.

    samplePositions: (electrodes, samples, 3) mm; voltage: (nodes, time) mV -> (electrodes, time) mV.
    """
    samplePositions = np.asarray(samplePositions, float)
    flat = samplePositions.reshape(-1, 3)
    traces = pointOperator(geometry, flat, **options) @ voltage
    return traces.reshape(*samplePositions.shape[:-1], -1).mean(axis=-2)
