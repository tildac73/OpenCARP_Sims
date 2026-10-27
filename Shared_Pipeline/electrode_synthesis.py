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
    if source == "monodomain":
        if geometry.extracellularConductivity is None:
            raise ValueError("Geometry lacks extracellular conductivity; read it with ground_truth.readElementGeometry")
        return monodomainConductivity(geometry.conductivity, geometry.extracellularConductivity)
    if source == "intracellular":
        return np.asarray(geometry.conductivity, float)
    raise ValueError("source must be 'monodomain' or 'intracellular'")


def pointOperator(geometry, points, sigmaBathSPerM=1.0, source="monodomain", batch=8):
    """Dense (points, nodes) matrix mapping nodal Vm (mV) to potential (mV) at each point.

    Fibres must be x-aligned (checked by readElementGeometry), so the diagonal conductivity
    is expressed in mesh axes.
    """
    points = np.atleast_2d(np.asarray(points, float))
    if points.shape[1] != 3 or not np.isfinite(points).all():
        raise ValueError("points must be finite (n, 3) positions in mm")
    if sigmaBathSPerM <= 0:
        raise ValueError("sigma_b must be positive")
    conductivity = sourceConductivity(geometry, source)
    # Element weight vector w_e,j such that sum_j w_e,j Vm_j = (G grad Vm_e) . L * vol, for unit L axis.
    # Combined per point below; computed in batches to bound memory.
    weightedGradients = conductivity[:, :, None] * geometry.gradients * geometry.volumes[:, None, None]
    nodeCount = len(geometry.nodePositions)
    flatNodes = geometry.elementNodes.ravel()
    operator = np.empty((len(points), nodeCount))
    prefactor = -1.0 / (4.0 * np.pi * sigmaBathSPerM)
    for first in range(0, len(points), batch):
        chunk = points[first:first + batch]
        displacement = chunk[:, None, :] - geometry.centres[None, :, :]
        distance = np.linalg.norm(displacement, axis=2)
        if np.any(distance < 1e-9):
            raise ValueError("A field point coincides with an element centroid")
        leadField = displacement / distance[..., None] ** 3            # (p, e, 3) 1/mm^2
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
