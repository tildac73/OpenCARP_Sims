"""Fibre fields and conductivity tensors for the slab (readiness plan Step 7).

Convention. Fibres lie in planes parallel to the tissue surface (no transmural imbrication).
Their angle theta is measured in the x-y plane from +x, counter-clockwise seen from the catheter
(+z). Depth d is measured from the catheter-facing surface (z = +thickness/2, the endocardium in
this model) towards the far surface (d = thickness, the epicardium).

Models (config key "fibres"; absent = every fibre along +x, the v2-pilot behaviour):
  {"model": "uniform", "angleDeg": a}
  {"model": "linearRotation", "endocardialAngleDeg": a, "epicardialAngleDeg": b}
      theta(d) = a + (b - a) d / thickness
  {"model": "bilayer", "endocardialAngleDeg": a, "epicardialAngleDeg": b,
   "transitionDepthMm": d0, "transitionWidthMm": w}
      theta = a for d < d0 - w/2, b for d > d0 + w/2, linear in between (w = 0: abrupt)

The bilayer form follows the qualitative description of Pashakhanloo et al. 2016
(doi:10.1161/CIRCEP.116.004133): nearly perpendicular layers with an abrupt mid-wall transition in
the posterior and anterior left atrium. No angles are taken from that paper (none are reported
numerically); every angle in a config is an assumption to be labelled as such.

Tensors. With orthonormal local axes a_k = (fibre, sheet, normal) and principal conductivities
s_k, the tensor in mesh axes is sum_k s_k a_k a_k^T. For transversely isotropic tissue (sheet and
normal values equal) the choice of sheet axis does not matter. Intracellular and extracellular
tensors share the same axes, so the monodomain tensor is the per-axis series combination rotated
the same way.
"""
import numpy as np

MODELS = ("uniform", "linearRotation", "bilayer")


def validateSpec(spec, thicknessMm):
    if not isinstance(spec, dict) or spec.get("model") not in MODELS:
        raise ValueError(f"fibres.model must be one of {MODELS}")
    required = {"uniform": ["angleDeg"], "linearRotation": ["endocardialAngleDeg", "epicardialAngleDeg"],
                "bilayer": ["endocardialAngleDeg", "epicardialAngleDeg", "transitionDepthMm", "transitionWidthMm"]}
    for key in required[spec["model"]]:
        if key not in spec or not np.isfinite(float(spec[key])):
            raise ValueError(f"fibres.{key} is required for model {spec['model']}")
    if spec["model"] == "bilayer":
        depth, width = float(spec["transitionDepthMm"]), float(spec["transitionWidthMm"])
        if width < 0 or not 0 < depth - width / 2 or not depth + width / 2 < thicknessMm:
            raise ValueError("Bilayer transition must lie strictly inside the wall")


def fibreAnglesDeg(depthMm, thicknessMm, spec):
    depth = np.asarray(depthMm, float)
    model = spec["model"]
    if model == "uniform":
        return np.full(depth.shape, float(spec["angleDeg"]))
    a, b = float(spec["endocardialAngleDeg"]), float(spec["epicardialAngleDeg"])
    if model == "linearRotation":
        return a + (b - a) * np.clip(depth / thicknessMm, 0.0, 1.0)
    centre, width = float(spec["transitionDepthMm"]), float(spec["transitionWidthMm"])
    if width == 0:
        return np.where(depth < centre, a, b)
    fraction = np.clip((depth - (centre - width / 2)) / width, 0.0, 1.0)
    return a + (b - a) * fraction


def fibreVectors(elementCentresMm, slabSizeMm, spec):
    """Unit fibre vector per element from its centroid depth below the catheter-facing surface."""
    thickness = float(slabSizeMm[2])
    validateSpec(spec, thickness)
    depth = thickness / 2 - np.asarray(elementCentresMm)[:, 2]
    angle = np.radians(fibreAnglesDeg(depth, thickness, spec))
    return np.column_stack([np.cos(angle), np.sin(angle), np.zeros_like(angle)])


def isXAligned(fibres):
    fibres = np.asarray(fibres, float)
    return bool(np.allclose(np.abs(fibres[:, :3]), [1.0, 0.0, 0.0]))


def localFrames(fibres, sheets=None):
    """(E, 3, 3) rows = unit fibre, sheet, normal. Without sheets, a right-handed completion is chosen."""
    fibres = np.asarray(fibres, float)
    length = np.linalg.norm(fibres, axis=1)
    if np.any(length < 1e-12):
        raise ValueError("Zero fibre vector (bath element?) in tissue")
    f = fibres / length[:, None]
    if sheets is None:
        helper = np.where(np.abs(f[:, 2:3]) < 0.9, [[0.0, 0.0, 1.0]], [[1.0, 0.0, 0.0]])
        s = np.cross(helper, f)
    else:
        s = np.asarray(sheets, float) - np.sum(np.asarray(sheets, float) * f, axis=1, keepdims=True) * f
    s /= np.linalg.norm(s, axis=1, keepdims=True)
    n = np.cross(f, s)
    return np.stack([f, s, n], axis=1)


def tensors(principal, frames):
    """sum_k principal[:, k] a_k a_k^T in mesh axes: (E, 3, 3)."""
    return np.einsum("eki,ek,ekj->eij", frames, np.asarray(principal, float), frames)
