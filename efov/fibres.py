"""Fibre fields of the slab (FM-06), ported from Shared_Pipeline/fibres.py (pilot_v2; DL-2026-10-28-12).

Convention. Fibres lie in planes parallel to the tissue surface. Their angle is measured in the x-y
plane from +x, counter-clockwise seen from the catheter (+z). Depth d is measured from the
catheter-facing (endocardial) surface z = +wall/2 to the far surface (d = wall).

Models (model.json sweep variable 'fibres'):
  {"model": "uniform", "angleDeg": a}
  {"model": "linearRotation", "endocardialAngleDeg": a, "epicardialAngleDeg": b}
      theta(d) = a + (b - a) d / wall
  {"model": "bilayer", "endocardialAngleDeg": a, "epicardialAngleDeg": b,
   "transitionDepthFraction": f, "transitionWidthMm": w}
      theta = a above depth f*wall - w/2, b below f*wall + w/2, linear in between (w = 0: abrupt)

The bilayer form follows the qualitative description of Pashakhanloo et al. 2016
(doi:10.1161/CIRCEP.116.004133). No angle is taken from that paper: every angle is an assumption.
Tissue is transversely isotropic, so one fibre vector per element defines the tensors.
"""
import numpy as np

MODELS = ("uniform", "linearRotation", "bilayer")
REQUIRED = {"uniform": ["angleDeg"], "linearRotation": ["endocardialAngleDeg", "epicardialAngleDeg"],
            "bilayer": ["endocardialAngleDeg", "epicardialAngleDeg", "transitionDepthFraction", "transitionWidthMm"]}


def validateSpec(spec, wallMm):
    if not isinstance(spec, dict) or spec.get("model") not in MODELS:
        raise ValueError(f"fibres.model must be one of {MODELS}")
    for key in REQUIRED[spec["model"]]:
        if key not in spec or not np.isfinite(float(spec[key])):
            raise ValueError(f"fibres.{key} is required for model {spec['model']}")
    if spec["model"] == "bilayer":
        depth, width = float(spec["transitionDepthFraction"]) * wallMm, float(spec["transitionWidthMm"])
        if width < 0 or not 0 < depth - width / 2 or not depth + width / 2 < wallMm:
            raise ValueError("Bilayer transition must lie strictly inside the wall")


def fibreAnglesDeg(depthMm, wallMm, spec):
    """Fibre angle (deg from +x) at each depth below the catheter-facing surface."""
    depth = np.asarray(depthMm, float)
    if spec["model"] == "uniform":
        return np.full(depth.shape, float(spec["angleDeg"]))
    a, b = float(spec["endocardialAngleDeg"]), float(spec["epicardialAngleDeg"])
    if spec["model"] == "linearRotation":
        return a + (b - a) * np.clip(depth / wallMm, 0.0, 1.0)
    centre, width = float(spec["transitionDepthFraction"]) * wallMm, float(spec["transitionWidthMm"])
    if width == 0:
        return np.where(depth < centre, a, b)
    return a + (b - a) * np.clip((depth - (centre - width / 2)) / width, 0.0, 1.0)


def fibreVectors(elementCentresMm, wallMm, spec):
    """Unit fibre vector per element from its centroid depth (catheter-facing surface at z = +wall/2)."""
    validateSpec(spec, wallMm)
    depth = wallMm / 2 - np.asarray(elementCentresMm)[:, 2]
    angle = np.radians(fibreAnglesDeg(depth, wallMm, spec))
    return np.column_stack([np.cos(angle), np.sin(angle), np.zeros_like(angle)])


def endocardialAngleDeg(spec):
    """Fibre angle at the catheter-facing surface."""
    return float(spec["angleDeg"] if spec["model"] == "uniform" else spec["endocardialAngleDeg"])
