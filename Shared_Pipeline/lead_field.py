"""Element-averaged point lead field with near-field subdivision (readiness plan, Step 4).

The potential of element e at field point x needs the element integral of
    L(x, y) = (x - y) / |x - y|^3 = grad_y (1 / |x - y|)
because sigma grad Vm is constant on a linear tetrahedron. One-point (centroid) quadrature is
accurate far away but not when x is within a few element sizes of the element: at the pilot's
0.25 mm standoff the nearest centroids are ~0.4 mm from the electrode surface.

Near elements (centroid within nearRadiusMm of x) are integrated by recursive midpoint
subdivision: each level splits every tetrahedron into 8 children of equal volume (4 corner
tetrahedra + 4 from the inner octahedron, split along one diagonal) and evaluates L at the child
centroids. Level 0 reproduces the original centroid rule exactly.

Accuracy is checked against a closed-form solution in tests/test_lead_field.py and
3D_Code/numerics/quadrature_study.py.
"""
from functools import lru_cache

import numpy as np

_CORNERS = np.eye(4)


def _midpoint(a, b):
    return 0.5 * (a + b)


def _children(tet):
    """Eight equal-volume children of a tetrahedron given as 4 barycentric vertex rows."""
    v0, v1, v2, v3 = tet
    m01, m02, m03 = _midpoint(v0, v1), _midpoint(v0, v2), _midpoint(v0, v3)
    m12, m13, m23 = _midpoint(v1, v2), _midpoint(v1, v3), _midpoint(v2, v3)
    return [np.array(t) for t in (
        (v0, m01, m02, m03), (v1, m01, m12, m13), (v2, m02, m12, m23), (v3, m03, m13, m23),
        (m01, m02, m03, m13), (m01, m02, m12, m13), (m02, m03, m13, m23), (m02, m12, m13, m23))]


@lru_cache(maxsize=None)
def subdivisionBarycentres(level):
    """Barycentric coordinates (8**level, 4) of child centroids after `level` subdivisions."""
    if level < 0:
        raise ValueError("Subdivision level must be non-negative")
    tets = [_CORNERS]
    for _ in range(level):
        tets = [child for tet in tets for child in _children(tet)]
    return np.array([tet.mean(axis=0) for tet in tets])


def averageLeadField(points, vertices, centres, nearRadiusMm=0.0, level=0):
    """Element-averaged L for each point: (points, elements, 3), units 1/mm^2.

    points (P, 3) mm; vertices (E, 4, 3) mm; centres (E, 3) mm.
    """
    points = np.atleast_2d(np.asarray(points, float))
    displacement = points[:, None, :] - centres[None, :, :]
    distance = np.linalg.norm(displacement, axis=2)
    if np.any(distance < 1e-12):
        raise ValueError("A field point coincides with an element centroid")
    field = displacement / distance[..., None] ** 3
    if level == 0 or nearRadiusMm <= 0:
        return field
    weights = subdivisionBarycentres(level)                         # (Q, 4)
    for p, point in enumerate(points):
        near = np.flatnonzero(distance[p] < nearRadiusMm)
        if near.size == 0:
            continue
        quadrature = np.einsum("qk,ekc->eqc", weights, vertices[near])  # (n, Q, 3)
        offsets = point - quadrature
        r = np.linalg.norm(offsets, axis=2)
        if np.any(r < 1e-12):
            raise ValueError("A field point coincides with a quadrature point")
        field[p, near] = (offsets / r[..., None] ** 3).mean(axis=1)
    return field
