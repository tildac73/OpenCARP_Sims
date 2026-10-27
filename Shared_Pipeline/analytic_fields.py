"""Closed-form reference potentials for verifying lead-field quadrature (Step 4).

Uniformly polarised box: tissue occupying an axis-aligned box with uniform source density
J = G grad Vm (Vm linear, G constant). In the monodomain recovery model

    phi(x) = -1/(4 pi sigma_b) * integral_box J . grad_y(1/|x-y|) dV
           = -1/(4 pi sigma_b) * sum_a J_a [ integral_{face a=max} 1/r dA - integral_{face a=min} 1/r dA ]

by the divergence theorem applied to each component. Each face integral of 1/r over a
rectangle has the closed form (u, v in-plane offsets of the corners, w normal offset)

    F(u, v, w) = u ln(v + R) + v ln(u + R) - w atan(u v / (w R)),  R = sqrt(u^2 + v^2 + w^2)

combined over the four corners with alternating signs. Points must not lie on a face edge.
"""
import numpy as np


def _primitive(u, v, w):
    r = np.sqrt(u * u + v * v + w * w)
    term = np.zeros_like(r)
    nonzero = np.abs(w) > 0
    term[nonzero] = w[nonzero] * np.arctan(u[nonzero] * v[nonzero] / (w[nonzero] * r[nonzero]))
    return u * np.log(v + r) + v * np.log(u + r) - term


def rectangleInverseDistance(point, axis, planeCoord, bounds):
    """Integral of 1/|x - y| over the rectangle {y_axis = planeCoord, other coords in bounds}.

    bounds: ((lo, hi), (lo, hi)) for the two remaining axes in increasing axis order.
    """
    others = [a for a in range(3) if a != axis]
    w = np.atleast_1d(np.asarray(planeCoord - point[axis], float))
    total = 0.0
    for (corner_u, sign_u) in ((bounds[0][1], 1), (bounds[0][0], -1)):
        for (corner_v, sign_v) in ((bounds[1][1], 1), (bounds[1][0], -1)):
            u = np.atleast_1d(np.asarray(corner_u - point[others[0]], float))
            v = np.atleast_1d(np.asarray(corner_v - point[others[1]], float))
            total = total + sign_u * sign_v * _primitive(u, v, w)
    return float(np.squeeze(total))


def uniformBoxPotential(point, boxMin, boxMax, sourceDensity, sigmaBath=1.0):
    """phi (mV) at `point` for uniform J (S/m * mV/mm) in the box; units as in electrode_synthesis."""
    point = np.asarray(point, float)
    total = 0.0
    for axis in range(3):
        if sourceDensity[axis] == 0:
            continue
        others = [a for a in range(3) if a != axis]
        bounds = ((boxMin[others[0]], boxMax[others[0]]), (boxMin[others[1]], boxMax[others[1]]))
        upper = rectangleInverseDistance(point, axis, boxMax[axis], bounds)
        lower = rectangleInverseDistance(point, axis, boxMin[axis], bounds)
        total += sourceDensity[axis] * (upper - lower)
    return -total / (4 * np.pi * sigmaBath)
