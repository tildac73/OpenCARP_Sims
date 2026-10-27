#!/usr/bin/env python3
"""Step 4: accuracy of element lead-field quadrature near the tissue surface.

Reference: a uniformly polarised tissue box (analytic_fields.uniformBoxPotential), meshed with
regular tetrahedra (6 per cube, Kuhn decomposition) at the pilot's 0.5 mm spacing. A linear Vm
is represented exactly by linear elements, so the only error is lead-field quadrature.
Reports relative error vs height above the surface for subdivision levels 0-3.

Usage: python3 3D_Code/numerics/quadrature_study.py [--spacing 0.5] [--near 1.5]
Writes _verification/quadrature/quadratureStudy.csv
"""
import argparse
import itertools
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "Shared_Pipeline"))
import analytic_fields  # noqa: E402
import lead_field  # noqa: E402


def boxMesh(boxMin, boxMax, spacing):
    """Regular tetrahedral mesh: 6 tetrahedra per cube (Kuhn/Freudenthal decomposition)."""
    axes = [np.linspace(lo, hi, round((hi - lo) / spacing) + 1) for lo, hi in zip(boxMin, boxMax)]
    grid = np.stack(np.meshgrid(*axes, indexing="ij"), axis=-1)
    shape = grid.shape[:3]
    nodes = grid.reshape(-1, 3)
    elements = []
    for i, j, k in itertools.product(*(range(n - 1) for n in shape)):
        base = np.array([i, j, k])
        for order in itertools.permutations(range(3)):
            corner = base.copy(); ids = [np.ravel_multi_index(tuple(corner), shape)]
            for axis in order:
                corner = corner.copy(); corner[axis] += 1
                ids.append(np.ravel_multi_index(tuple(corner), shape))
            elements.append(ids)
    return nodes, np.array(elements)


def numericalPotential(point, vertices, centres, volumes, sourceDensity, near, level, sigmaBath=1.0):
    field = lead_field.averageLeadField(point, vertices, centres, near, level)[0]   # (E, 3)
    return -float((field @ sourceDensity) @ volumes) / (4 * np.pi * sigmaBath)


def study(spacing=0.5, near=1.5):
    """Two source geometries, each meshed on the same regular grid:
    slab  - whole 10 x 10 x 4 mm box polarised (far-field dominated: easy case);
    local - only a 1 x 1 x 0.5 mm block at the surface under the point (near-field dominated,
            like the region under a passing wavefront: demanding case).
    Error is normalised by |J| t / (2 sigma_b) (t = source thickness), the far-field potential
    scale of a polarised slab, so tangential cases with zero exact potential remain defined.
    """
    cases = {"slab": (np.array([-5.0, -5.0, -2.0]), np.array([5.0, 5.0, 2.0])),
             "local": (np.array([-0.5, -0.5, 1.5]), np.array([0.5, 0.5, 2.0]))}
    rows = []
    for caseName, (boxMin, boxMax) in cases.items():
        nodes, elements = boxMesh(boxMin, boxMax, spacing)
        vertices = nodes[elements]
        centres = vertices.mean(axis=1)
        volumes = np.abs(np.linalg.det(np.concatenate([np.ones((len(elements), 4, 1)), vertices], axis=2))) / 6
        assert np.isclose(volumes.sum(), np.prod(boxMax - boxMin))
        scale = (boxMax[2] - boxMin[2]) / 2.0                 # |J| t / (2 sigma_b) with |J| = sigma_b = 1
        for name, density in [("normal (z)", np.array([0.0, 0.0, 1.0])), ("tangential (x)", np.array([1.0, 0.0, 0.0]))]:
            for lateral in [(0.0, 0.0), (0.25, 0.25), (0.1, 0.37)]:
                for height in [0.05, 0.1, 0.25, 0.5, 1.0, 2.0]:
                    point = np.array([lateral[0], lateral[1], boxMax[2] + height])
                    exact = analytic_fields.uniformBoxPotential(point, boxMin, boxMax, density)
                    for level in range(4):
                        value = numericalPotential(point[None], vertices, centres, volumes, density, near, level)
                        rows.append(dict(case=caseName, source=name, x=lateral[0], y=lateral[1], heightMm=height,
                                         level=level, exact=exact, numerical=value,
                                         scaledError=abs(value - exact) / scale,
                                         relativeError=abs(value - exact) / abs(exact) if abs(exact) > 1e-9 * scale else np.nan))
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--spacing", type=float, default=0.5)
    parser.add_argument("--near", type=float, default=1.5)
    a = parser.parse_args()
    table = study(a.spacing, a.near)
    out = ROOT / "_verification" / "quadrature"; out.mkdir(parents=True, exist_ok=True)
    table.to_csv(out / f"quadratureStudy_h{a.spacing}_near{a.near}.csv", index=False)
    pd.set_option("display.width", 200)
    for column, label in [("scaledError", "error / far-field scale"), ("relativeError", "relative error (where exact != 0)")]:
        summary = table.groupby(["case", "source", "heightMm", "level"])[column].max().unstack("level")
        print(f"Max {label} over 3 lateral positions (spacing {a.spacing} mm, near radius {a.near} mm):")
        print(summary.map(lambda e: "-" if np.isnan(e) else f"{100 * e:.3f}%").to_string(), "\n")


if __name__ == "__main__":
    main()
