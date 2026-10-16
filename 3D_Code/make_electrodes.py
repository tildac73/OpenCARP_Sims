"""Generate electrodes/egm_points.pts for openCARP phie recovery.

One 1 mm lattice covers EVERY grid in the paper's Table 1 (1/3/6/9 mm
spacing) and every +/-1 mm offset, so all 656 configurations are subsets
of a single recovery run.

Electrodes: cylinders r = 0.5 mm, length 1 mm (paper), axis parallel to
the wall along x (like electrodes on a grid-catheter spline). Each is
sampled at 8 surface points (2 axial x 4 around), averaged afterwards.
Gap to tissue 0.25 or 1.0 mm (paper) measured to the cylinder surface.
Endocardium at z = +2 mm. Output in um.

Point order: gap, x, y, sample  ->  reshape (T, 2, 23, 23, 8).mean(-1)
"""
import os
import numpy as np

ENDO_Z = 2.0                     # mm
R, HALF_LEN = 0.5, 0.5           # mm
GAPS = (0.25, 1.0)               # mm
LATTICE = np.arange(-11, 12)     # mm, 23 positions centred on bundle

pts = []
for gap in GAPS:
    zc = ENDO_Z + gap + R
    for x in LATTICE:
        for y in LATTICE:
            for dx in (-HALF_LEN / 2, HALF_LEN / 2):
                for th in np.deg2rad([0, 90, 180, 270]):
                    pts.append([x + dx, y + R * np.cos(th), zc + R * np.sin(th)])

pts = np.asarray(pts) * 1000.0   # mm -> um
os.makedirs('electrodes', exist_ok=True)
np.savetxt('electrodes/egm_points.pts', pts, fmt='%.1f',
           header=str(len(pts)), comments='')
print(f'{len(pts)} points written')  # 2 * 23 * 23 * 8 = 8464


def grid_indices(spacing, n, ox=0, oy=0):
    """Lattice indices (ix, iy) of an n x n grid with given spacing (mm),
    centred on the bundle, shifted by (ox, oy) mm. Drop any outside."""
    start = -spacing * (n - 1) / 2
    coords = start + spacing * np.arange(n)
    ix = [int(round(c + ox)) + 11 for c in coords]
    iy = [int(round(c + oy)) + 11 for c in coords]
    ix = [i for i in ix if 0 <= i < len(LATTICE)]
    iy = [i for i in iy if 0 <= i < len(LATTICE)]
    return ix, iy

# Paper's base grids: (1 mm, 21), (3 mm, 7), (6 mm, 4), (9 mm, 3);
# offsets ox, oy in {-1, 0, 1} mm.
