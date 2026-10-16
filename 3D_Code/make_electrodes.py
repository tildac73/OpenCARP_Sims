"""Write the recovery samples for the 3D slab's electrode lattice.

The slab surface is z=2 mm. Each cylinder has radius 0.5 mm and length
1 mm, with its axis along x. Eight surface samples are averaged per electrode.
Coordinates are written in micrometres, ordered by gap, x, y, then sample.
The 23 x 23 lattice supports the 242 four-by-four placements exported by run.py.
"""

from pathlib import Path

import numpy as np

ENDO_Z_MM = 2.0
RADIUS_MM = 0.5
HALF_LENGTH_MM = 0.5
GAPS_MM = (0.25, 1.0)
LATTICE_MM = np.arange(-11, 12)


def write_electrodes(path="electrodes/egm_points.pts"):
    """Return and write (8464, 3) recovery sample coordinates in micrometres."""
    points = []
    for gap in GAPS_MM:
        centre_z = ENDO_Z_MM + gap + RADIUS_MM
        for x in LATTICE_MM:
            for y in LATTICE_MM:
                for dx in (-HALF_LENGTH_MM / 2, HALF_LENGTH_MM / 2):
                    for angle in np.deg2rad([0, 90, 180, 270]):
                        points.append(
                            [
                                x + dx,
                                y + RADIUS_MM * np.cos(angle),
                                centre_z + RADIUS_MM * np.sin(angle),
                            ]
                        )
    points_um = np.asarray(points) * 1000
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savetxt(path, points_um, fmt="%.1f", header=str(len(points_um)), comments="")
    return points_um


if __name__ == "__main__":
    points = write_electrodes()
    print(f"{len(points)} recovery points written")
