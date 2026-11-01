"""Diagnose the bath-check mismatch (31 Oct): openCARP phi_e vs the independent FEM, from bathSignals.npz.

The check found correlation >= 0.977 but relative RMS 0.75-0.81. Shape agreement with a large
amplitude error suggests a scale factor between the solvers. This prints, per electrode sample, the
least-squares factor k in  openCARP ~= k * FEM, how k varies (overall, by electrode gap, by height
above the tissue), and the residual left after removing k. A uniform k points to a conductivity or
units mismatch; k varying with height or gap points to a boundary or geometry difference.

Run from the repository root (seconds, no simulation, prints a few lines):
    python3 3D_Code/readiness/bath_diagnose.py 3D_Code/runs/readiness/bath/bath
"""
import json
import sys
from pathlib import Path

import numpy as np


def main(folder):
    data = np.load(Path(folder) / "bathSignals.npz")
    carp, fem = data["openCarpMv"].astype(float), data["femMv"].astype(float)
    samples, placements = data["samples"], data["placements"]          # (placements, 16, S, 3) mm; (placements, 3)
    carp = carp - carp.mean(axis=1, keepdims=True)
    fem = fem - fem.mean(axis=1, keepdims=True)
    k = np.sum(carp * fem, axis=1) / np.maximum(np.sum(fem * fem, axis=1), 1e-30)
    residual = carp - k[:, None] * fem
    rms = lambda x: np.sqrt(np.mean(x ** 2, axis=1))
    residualRelative = rms(residual) / np.maximum(rms(carp), 1e-30)
    perPlacement = samples.reshape(len(placements), -1, 3)
    kByPlacement = k.reshape(len(placements), -1)
    heights = samples.reshape(-1, 3)[:, 2]
    order = np.argsort(heights)
    thirds = np.array_split(order, 3)
    out = dict(
        sampleCount=int(len(k)),
        scaleOpenCarpOverFem=dict(median=float(np.median(k)), min=float(k.min()), max=float(k.max()),
                                  p05=float(np.percentile(k, 5)), p95=float(np.percentile(k, 95))),
        scaleByGap={f"{p[2]:g} mm": dict(median=float(np.median(kb)), min=float(kb.min()), max=float(kb.max()),
                                         heightRangeMm=[float(ps[:, 2].min()), float(ps[:, 2].max())])
                    for p, kb, ps in zip(placements, kByPlacement, perPlacement)},
        scaleByHeightThird=[dict(heightRangeMm=[float(heights[t].min()), float(heights[t].max())],
                                 median=float(np.median(k[t]))) for t in thirds],
        residualAfterScaleRelativeRms=dict(median=float(np.median(residualRelative)), max=float(residualRelative.max())),
        peakToPeakMv=dict(openCarpMedian=float(np.median(np.ptp(carp, axis=1))), femMedian=float(np.median(np.ptp(fem, axis=1)))),
    )
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit(__doc__)
    main(sys.argv[1])
