#!/usr/bin/env python3
"""Write the compact reference used by tests/test_v2_reference.py.

Provenance: built from the existing analysed outputs, never from a fresh run.
  * original 0-400 ms window  : boundaryPilotV2 2/results (analysed 20 Oct 2026, scipy)
  * declared 25-100 ms window : _refinement_work/run_w25_100 (27 Oct 2026, numpy shim)
The two windows are stored separately so either can be reproduced independently.

Run from the repository root:
    python3 tests/reference/make_v2_reference.py
"""
import hashlib
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent / "v2_pilot"
SOURCES = {
    "window0_400": ROOT / "boundaryPilotV2 2" / "results",
    "window25_100": ROOT / "_refinement_work" / "run_w25_100" / "results",
}
CHANGE_POINT_KEYS = ["condition", "gapMm", "noiseRepeat", "kind", "label",
                     "distanceConvention", "preprocessing", "model"]


def weightHash(row):
    # Selection weights decide which bipole/omnipolar direction was recorded.
    # Rounded to 1e-9 so float formatting differences cannot change the hash.
    weights = np.round(row[[f"weightE{i}" for i in range(16)]].to_numpy(float), 9) + 0.0
    return hashlib.sha256(weights.tobytes()).hexdigest()[:16]


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    for name, folder in SOURCES.items():
        points = pd.concat([pd.read_csv(folder / f"{c}Points.csv") for c in ["intact", "lesion"]],
                           ignore_index=True)
        points["weightHash"] = points.apply(weightHash, axis=1)
        points[["recordingId", "ppMv", "weightHash"]].to_csv(OUT / f"points_{name}.csv", index=False)

        contributions = pd.read_csv(folder / "contributions.csv")
        contributions[["recordingId", "rawFov80ReconstructionMm", "signedCorrelation",
                       "shapeError", "fovEligible"]].to_csv(OUT / f"contributions_{name}.csv", index=False)

        fits = pd.read_csv(folder / "clinical" / "changePointFits.csv")
        fits = fits[fits.selected == True]  # noqa: E712 (CSV booleans)
        fits[CHANGE_POINT_KEYS + ["changePointMm", "fitStatus", "plateauMv"]].sort_values(
            CHANGE_POINT_KEYS).to_csv(OUT / f"changepoints_{name}.csv", index=False)
        print(name, len(points), "points;", len(contributions), "contributions;", len(fits), "fits")


if __name__ == "__main__":
    main()
