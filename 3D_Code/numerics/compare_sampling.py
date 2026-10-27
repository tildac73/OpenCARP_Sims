#!/usr/bin/env python3
"""Compare two analysed runs that differ only in electrode surface sampling (Step 3).

Tolerances (DL-2026-10-28-05): PP <= 5 %, FOV80 <= 0.5 mm, change point <= 0.5 mm.
Usage: python3 3D_Code/numerics/compare_sampling.py <coarse run> <fine run> [--label NAME]
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

TOL = {"ppRelative": 0.05, "fov80Mm": 0.5, "changePointMm": 0.5}
KEYS = ["condition", "gapMm", "noiseRepeat", "kind", "label", "distanceConvention", "preprocessing", "model"]


def load(run):
    results = Path(run) / "results"
    points = pd.concat([pd.read_csv(results / f"{c}Points.csv") for c in ["intact", "lesion"]])
    contributions = pd.read_csv(results / "contributions.csv")
    fits = pd.read_csv(results / "clinical" / "changePointFits.csv")
    return points, contributions, fits[fits.selected == True]  # noqa: E712


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("coarse"); parser.add_argument("fine"); parser.add_argument("--label", default="sampling")
    a = parser.parse_args()
    (pc, cc, fc), (pf, cf, ff) = load(a.coarse), load(a.fine)
    p = pc.merge(pf, on="recordingId", suffixes=("C", "F"), validate="one_to_one")
    clean = p[p.noiseRepeatC == 0]
    rel = (clean.ppMvF - clean.ppMvC).abs() / clean.ppMvF
    near = clean.distanceInsideMmC.abs() <= 10           # where the signal is well above any floor
    c = cc.merge(cf, on="recordingId", suffixes=("C", "F"), validate="one_to_one")
    dFov = (c.fov80ReconstructionMmF - c.fov80ReconstructionMmC).abs()
    f = fc.merge(ff, on=KEYS, suffixes=("C", "F"), validate="one_to_one")
    primary = f[(f.preprocessing == "clinicalTrimmed") & (f.distanceConvention == "recordingAnchor")]
    dCp = (primary.changePointMmF - primary.changePointMmC).abs()
    supported = primary[(primary.fitStatusC == "supportedDescriptiveChangePoint") | (primary.fitStatusF == "supportedDescriptiveChangePoint")]
    statusFlips = int((primary.fitStatusC != primary.fitStatusF).sum())
    summary = dict(
        label=a.label, coarse=a.coarse, fine=a.fine, tolerances=TOL,
        ppRelativeMaxAll=float(rel.max()), ppRelativeMaxWithin10mm=float(rel[near].max()),
        ppRelativeMedian=float(rel.median()),
        ppWorst=clean.loc[rel.idxmax(), ["recordingIdC" if "recordingIdC" in clean else "recordingId", "kindC", "gapMmC", "distanceInsideMmC", "ppMvC", "ppMvF"]].to_dict(),
        fov80MaxAbsMm=float(dFov.max()), fov80Changed=int((dFov > 0).sum()), fov80Count=int(len(c)),
        changePointMaxAbsMmAllPrimary=float(dCp.max()), changePointMaxAbsMmSupported=float((supported.changePointMmF - supported.changePointMmC).abs().max()) if len(supported) else None,
        primaryFits=int(len(primary)), fitStatusChanges=statusFlips,
    )
    summary["passed"] = bool(summary["ppRelativeMaxAll"] <= TOL["ppRelative"] and summary["fov80MaxAbsMm"] <= TOL["fov80Mm"]
                             and summary["changePointMaxAbsMmAllPrimary"] <= TOL["changePointMm"] and statusFlips == 0)
    out = Path(a.fine).parent / f"comparison_{a.label}.json"
    out.write_text(json.dumps(summary, indent=2, default=float) + "\n")
    print(json.dumps(summary, indent=2, default=float))


if __name__ == "__main__":
    main()
