#!/usr/bin/env python3
"""Step 4: extended ground-truth descriptors for the pilot's intact recordings, and the
half-plane prediction of each lesion voltage profile from intact contributions alone.

Recordings: intact, catheter centre y = 0, both standoffs, the four declared configurations,
25-100 ms window (as run_w25_100). Lead field: level-2 near-field subdivision (DL-2026-10-28-06).

Check built in: the cylinder family must reproduce the stored FOV80 exactly.
Half-plane prediction: PP(d) of the intact signal from tissue beyond a plane at distance d
(parallel to the boundary, on the viable side) vs the measured lesion PP at distance d. Uses the
intact recording's selection weights; lesion recordings may select a different bipole or
omnipolar direction, which the comparison reports rather than hides.

Usage: python3 3D_Code/numerics/ground_truth_metrics.py
Writes _verification/ground_truth_metrics/{metrics,halfPlaneProfiles,curves}.csv
"""
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "3D_Code"), str(ROOT / "Shared_Pipeline")]
import boundary_analysis as analysis  # noqa: E402
import contribution_metrics as metrics  # noqa: E402
import electrode_synthesis as synthesis  # noqa: E402
import ground_truth  # noqa: E402
from boundary_geometry import signedDistance  # noqa: E402

RUN = ROOT / "boundaryPilotV2 2"
ANALYSED = ROOT / "_refinement_work" / "run_w25_100"
OUT = ROOT / "_verification" / "ground_truth_metrics"
QUADRATURE = dict(nearRadiusMm=1.5, subdivisionLevel=2)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    config = json.loads((ANALYSED / "config.json").read_text())
    gain = json.loads((RUN / "results" / "calibration.json").read_text())["gain"]
    points = pd.read_csv(ANALYSED / "results" / "intactPoints.csv")
    lesion = pd.read_csv(ANALYSED / "results" / "lesionPoints.csv")
    stored = pd.read_csv(ANALYSED / "results" / "contributions.csv").set_index("recordingId")
    with np.load(ANALYSED / "results" / "intactTraces.npz") as archive:
        recoveredMv = archive["cleanAtSelectedWeightsMv"]
    source = dict(np.load(RUN / "intact" / "egm.npz"))
    rawSlice, measureSlice = analysis.measurementSlices(source["t_ms"], config)
    sampleRate = 1000 / np.diff(source["t_ms"][:2])[0]
    geometry = ground_truth.readElementGeometry(RUN, "intact")
    conductivity = synthesis.sourceConductivity(geometry, "monodomain")
    vertices = geometry.nodePositions[geometry.elementNodes]
    surfaceZ = geometry.nodePositions[:, 2].max()
    selected = points[(points.noiseRepeat == 0) & np.isclose(points.centreYmm, 0.0)]
    rows, profiles, curves = [], [], []
    for mode in ["uni", "differential"]:
        band = None if mode == "uni" else config["analysis"]["differentialBandHz"]
        voltage = ground_truth.readFilteredVoltage(RUN / "intact", rawSlice, band, sampleRate,
                                                   config["analysis"]["filterOrder"])[:, measureSlice].astype(float)
        records = selected[selected.kind == "uni"] if mode == "uni" else selected[selected.kind != "uni"]
        for record in records.to_dict("records"):
            weights = np.array([record[f"weightE{i}"] for i in range(16)])
            anchor = np.array([record["anchorXmm"], record["anchorYmm"], record["anchorZmm"]])
            samples = source["uni_sample_xyz"][record["placement"]]
            field = ground_truth.finiteRecordingField(samples, weights, geometry.centres, vertices, QUADRATURE)
            coefficients = metrics.elementCoefficients(geometry, field, conductivity)
            keys = dict(
                cylinder=np.linalg.norm(geometry.centres[:, :2] - anchor[:2], axis=1),
                sphere=np.linalg.norm(geometry.centres - anchor, axis=1),
                depth=surfaceZ - geometry.centres[:, 2],
            )
            row = dict(recordingId=record["recordingId"], kind=record["kind"], label=record["label"], gapMm=record["gapMm"])
            for family, key in keys.items():
                step = 0.25 if family == "depth" else 0.5
                found, upper, fraction = metrics.extents(geometry, coefficients, voltage, key, step)
                for level, value in found.items():
                    row[f"{family}{int(level * 100)}Mm"] = value
                curves += [dict(recordingId=record["recordingId"], family=family, upperMm=u, fraction=f)
                           for u, f in zip(upper, fraction)]
            for level in metrics.LEVELS:
                row[f"sphereVolume{int(level * 100)}Mm3"] = metrics.sphereVolume(geometry, anchor, row[f"sphere{int(level * 100)}Mm"])
            row["storedFov80Mm"] = stored.loc[record["recordingId"], "fov80ReconstructionMm"]
            footprint, _ = metrics.energyFootprint(geometry, coefficients, voltage, anchor)
            row.update(footprint)
            # Half-plane: key = distance behind a boundary-parallel plane through the anchor (viable side).
            behind = signedDistance(anchor[None], config["boundary"])[0] - signedDistance(geometry.centres, config["boundary"])
            edges = np.arange(np.floor(behind.min()), behind.max() + 0.5, 0.5)
            binTraces = metrics.binnedTraces(geometry, coefficients, voltage, behind, edges)
            fullTrace = binTraces.sum(axis=0)
            recovered = recoveredMv[record["traceIndex"]].astype(float)
            scale = float(fullTrace @ recovered / (fullTrace @ fullTrace))          # mV per model unit (includes gain)
            row["reconstructionCorrelation"] = float(np.corrcoef(fullTrace, recovered)[0, 1])
            beyond = np.cumsum(binTraces[::-1], axis=0)[::-1]                        # sources with key >= edges[i]
            lesionRows = lesion[(lesion.noiseRepeat == 0) & (lesion.label == record["label"]) &
                                np.isclose(lesion.gapMm, record["gapMm"])]
            measured = lesionRows.groupby("distanceInsideMm").ppMv.median()
            for d in np.arange(0.0, 15.01, 0.5):
                index = int(np.searchsorted(edges, d))
                predicted = float(np.ptp(scale * beyond[index])) if index < len(beyond) else 0.0
                nearest = measured.index[np.argmin(np.abs(measured.index - d))]
                profiles.append(dict(recordingId=record["recordingId"], kind=record["kind"], label=record["label"],
                                     gapMm=record["gapMm"], distanceMm=d, predictedPpMv=predicted,
                                     measuredPpMv=float(measured.loc[nearest]) if abs(nearest - d) <= 0.26 else np.nan,
                                     intactPpMv=float(np.ptp(recovered))))
            rows.append(row)
            print(f"{record['recordingId']}: cylinder80 {row['cylinder80Mm']} (stored {row['storedFov80Mm']}), "
                  f"sphere80 {row['sphere80Mm']} mm / {row['sphereVolume80Mm3']:.1f} mm3, depth80 {row['depth80Mm']} mm, "
                  f"ellipse {footprint['sigmaMajorMm']:.2f} x {footprint['sigmaMinorMm']:.2f} mm @ {footprint['majorAxisAngleToXDeg']:.0f} deg",
                  flush=True)
    pd.DataFrame(rows).to_csv(OUT / "metrics.csv", index=False)
    pd.DataFrame(profiles).to_csv(OUT / "halfPlaneProfiles.csv", index=False)
    pd.DataFrame(curves).to_csv(OUT / "curves.csv", index=False)


if __name__ == "__main__":
    main()
