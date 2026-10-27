#!/usr/bin/env python3
"""Step 4: matched tissue-removal prediction of every clean lesion recording.

Question: how much of the lesion voltage profile is explained purely by removing tissue
(the lesion half-space contributes nothing), with the intact activation unchanged?

For each lesion recording (noiseRepeat 0): its own placement, sample positions and selected
weights; intact Vm; lead field with level-2 near-field quadrature; source summed only over
elements that stay viable (signed distance < 0; the lesion half-space has zero conductivity
in the lesion mesh). Scale: the fitted intact model-to-mV scale for that gap and mode
(fixed gain and sign; not refitted per lesion recording).

Residual (measured / predicted) is therefore the effect of the lesion on activation and on
the source field near the boundary (and the openCARP-vs-synthesis difference, about 1-2%,
VL-2026-10-28-02/-08), not of tissue removal itself.

Stages, each fits the 180 s VM limit:
  cache --start i --stop j    per-placement per-electrode viable-tissue traces (uni + differential)
  assemble                    weights, scale, PP; writes removalPrediction.csv
"""
import argparse
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
CACHE = OUT / "removal_cache"
QUADRATURE = dict(nearRadiusMm=1.5, subdivisionLevel=2)
WEIGHTS = [f"weightE{i}" for i in range(16)]


def setup():
    config = json.loads((ANALYSED / "config.json").read_text())
    source = dict(np.load(RUN / "intact" / "egm.npz"))
    rawSlice, measureSlice = analysis.measurementSlices(source["t_ms"], config)
    sampleRate = 1000 / np.diff(source["t_ms"][:2])[0]
    geometry = ground_truth.readElementGeometry(RUN, "intact")
    voltages = {}
    for mode, band in [("uni", None), ("differential", config["analysis"]["differentialBandHz"])]:
        voltages[mode] = ground_truth.readFilteredVoltage(RUN / "intact", rawSlice, band, sampleRate,
                                                          config["analysis"]["filterOrder"])[:, measureSlice].astype(float)
    return config, geometry, voltages


def electrodeTraces(geometry, conductivity, vertices, samples, electrode, voltages, viable):
    weights = np.zeros(16); weights[electrode] = 1.0
    field = ground_truth.finiteRecordingField(samples, weights, geometry.centres, vertices, QUADRATURE)
    coefficients = metrics.elementCoefficients(geometry, field, conductivity)
    keys = np.where(viable, 0.5, 1.5)          # bin 0 = viable, bin 1 = removed
    return {mode: metrics.binnedTraces(geometry, coefficients, v, keys, np.array([0.0, 1.0, 2.0]))
            for mode, v in voltages.items()}     # (2 bins, time)


def cache(start, stop):
    CACHE.mkdir(parents=True, exist_ok=True)
    config, geometry, voltages = setup()
    conductivity = synthesis.sourceConductivity(geometry, "monodomain")
    vertices = geometry.nodePositions[geometry.elementNodes]
    viable = signedDistance(geometry.centres, config["boundary"]) < 0
    jobs = [("lesion", p) for p in range(start, stop)]
    if start == 0:
        jobs += [("intact", p) for p in sorted(set(intactReference().placement))]
    lesion = pd.read_csv(ANALYSED / "results" / "lesionPoints.csv")
    intact = intactReference()
    for condition, placement in jobs:
        target = CACHE / f"{condition}_{placement:03d}.npz"
        if target.exists():
            continue
        rows = (lesion[(lesion.noiseRepeat == 0) & (lesion.placement == placement)] if condition == "lesion"
                else intact[intact.placement == placement])
        electrodes = sorted(set(np.flatnonzero(np.abs(rows[WEIGHTS].values).max(axis=0) > 1e-12)))
        samples = np.load(RUN / condition / "egm.npz")["uni_sample_xyz"][placement]
        traces = {e: electrodeTraces(geometry, conductivity, vertices, samples, e, voltages, viable) for e in electrodes}
        np.savez(target, electrodes=np.array(electrodes),
                 **{f"{mode}_{e}": traces[e][mode] for e in electrodes for mode in voltages})
        print(condition, placement, "electrodes", electrodes, flush=True)


def intactReference():
    points = pd.read_csv(ANALYSED / "results" / "intactPoints.csv")
    return points[(points.noiseRepeat == 0) & np.isclose(points.centreYmm, 0.0)]


def combine(archive, weights, mode, bins):
    return sum(w * archive[f"{mode}_{e}"][bins].sum(axis=0) for e, w in enumerate(weights) if abs(w) > 1e-12)


def assemble():
    with np.load(ANALYSED / "results" / "intactTraces.npz") as archive:
        recoveredMv = archive["cleanAtSelectedWeightsMv"]
    scales = {}
    for record in intactReference().to_dict("records"):
        with np.load(CACHE / f"intact_{record['placement']:03d}.npz") as archive:
            mode = "uni" if record["kind"] == "uni" else "differential"
            model = combine(archive, np.array([record[w] for w in WEIGHTS]), mode, slice(None))
        recovered = recoveredMv[record["traceIndex"]].astype(float)
        scales[(record["kind"], record["label"], record["gapMm"])] = float(model @ recovered / (model @ model))
    lesion = pd.read_csv(ANALYSED / "results" / "lesionPoints.csv")
    rows = []
    for record in lesion[lesion.noiseRepeat == 0].to_dict("records"):
        mode = "uni" if record["kind"] == "uni" else "differential"
        scale = scales[(record["kind"], record["label"], record["gapMm"])]
        with np.load(CACHE / f"lesion_{record['placement']:03d}.npz") as archive:
            weights = np.array([record[w] for w in WEIGHTS])
            predicted = scale * combine(archive, weights, mode, slice(0, 1))
            unchangedTissue = scale * combine(archive, weights, mode, slice(None))
        rows.append(dict(recordingId=record["recordingId"], kind=record["kind"], label=record["label"],
                         gapMm=record["gapMm"], centreYmm=record["centreYmm"],
                         distanceInsideMm=record["distanceInsideMm"],
                         measuredPpMv=record["cleanAtSelectedWeightsPpMv"],
                         removalPredictedPpMv=float(np.ptp(predicted)),
                         intactActivationAllTissuePpMv=float(np.ptp(unchangedTissue)),
                         modelToMvScale=scale))
    table = pd.DataFrame(rows)
    table["measuredOverPredicted"] = table.measuredPpMv / table.removalPredictedPpMv
    table.to_csv(OUT / "removalPrediction.csv", index=False)
    print(json.dumps({str(k): v for k, v in scales.items()}, indent=1))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=["cache", "assemble"])
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--stop", type=int, default=130)
    arguments = parser.parse_args()
    cache(arguments.start, arguments.stop) if arguments.stage == "cache" else assemble()
