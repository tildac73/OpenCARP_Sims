#!/usr/bin/env python3
"""Step 2 gate: does electrode_synthesis reproduce openCARP's recovered phi_e from saved Vm?

For a block of recovery points, computes phi_e from vm.igb with
Shared_Pipeline/electrode_synthesis.py and compares it with phie_recovery.igb over the same
frames. Metrics per point: correlation, relative RMS error, peak-to-peak (amplitude) ratio.
The amplitude ratio tests the prefactor and source conductivity, not just waveform shape.

Default frames: 7500-7600 ms (beat 16: stimulus, strip far-field and local activation).
Chunked so each call fits a short time limit; results are written per chunk, then summarised.

Usage (repository root):
    python3 3D_Code/numerics/verify_synthesis.py "boundaryPilotV2 2" intact --start 0 --count 1000
    python3 3D_Code/numerics/verify_synthesis.py "boundaryPilotV2 2" intact --summarise
Outputs: <run>/../_verification/synthesis_<condition>_<start>.csv and synthesisSummary_<condition>.json
"""
import argparse
import json
from pathlib import Path
import sys
import time

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "Shared_Pipeline"))
import electrode_synthesis as synthesis  # noqa: E402
import ground_truth  # noqa: E402
import igb_io  # noqa: E402

GATE_MIN_CORRELATION = 0.999  # PROPOSED, readiness plan Step 2


def frameSlice(path, startMs, endMs):
    times = igb_io.timeAxis(path)
    first, stop = int(np.searchsorted(times, startMs)), int(np.searchsorted(times, endMs))
    if stop - first < 2:
        raise ValueError(f"No frames in [{startMs}, {endMs}) ms")
    return slice(first, stop)


def runChunk(run, condition, start, count, startMs, endMs, outFolder, sigmaBath, source):
    began = time.time()
    geometry = ground_truth.readElementGeometry(run, condition)
    points = np.loadtxt(run / "electrodes" / "recoveryPoints.pts", skiprows=1, ndmin=2) / 1000.0
    stop = min(start + count, len(points))
    vmPath, phiePath = run / condition / "vm.igb", run / condition / "phie_recovery.igb"
    frames = frameSlice(vmPath, startMs, endMs)
    if frameSlice(phiePath, startMs, endMs) != frames:
        raise ValueError("vm.igb and phie_recovery.igb time axes differ")
    voltage = igb_io.read(vmPath, frames).T.astype(np.float64)              # (nodes, time)
    recovered = igb_io.read(phiePath, frames)[:, start:stop].T.astype(np.float64)
    synthesised = synthesis.pointOperator(geometry, points[start:stop], sigmaBathSPerM=sigmaBath,
                                          source=source) @ voltage
    rows = []
    for offset, (a, b) in enumerate(zip(synthesised, recovered)):
        ac, bc = a - a.mean(), b - b.mean()
        rows.append(dict(point=start + offset, zMm=points[start + offset, 2],
                         correlation=float(ac @ bc / np.sqrt((ac @ ac) * (bc @ bc))),
                         relativeRmsError=float(np.linalg.norm(a - b) / np.linalg.norm(b)),
                         amplitudeRatio=float(np.ptp(a) / np.ptp(b)),
                         recoveredPeakToPeakMv=float(np.ptp(b))))
    outFolder.mkdir(parents=True, exist_ok=True)
    table = pd.DataFrame(rows)
    table.to_csv(outFolder / f"synthesis_{condition}_{start:05d}.csv", index=False)
    print(f"{condition} points {start}-{stop - 1}: corr min {table.correlation.min():.6f}, "
          f"amplitude ratio {table.amplitudeRatio.min():.4f}-{table.amplitudeRatio.max():.4f}, "
          f"rel RMS max {table.relativeRmsError.max():.4f} ({time.time() - began:.0f} s)")


def summarise(condition, outFolder, totalPoints):
    table = pd.concat([pd.read_csv(p) for p in sorted(outFolder.glob(f"synthesis_{condition}_*.csv"))],
                      ignore_index=True).drop_duplicates("point")
    heightBins = pd.cut(table.zMm, [2.0, 2.5, 3.0, 4.01])
    summary = dict(
        condition=condition, pointsChecked=int(len(table)), pointsTotal=int(totalPoints),
        correlationMin=float(table.correlation.min()),
        correlationP01=float(table.correlation.quantile(0.01)),
        relativeRmsErrorMax=float(table.relativeRmsError.max()),
        relativeRmsErrorMedian=float(table.relativeRmsError.median()),
        amplitudeRatioMin=float(table.amplitudeRatio.min()),
        amplitudeRatioMedian=float(table.amplitudeRatio.median()),
        amplitudeRatioMax=float(table.amplitudeRatio.max()),
        byHeightAboveTissueMm={str(k): dict(n=int(len(g)), correlationMin=float(g.correlation.min()),
                                            relativeRmsErrorMax=float(g.relativeRmsError.max()))
                               for k, g in table.groupby(heightBins, observed=True)},
        gateMinCorrelationProposed=GATE_MIN_CORRELATION,
        gatePassed=bool(len(table) == totalPoints and table.correlation.min() >= GATE_MIN_CORRELATION),
    )
    (outFolder / f"synthesisSummary_{condition}.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("run", type=Path)
    parser.add_argument("condition", choices=["intact", "lesion"])
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--count", type=int, default=500)
    parser.add_argument("--window", type=float, nargs=2, default=[7500.0, 7600.0], metavar=("START_MS", "END_MS"))
    parser.add_argument("--sigma-bath", type=float, default=1.0, help="S/m; openCARP g_bath default")
    parser.add_argument("--source", choices=["monodomain", "intracellular"], default="monodomain")
    parser.add_argument("--summarise", action="store_true")
    arguments = parser.parse_args()
    run = arguments.run.resolve()
    outFolder = ROOT / "_verification" / run.name.replace(" ", "_")
    total = len(np.loadtxt(run / "electrodes" / "recoveryPoints.pts", skiprows=1, ndmin=2))
    if arguments.summarise:
        summarise(arguments.condition, outFolder, total)
    else:
        runChunk(run, arguments.condition, arguments.start, arguments.count, *arguments.window,
                 outFolder, arguments.sigma_bath, arguments.source)


if __name__ == "__main__":
    main()
