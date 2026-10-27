#!/usr/bin/env python3
"""Step 3 (analysis-only part): electrode surface-sampling convergence from saved Vm.

Recomputes the 16 x 130 electrode signals of boundaryPilotV2 with a chosen number of surface
samples per electrode (axial x angular), using the verified electrode_synthesis module
(VL-2026-10-28-02). Both the baseline (2 x 4 = 8) and refined settings are synthesised with the
same code, so differences isolate the surface sampling, not openCARP-vs-synthesis quadrature.

Stages (repository root):
  cache      copy the Vm frames needed (7125-7600 ms) to fast local storage (once per condition)
  synthesise compute point potentials for a block of unique sample points
  assemble   average samples into electrodes and write an egm.npz in the pipeline's format
Example:
  python3 3D_Code/numerics/electrode_sampling.py cache intact --cache ~/vmcache
  python3 3D_Code/numerics/electrode_sampling.py synthesise intact --axial 4 --angular 8 --start 0 --count 4000 --cache ~/vmcache
  python3 3D_Code/numerics/electrode_sampling.py assemble intact --axial 4 --angular 8
"""
import argparse
import json
from pathlib import Path
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "3D_Code"), str(ROOT / "Shared_Pipeline")]
import electrode_synthesis as synthesis  # noqa: E402
import ground_truth  # noqa: E402
import igb_io  # noqa: E402
from boundary_geometry import makeElectrodeLayout  # noqa: E402

RUN = ROOT / "boundaryPilotV2 2"
FRAMES_MS = (7125.0, 7600.0)   # inclusive; 400 ms filter-history check before the 7525 ms window start
OUT = ROOT / "_verification" / "electrode_sampling"


def layoutFor(axial, angular):
    config = json.loads((RUN / "config.json").read_text())
    config["electrodes"]["axialSamples"], config["electrodes"]["angularSamples"] = axial, angular
    return makeElectrodeLayout(config)


def cache(condition, folder):
    folder.mkdir(parents=True, exist_ok=True)
    path = RUN / condition / "vm.igb"
    times = igb_io.timeAxis(path)
    # Inclusive end: the pipeline's window check needs a frame at the window end (7600 ms).
    first, stop = int(np.searchsorted(times, FRAMES_MS[0])), int(np.searchsorted(times, FRAMES_MS[1])) + 1
    voltage = igb_io.read(path, slice(first, stop))                      # (time, nodes) float32
    np.save(folder / f"{condition}_vm.npy", np.ascontiguousarray(voltage.T))   # (nodes, time)
    np.save(folder / f"{condition}_t.npy", times[first:stop])
    print(condition, voltage.shape, times[first], times[stop - 1])


def synthesise(condition, axial, angular, start, count, folder):
    began = time.time()
    geometry = ground_truth.readElementGeometry(RUN, condition)
    points = layoutFor(axial, angular)["recoveryPoints"]
    stop = min(start + count, len(points))
    voltage = np.load(folder / f"{condition}_vm.npy", mmap_mode="r")
    traces = np.empty((stop - start, voltage.shape[1]), np.float32)
    for first in range(start, stop, 250):
        last = min(first + 250, stop)
        operator = synthesis.pointOperator(geometry, points[first:last]).astype(np.float32)
        traces[first - start:last - start] = operator @ voltage
    target = OUT / f"s{axial}x{angular}" / condition
    target.mkdir(parents=True, exist_ok=True)
    np.save(target / f"points_{start:06d}.npy", traces)
    print(f"{condition} {axial}x{angular}: points {start}-{stop - 1} of {len(points)} ({time.time() - began:.0f} s)")


def timeAxisFrom(originalTimes, cachedTimes):
    """Use the run's own egm time axis (exact sample interval) for the cached frames."""
    first = int(np.argmin(np.abs(originalTimes - cachedTimes[0])))
    times = originalTimes[first:first + len(cachedTimes)]
    if len(times) != len(cachedTimes) or not np.allclose(times, cachedTimes, atol=1e-6):
        raise ValueError("Cached Vm frames do not align with the egm time axis")
    return times


def assemble(condition, axial, angular, folder):
    layout = layoutFor(axial, angular)
    target = OUT / f"s{axial}x{angular}" / condition
    blocks = sorted(target.glob("points_*.npy"))
    pointTraces = np.concatenate([np.load(b) for b in blocks])
    if len(pointTraces) != len(layout["recoveryPoints"]):
        raise ValueError(f"{len(pointTraces)} of {len(layout['recoveryPoints'])} points synthesised")
    electrodeTraces = pointTraces[layout["recoveryIndex"]].mean(axis=2)     # (placement, electrode, time)
    original = dict(np.load(RUN / condition / "egm.npz"))
    if not np.allclose(original["uni_xy"], layout["centres"]):
        raise ValueError("Electrode centres differ from the original run")
    # Keep the run's full time axis: downstream code indexes vm.igb with egm frame indices.
    # Frames outside the synthesised interval are NaN so any accidental use fails loudly.
    cached = np.load(folder / f"{condition}_t.npy")
    timeAxisFrom(original["t_ms"], cached)                                   # alignment check
    first = int(np.argmin(np.abs(original["t_ms"] - cached[0])))
    full = np.full(original["uni"].shape, np.nan, dtype=np.float32)
    full[..., first:first + electrodeTraces.shape[-1]] = electrodeTraces
    np.savez(target / "egm.npz", uni=full, uni_xy=original["uni_xy"], uni_sample_xyz=layout["samples"],
             d_mm=original["d_mm"], centre_xy_mm=original["centre_xy_mm"], p_gap_mm=original["p_gap_mm"],
             t_ms=original["t_ms"])
    (target / "provenance.json").write_text(json.dumps(dict(
        source=str(RUN.name), condition=condition, axialSamples=axial, angularSamples=angular,
        method="electrode_synthesis.pointOperator (monodomain source, sigma_b = 1 S/m)",
        framesMs=FRAMES_MS), indent=2))
    print("wrote", target / "egm.npz", electrodeTraces.shape)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("stage", choices=["cache", "synthesise", "assemble"])
    parser.add_argument("condition", choices=["intact", "lesion"])
    parser.add_argument("--axial", type=int, default=2)
    parser.add_argument("--angular", type=int, default=4)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--count", type=int, default=2000)
    parser.add_argument("--cache", type=Path, default=Path.home() / "vmcache")
    a = parser.parse_args()
    if a.stage == "cache":
        cache(a.condition, a.cache)
    elif a.stage == "synthesise":
        synthesise(a.condition, a.axial, a.angular, a.start, a.count, a.cache)
    else:
        assemble(a.condition, a.axial, a.angular, a.cache)


if __name__ == "__main__":
    main()
