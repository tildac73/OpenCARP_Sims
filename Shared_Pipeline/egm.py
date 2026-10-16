#!/usr/bin/env python3
"""
egmpipe - shared EnSite-style acquisition chain for simulated HD Grid electrograms.

Model-agnostic: any model (2D sheet, bilayer, 3D) only has to provide, per run folder,
an egm.npz with
    uni     (P, 16, T)  unipolar electrograms in mV; P catheter placements,
                        electrode e = row*4 + col (col = along spline, row = across)
    uni_xy  (P, 16, 2+) electrode positions in mm (x, y[, z]); z defaults to standoff_mm
                        above the top of the tissue
    t_ms    (T,)        time axis
optionally
    d_mm            (P, 16)       signed distance to the scar boundary used for d_mm
                                  (negative = healthy); default x
    uni_sample_xyz  (P, 16, S, 3) points averaged into each unipole (finite electrodes),
                                  used by ground_truth.py; default the electrode position
    centre_xy_mm    (P, 2)        catheter centre of each placement; default (xc_mm, 0)
    p_<name>        (P,)          per-placement metadata -> column <name> (e.g. p_gap_mm)
and metadata either in meta.json (any keys, e.g. model, cv, state, stim_angle) or in
the run-folder name used by run.py.

Chain (all settings in a JSON config):
    1. gain        calibrate amplitude (fixed, or auto: match a healthy bipolar median
                   measured on the 'pre' runs)
    2. noise       white Gaussian noise per electrode, independent per placement
                   (fixed SD, or auto: SD giving a target filtered bipolar noise floor)
    3. filters     unipolar band (or none) and differential band, Butterworth
    4. recordings  unipolar, bipolar (along / across / both / max_orthogonal), omnipolar
    5. measure     peak-to-peak within an optional window of interest, and peak frequency
                   (bipolar/omnipolar: frequency of the power-spectrum maximum, as openCARP
                   igbdft --op DomF)
    6. export      points.csv (everything) + xlsx files in Wang's Rmd input format

Usage:
    python3 egm.py configs/clin_default.json
    python3 egm.py configs/noise_sweep.json          # "sweep" key -> one output per value
Outputs go to results/<config name>[__<sweep values>]/
"""
import argparse
import copy
import hashlib
import glob
import itertools
import json
import os
import re
import sys

import numpy as np
from scipy.signal import butter, sosfilt, sosfiltfilt

REPO = os.path.dirname(os.path.abspath(__file__))

DEFAULTS = {
    "name": "default",
    "runs": "cv*/20*_simple_*",  # glob, relative to the repo folder
    "out_dir": "results",
    "seed": 1,
    "gain": {
        "mode": "auto",
        "value": 1.0,
        "target_bipolar_healthy_mV": 1.44,
    },  # Clinical Table 2, inferior healthy bipolar median
    "noise": {
        "mode": "floor",
        "sd_mV": 0.0,
        "floor_bipolar_mV": 0.04,
    },  # median filtered bipolar noise peak-to-peak
    "filters": {"unipolar_hz": None, "differential_hz": [30, 500], "order": 2, "zero_phase": False},
    "bipolar_mode": "max_orthogonal",  # along | across | both | max_orthogonal
    "omni_angle_step_deg": 2,
    "woi_ms": None,  # [start, end] or None = whole trace
    "peak_freq": {
        "enabled": True,
        "band_hz": None,  # search band; None = filters.differential_hz
        "nfft": 4096,  # zero-padded FFT length (finer frequency grid)
        "window_ms": 50,  # window of interest centred on the largest
        # deflection (local activation); None = whole trace
        "min_snr": 3.0,
    },  # PF = 0 (undetectable) if PP < min_snr x the
    # noise-only PP of that recording kind
    "wang_export": {
        "enabled": True,
        "patient_field": "cv",  # metadata field -> patientID
        "location_field": "stim_angle",  # metadata field -> inf0_roof1
        "location_map": {"0": 0, "45": 0, "90": 1},  # Which runs count as 'roof'
        "boundary_offset_mm": 3.0,  # Clinical pipeline adds 3 mm: DistAcrossPath = d - 3
    },
    "analysis_window_last_ms": None,
    "filter_warmup_ms": 100.0,
    "sweep": {},  # {"noise.floor_bipolar_mV": [0.02, 0.04]}
}

RUN_RE = re.compile(r"_(pre|post_lw([\d.]+)_lg([\d.eE+-]+))_a(\d+)_dx([\d.]+)_cv([\d.]+)_")


# ---------------- config ----------------
def deep_update(base, new):
    out = copy.deepcopy(base)
    for k, v in new.items():
        out[k] = (
            deep_update(out[k], v) if isinstance(v, dict) and isinstance(out.get(k), dict) else v
        )
    return out


def set_path(cfg, dotted, value):
    d = cfg
    keys = dotted.split(".")
    for k in keys[:-1]:
        d = d[k]
    d[keys[-1]] = value


def expand_sweep(cfg):
    """One config per combination of sweep values."""
    sweep = cfg.get("sweep") or {}
    if not sweep:
        return [cfg]
    keys = list(sweep)
    out = []
    for vals in itertools.product(*[sweep[k] for k in keys]):
        c = copy.deepcopy(cfg)
        c["sweep"] = {}
        for k, v in zip(keys, vals):
            set_path(c, k, v)
        c["name"] = (
            cfg["name"]
            + "__"
            + "_".join("{}={}".format(k.split(".")[-1], v) for k, v in zip(keys, vals))
        )
        out.append(c)
    return out


# ---------------- inputs ----------------
def run_meta(folder):
    """Metadata from meta.json if present, else from run.py's folder name."""
    meta = {"run": os.path.basename(folder.rstrip("/")), "model": "sheet2d"}
    m = RUN_RE.search(meta["run"] + "_")
    if m:
        meta.update(
            state="post" if m.group(1).startswith("post") else "pre",
            stim_angle=int(m.group(4)),
            dx=float(m.group(5)),
            cv=float(m.group(6)),
        )
    mj = os.path.join(folder, "meta.json")
    if os.path.exists(mj):
        with open(mj) as stream:
            meta.update(json.load(stream))
    return meta


def run_folders(cfg):
    """Resolve either a legacy glob or an explicit list of stage folders."""
    patterns = cfg["runs"]
    if isinstance(patterns, str):
        patterns = [patterns]
    folders = []
    for pattern in patterns:
        matches = sorted(glob.glob(os.path.join(REPO, pattern)))
        valid = [f for f in matches if os.path.isfile(os.path.join(f, "egm.npz"))]
        if not valid:
            raise FileNotFoundError(f"No egm.npz found for {pattern}")
        folders.extend(valid)
    return sorted(set(folders))


def analysis_slices(t, cfg):
    """Raw frames including filter warmup, and measurement frames within that slice."""
    t = np.asarray(t)
    if t.ndim != 1 or len(t) < 2 or not np.isfinite(t).all():
        raise ValueError("Time axis must contain at least two finite samples")
    increments = np.diff(t)
    if increments[0] <= 0 or not np.allclose(increments, increments[0], rtol=1e-5):
        raise ValueError("Time axis must increase uniformly")
    start, stop = 0, len(t)
    duration = cfg.get("analysis_window_last_ms")
    if duration is not None:
        if not np.isfinite(duration) or duration <= 0:
            raise ValueError("analysis_window_last_ms must be positive")
        start = max(start, len(t) - max(2, int(round(duration / increments[0]))))
    if cfg["woi_ms"] is not None:
        window = woi_slice(t, cfg["woi_ms"])
        start, stop = max(start, window.start), min(stop, window.stop)
    if stop - start < 2:
        raise ValueError("Analysis window contains fewer than two samples")
    warmup = cfg.get("filter_warmup_ms", 100.0)
    if not np.isfinite(warmup) or warmup < 0:
        raise ValueError("filter_warmup_ms must be non-negative")
    # Legacy whole-record filtering is retained unless a bounded analysis window is set.
    raw_start = max(0, start - int(round(warmup / increments[0]))) if duration else 0
    return slice(raw_start, stop), slice(start - raw_start, stop - raw_start)


def load_runs(cfg):
    runs = []
    for folder in run_folders(cfg):
        meta = run_meta(folder)
        if "state" not in meta:
            raise ValueError(f"{folder}: missing state in metadata")
        with np.load(os.path.join(folder, "egm.npz")) as z:
            t = z["t_ms"]
            raw, window = analysis_slices(t, cfg)
            uni = z["uni"]
            xy = z["uni_xy"][..., :2].astype(np.float64)
            if uni.ndim != 3 or uni.shape[1:] != (16, len(t)) or xy.shape != uni.shape[:2] + (2,):
                raise ValueError(
                    f"{folder}: expected uni (placements, 16, time) and matching positions"
                )
            if not np.isfinite(uni).all() or not np.isfinite(xy).all():
                raise ValueError(f"{folder}: non-finite electrode data")
            d = z["d_mm"] if "d_mm" in z.files else xy[..., 0]
            runs.append(
                dict(
                    meta=meta,
                    uni=uni[..., raw].astype(np.float64),
                    t=t[raw],
                    window=window,
                    xy=np.concatenate([xy, d[..., None]], axis=-1),
                    pmeta={k[2:]: z[k] for k in z.files if k.startswith("p_")},
                )
            )
    return runs


# ---------------- signal chain ----------------
def make_filter(band, fs, order):
    if band is None:
        return None
    if len(band) != 2 or not 0 < band[0] < band[1] < fs / 2:
        raise ValueError(
            f"Filter band {band} must lie strictly between 0 and Nyquist ({fs / 2:g} Hz)"
        )
    return butter(order, band, btype="bandpass", fs=fs, output="sos")


def apply_filter(sos, x, zero_phase):
    if sos is None:
        return x
    return sosfiltfilt(sos, x, axis=-1) if zero_phase else sosfilt(sos, x, axis=-1)


def bipolar_pairs(mode):
    """List of electrode pairs (a, b, b2): bipole b - a; for max_orthogonal also a -> b2."""
    along = [(r * 4 + c, r * 4 + c + 1) for r in range(4) for c in range(3)]
    across = [(r * 4 + c, (r + 1) * 4 + c) for r in range(3) for c in range(4)]
    if mode == "along":
        return [(a, b, None) for a, b in along]
    if mode == "across":
        return [(a, b, None) for a, b in across]
    if mode == "both":
        return [(a, b, None) for a, b in along + across]
    if mode == "max_orthogonal":  # orthogonal pair sharing vertex (r, c)
        return [(r * 4 + c, r * 4 + c + 1, (r + 1) * 4 + c) for r in range(3) for c in range(3)]
    sys.exit("unknown bipolar_mode " + mode)


def omni_cliques():
    """(vertex, x-neighbour, y-neighbour, sx, sy) for the 4 right-angle cliques of each square."""
    out = []
    for r in range(3):
        for c in range(3):
            sq = {
                (0, 0): r * 4 + c,
                (1, 0): r * 4 + c + 1,
                (0, 1): (r + 1) * 4 + c,
                (1, 1): (r + 1) * 4 + c + 1,
            }
            for (vx, vy), v in sq.items():
                out.append((v, sq[(1 - vx, vy)], sq[(vx, 1 - vy)], 1 - 2 * vx, 1 - 2 * vy))
    return out


def recordings(uf, df, xy, cfg, omni=True, with_weights=False):
    """Build traces and their exact electrode weights (including noisy orientation choices).

    Each result is (kind, label, position, trace). With with_weights=True, append
    a (placements, 16) weight array, so ground truth uses the same chosen recording.
    """
    count = len(xy)
    basis = np.eye(16)
    out = []

    def add(kind, label, position, trace, weights):
        row = (kind, label, position, trace)
        if with_weights:
            row += (np.broadcast_to(weights, (count, 16)).copy(),)
        out.append(row)

    for e in range(16):
        add("uni", f"E{e}", xy[:, e], uf[:, e], basis[e])
    for a, b, b2 in bipolar_pairs(cfg["bipolar_mode"]):
        first = df[:, b] - df[:, a]
        if b2 is None:
            add("bi", f"E{a}-E{b}", (xy[:, a] + xy[:, b]) / 2, first, basis[b] - basis[a])
        else:
            second = df[:, b2] - df[:, a]
            choose_first = np.ptp(first, axis=-1) >= np.ptp(second, axis=-1)
            weights = np.where(choose_first[:, None], basis[b] - basis[a], basis[b2] - basis[a])
            add(
                "bi",
                f"E{a}-E{b}|E{b2}",
                (xy[:, a] + xy[:, b] + xy[:, b2]) / 3,
                np.where(choose_first[:, None], first, second),
                weights,
            )
    if omni:
        step = cfg["omni_angle_step_deg"]
        if not 0 < step <= 180:
            raise ValueError("omni_angle_step_deg must be in (0, 180]")
        angles = np.deg2rad(np.arange(0, 180, step))
        for v, hx, hy, sx, sy in omni_cliques():
            bx, by = sx * (df[:, hx] - df[:, v]), sy * (df[:, hy] - df[:, v])
            # Scan angles without allocating a placements x angles x time array.
            best_pp = np.full(count, -np.inf)
            best_angle = np.zeros(count)
            for angle in angles:
                pp = np.ptp(np.cos(angle) * bx + np.sin(angle) * by, axis=-1)
                better = pp > best_pp
                best_pp[better], best_angle[better] = pp[better], angle
            cosine, sine = np.cos(best_angle)[:, None], np.sin(best_angle)[:, None]
            weights = cosine * sx * (basis[hx] - basis[v]) + sine * sy * (basis[hy] - basis[v])
            add(
                "omni",
                f"E{v}-E{hx}-E{hy}",
                (xy[:, v] + xy[:, hx] + xy[:, hy]) / 3,
                cosine * bx + sine * by,
                weights,
            )
    return out


def woi_slice(t, woi):
    if woi is None:
        return slice(None)
    if len(woi) != 2 or woi[0] > woi[1]:
        raise ValueError("woi_ms must be [start, end] in increasing order")
    i = np.flatnonzero((t >= woi[0]) & (t <= woi[1]))
    if not len(i):
        raise ValueError(f"Window {woi} does not overlap time range {t[0]} to {t[-1]} ms")
    return slice(i[0], i[-1] + 1)


def peak_frequency(tr, fs, band, nfft, window_ms=None):
    """
    Peak (dominant) frequency of each trace: frequency of the maximum of the power spectrum
    within band, as openCARP igbdft --op DomF. With window_ms, only a window of that length
    centred on the trace's largest deflection is used (the local activation, as a mapping
    system's window of interest), so slow far-field and repolarisation content elsewhere in the
    trace does not set the PF. tr (P, T) -> (P,) Hz; 0 for a flat trace.
    """
    if window_ms:
        w = min(max(2, int(round(window_ms * fs / 1000.0))), tr.shape[-1])
        centre = np.argmax(np.abs(tr - tr.mean(axis=-1, keepdims=True)), axis=-1)
        start = np.clip(centre - w // 2, 0, tr.shape[-1] - w)
        tr = np.take_along_axis(tr, start[:, None] + np.arange(w)[None, :], axis=-1)
    x = tr - tr.mean(axis=-1, keepdims=True)
    x = x * np.hanning(x.shape[-1])
    n = max(int(nfft), x.shape[-1])
    power = np.abs(np.fft.rfft(x, n=n, axis=-1)) ** 2
    f = np.fft.rfftfreq(n, 1.0 / fs)
    sel = (f >= band[0]) & (f <= min(band[1], fs / 2))
    if not np.any(sel):
        raise ValueError(f"Peak-frequency band {band} contains no FFT bins at sampling rate {fs}")
    p = power[:, sel]
    return np.where(p.max(axis=-1) > 0, f[sel][np.argmax(p, axis=-1)], 0.0)


def process_run(run, cfg, gain, sd, rng, noise_pp):
    fs = 1000.0 / float(np.median(np.diff(run["t"])))
    F = cfg["filters"]
    uni = gain * run["uni"]
    if sd > 0:
        uni = uni + rng.normal(0.0, sd, uni.shape)
    uf = apply_filter(make_filter(F["unipolar_hz"], fs, F["order"]), uni, F["zero_phase"])
    df = apply_filter(make_filter(F["differential_hz"], fs, F["order"]), uni, F["zero_phase"])
    import pandas as pd

    w = run.get("window", woi_slice(run["t"], cfg["woi_ms"]))
    PF = cfg["peak_freq"]
    pf_band = PF["band_hz"] or F["differential_hz"] or [0.0, fs / 2]
    parts = []
    for kind, label, pos, tr, weights in recordings(
        uf[..., w], df[..., w], run["xy"], cfg, with_weights=True
    ):
        pp = np.ptp(tr, axis=-1)
        if PF["enabled"] and kind != "uni":
            pf = peak_frequency(tr, fs, pf_band, PF["nfft"], PF["window_ms"])
            pf[pp < PF["min_snr"] * noise_pp[kind]] = 0.0  # undetectable, as clinically
        else:
            pf = np.full(len(pos), np.nan)  # no unipolar peak frequency, as in the clinical data
        part = pd.DataFrame(
            {
                "placement": np.arange(len(pos)),
                "kind": kind,
                "label": label,
                "x_mm": pos[:, 0],
                "y_mm": pos[:, 1],
                "d_mm": pos[:, 2],
                "PP_mV": pp,
                "PF_Hz": pf,
                "window_start_ms": run["t"][w][0],
                "window_end_ms": run["t"][w][-1],
            }
        )
        for electrode in range(16):
            part[f"weight_E{electrode}"] = weights[:, electrode]
        parts.append(part)
    out = pd.concat(parts, ignore_index=True)
    for k, v in run["pmeta"].items():
        out.insert(0, k, v[out.placement.values])
    for k, v in run["meta"].items():
        out.insert(0, k, v)
    return out


# ---------------- calibration ----------------
def median_bipolar_pp(runs, cfg, unit_noise=False, rng=None):
    """Median filtered bipolar peak-to-peak: of the signal (pre runs) or of unit-SD noise."""
    vals = []
    for run in runs:
        fs = 1000.0 / float(np.median(np.diff(run["t"])))
        x = rng.normal(0.0, 1.0, run["uni"].shape) if unit_noise else run["uni"]
        df = apply_filter(
            make_filter(cfg["filters"]["differential_hz"], fs, cfg["filters"]["order"]),
            x,
            cfg["filters"]["zero_phase"],
        )[..., run.get("window", woi_slice(run["t"], cfg["woi_ms"]))]
        for kind, _, _, tr in recordings(df, df, run["xy"], cfg, omni=False):
            if kind == "bi":
                vals.append(np.ptp(tr, axis=-1))
        if unit_noise:
            break  # one run is plenty for the noise statistic
    return float(np.median(np.concatenate(vals)))


def noise_levels(runs, cfg, sd):
    """Median filtered peak-to-peak of noise alone (SD sd), per recording kind -> {kind: mV}."""
    run = runs[0]
    fs = 1000.0 / float(np.median(np.diff(run["t"])))
    F = cfg["filters"]
    x = np.random.default_rng(54321).normal(0.0, sd, run["uni"].shape)
    uf = apply_filter(make_filter(F["unipolar_hz"], fs, F["order"]), x, F["zero_phase"])
    df = apply_filter(make_filter(F["differential_hz"], fs, F["order"]), x, F["zero_phase"])
    w = run.get("window", woi_slice(run["t"], cfg["woi_ms"]))
    pp = {}
    for kind, _, _, tr in recordings(uf[..., w], df[..., w], run["xy"], cfg):
        pp.setdefault(kind, []).append(np.ptp(tr, axis=-1))
    return {k: float(np.median(np.concatenate(v))) for k, v in pp.items()}


def calibrate(runs, cfg):
    g = cfg["gain"]
    if g["mode"] not in ("fixed", "auto") or cfg["noise"]["mode"] not in ("sd", "none", "floor"):
        raise ValueError("Unknown gain or noise mode")
    if g["mode"] == "fixed":
        gain = float(g["value"])
    else:
        calibration = (
            load_runs(dict(cfg, runs=cfg["calibration_runs"]))
            if cfg.get("calibration_runs")
            else runs
        )
        pre = [r for r in calibration if r["meta"]["state"] == "pre"]
        if not pre:
            sys.exit("gain.mode = auto needs pre runs")
        reference = median_bipolar_pp(pre, cfg)
        if not np.isfinite(reference) or reference <= 0:
            raise ValueError("Auto gain requires a nonzero finite baseline bipolar signal")
        gain = g["target_bipolar_healthy_mV"] / reference
    n = cfg["noise"]
    if n["mode"] == "sd":
        sd = float(n["sd_mV"])
    elif n["mode"] == "none":
        sd = 0.0
    else:
        unit = median_bipolar_pp(runs, cfg, unit_noise=True, rng=np.random.default_rng(12345))
        sd = n["floor_bipolar_mV"] / unit
    if not np.isfinite(gain) or gain <= 0 or not np.isfinite(sd) or sd < 0:
        raise ValueError("Gain must be positive and noise SD non-negative, both finite")
    return gain, sd


# ---------------- export ----------------
def export_wang(df, cfg, out):
    import pandas as pd

    W = cfg["wang_export"]
    pats = {v: i + 1 for i, v in enumerate(sorted(df[W["patient_field"]].unique()))}
    loc_map = {str(k): v for k, v in W["location_map"].items()}
    wang = pd.DataFrame(
        {
            "patientID": df[W["patient_field"]].map(pats),
            "inf0_roof1": df[W["location_field"]].astype(str).map(loc_map),
            "uni1_bi2_omni3": df["kind"].map({"uni": 1, "bi": 2, "omni": 3}),
            "DistAcrossPath": df["d_mm"] - W["boundary_offset_mm"],
            "PP": df["PP_mV"],
            "PFreq": df["PF_Hz"],
            "state": df["state"],
        }
    )
    if wang["inf0_roof1"].isna().any():
        sys.exit("location_map does not cover all values of " + W["location_field"])
    post, pre = wang[wang.state == "post"].drop(columns="state"), wang[wang.state == "pre"].drop(
        columns="state"
    )
    post.to_excel(os.path.join(out, "merged_rectangle_summary_euclidean.xlsx"), index=False)
    pre.to_excel(os.path.join(out, "merged_rectangle_summary_euclidean_pre.xlsx"), index=False)
    # Flat sheet: geodesic = Euclidean. Placeholder validation files so the Rmd runs unchanged.
    import shutil

    shutil.copy(
        os.path.join(out, "merged_rectangle_summary_euclidean.xlsx"),
        os.path.join(out, "merged_rectangle_summary_geodesic.xlsx"),
    )
    v = np.linspace(-2.5, 9.5, 50)
    np.savetxt(os.path.join(out, "auto_y.txt"), v, fmt="%.3f")
    np.savetxt(os.path.join(out, "manual_y.txt"), np.round(v), fmt="%.0f")
    with open(os.path.join(out, "patient_map.json"), "w") as f:
        json.dump({str(k): v for k, v in pats.items()}, f, indent=1)


def check_peak_frequency(df, cfg):
    """
    Print and return PF per recording kind: share undetectable (PF = 0), median of the rest,
    and the share sitting on an edge of the search band - a PF pinned there is not a spectral
    peak (the true peak lies outside the band), and constant PFs break the R mixed models.
    """
    out = {}
    if not cfg["peak_freq"]["enabled"]:
        return out
    band = cfg["peak_freq"]["band_hz"] or cfg["filters"]["differential_hz"]
    for kind, d in df[df.kind != "uni"].groupby("kind"):
        det = d.PF_Hz[d.PF_Hz > 0]
        edge = (
            float(np.mean((det - band[0] < 2) | (band[1] - det < 2))) if band and len(det) else 0.0
        )
        out[kind] = dict(
            undetectable=float(np.mean(d.PF_Hz == 0)),
            median_Hz=float(det.median()) if len(det) else None,
            at_band_edge=edge,
        )
        print(
            "    PF {}: {:.0%} undetectable, median {} Hz, {:.0%} at a band edge{}".format(
                kind,
                out[kind]["undetectable"],
                "n/a" if not len(det) else "{:.0f}".format(out[kind]["median_Hz"]),
                edge,
                "  <-- WARNING: PF pinned to the search band" if edge > 0.2 else "",
            )
        )
    return out


def run_config(cfg):
    import pandas as pd

    runs = load_runs(cfg)
    gain, sd = calibrate(runs, cfg)
    out = os.path.join(REPO, cfg["out_dir"], cfg["name"])
    os.makedirs(out, exist_ok=True)
    noise_pp = noise_levels(runs, cfg, sd)
    print(
        "[{}] {} runs | gain {:.4g} | noise SD {:.4g} mV | noise PP {}".format(
            cfg["name"], len(runs), gain, sd, {k: round(v, 4) for k, v in noise_pp.items()}
        )
    )

    parts = []
    for run in runs:
        seed = int(
            hashlib.md5("{}|{}".format(cfg["seed"], run["meta"]["run"]).encode()).hexdigest()[:8],
            16,
        )
        parts.append(process_run(run, cfg, gain, sd, np.random.default_rng(seed), noise_pp))
    df = pd.concat(parts, ignore_index=True)
    df.to_csv(os.path.join(out, "points.csv"), index=False)
    pf_summary = check_peak_frequency(df, cfg)

    healthy = df[(df.state == "pre")].groupby("kind").PP_mV.median().to_dict()
    summary = dict(
        config=cfg,
        gain=gain,
        noise_sd_mV=sd,
        noise_PP_mV=noise_pp,
        n_points=len(df),
        pre_median_PP_mV=healthy,
        peak_frequency=pf_summary,
        wang_table2_inferior_healthy_mV={"uni": 1.83, "bi": 1.44, "omni": 2.05},
    )
    with open(os.path.join(out, "run_summary.json"), "w") as f:
        json.dump(summary, f, indent=1, default=str)
    if cfg["wang_export"]["enabled"]:
        export_wang(df, cfg, out)
    print(
        "[{}] {} points -> {}  | pre medians (mV): {}".format(
            cfg["name"], len(df), out, {k: round(v, 3) for k, v in healthy.items()}
        )
    )


def main():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("config", nargs="+", help="JSON config file(s)")
    a = p.parse_args()
    for path in a.config:
        with open(path) as stream:
            cfg = deep_update(DEFAULTS, json.load(stream))
        for c in expand_sweep(cfg):
            run_config(c)


if __name__ == "__main__":
    main()
