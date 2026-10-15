#!/usr/bin/env python3
"""
egmpipe - shared EnSite-style acquisition chain for simulated HD Grid electrograms.

Model-agnostic: any model (2D sheet, bilayer, 3D) only has to provide, per run folder,
an egm.npz with
    uni     (P, 16, T)  unipolar electrograms in mV; P catheter placements,
                        electrode e = row*4 + col (col = along spline, row = across)
    uni_xy  (P, 16, 2+) electrode positions in mm; x = signed distance to the scar
                        boundary (negative = healthy)
    t_ms    (T,)        time axis
and metadata either in meta.json (any keys, e.g. model, cv, state, stim_angle) or in
the run-folder name used by run.py.

Chain (all settings in a JSON config):
    1. gain        calibrate amplitude (fixed, or auto: match a healthy bipolar median
                   measured on the 'pre' runs)
    2. noise       white Gaussian noise per electrode, independent per placement
                   (fixed SD, or auto: SD giving a target filtered bipolar noise floor)
    3. filters     unipolar band (or none) and differential band, Butterworth
    4. recordings  unipolar, bipolar (along / across / both / max_orthogonal), omnipolar
    5. measure     peak-to-peak within an optional window of interest
    6. export      points.csv (everything) + xlsx files in Wang's Rmd input format

Usage:
    python3 egmpipe.py configs/clin_default.json
    python3 egmpipe.py configs/noise_sweep.json          # "sweep" key -> one output per value
Outputs go to results/<config name>[__<sweep values>]/
"""
import argparse
import copy
import hashlib
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
    "runs": "cv*/20*_simple_*",            # glob, relative to the repo folder
    "out_dir": "results",
    "seed": 1,
    "gain": {"mode": "auto", "value": 1.0,
             "target_bipolar_healthy_mV": 1.44},     # Clinical Table 2, inferior healthy bipolar median
    "noise": {"mode": "floor", "sd_mV": 0.0,
              "floor_bipolar_mV": 0.04},             # median filtered bipolar noise peak-to-peak
    "filters": {"unipolar_hz": None, "differential_hz": [30, 500],
                "order": 2, "zero_phase": False},
    "bipolar_mode": "max_orthogonal",                # along | across | both | max_orthogonal
    "omni_angle_step_deg": 2,
    "woi_ms": None,                                  # [start, end] or None = whole trace
    "wang_export": {
        "enabled": True,
        "patient_field": "cv",                       # metadata field -> patientID
        "location_field": "stim_angle",              # metadata field -> inf0_roof1
        "location_map": {"0": 0, "45": 0, "90": 1},  # Which runs count as 'roof'
        "boundary_offset_mm": 3.0                    # Clinical pipeline adds 3 mm: DistAcrossPath = d - 3
    },
    "sweep": {}                                      # {"noise.floor_bipolar_mV": [0.02, 0.04]}
}

RUN_RE = re.compile(r'_(pre|post_lw([\d.]+)_lg([\d.eE+-]+))_a(\d+)_dx([\d.]+)_cv([\d.]+)_')


# ---------------- config ----------------
def deep_update(base, new):
    out = copy.deepcopy(base)
    for k, v in new.items():
        out[k] = deep_update(out[k], v) if isinstance(v, dict) and isinstance(out.get(k), dict) else v
    return out


def set_path(cfg, dotted, value):
    d = cfg
    keys = dotted.split('.')
    for k in keys[:-1]:
        d = d[k]
    d[keys[-1]] = value


def expand_sweep(cfg):
    """One config per combination of sweep values."""
    sweep = cfg.get('sweep') or {}
    if not sweep:
        return [cfg]
    keys = list(sweep)
    out = []
    for vals in itertools.product(*[sweep[k] for k in keys]):
        c = copy.deepcopy(cfg)
        c['sweep'] = {}
        for k, v in zip(keys, vals):
            set_path(c, k, v)
        c['name'] = cfg['name'] + '__' + '_'.join('{}={}'.format(k.split('.')[-1], v)
                                                  for k, v in zip(keys, vals))
        out.append(c)
    return out


# ---------------- inputs ----------------
def run_meta(folder):
    """Metadata from meta.json if present, else from run.py's folder name."""
    meta = {'run': os.path.basename(folder.rstrip('/')), 'model': 'sheet2d'}
    m = RUN_RE.search(meta['run'] + '_')
    if m:
        meta.update(state='post' if m.group(1).startswith('post') else 'pre',
                    stim_angle=int(m.group(4)), dx=float(m.group(5)), cv=float(m.group(6)))
    mj = os.path.join(folder, 'meta.json')
    if os.path.exists(mj):
        meta.update(json.load(open(mj)))
    return meta


def load_runs(cfg):
    folders = sorted(f for f in __import__('glob').glob(os.path.join(REPO, cfg['runs']))
                     if os.path.exists(os.path.join(f, 'egm.npz')))
    if not folders:
        sys.exit('No run folders with egm.npz match ' + cfg['runs'])
    runs = []
    for f in folders:
        z = np.load(os.path.join(f, 'egm.npz'))
        meta = run_meta(f)
        if 'state' not in meta:
            print('skip (no state in metadata): ' + f)
            continue
        runs.append(dict(meta=meta, uni=z['uni'].astype(np.float64),
                         xy=z['uni_xy'][..., :2].astype(np.float64), t=z['t_ms']))
    return runs


# ---------------- signal chain ----------------
def make_filter(band, fs, order):
    return None if band is None else butter(order, band, btype='bandpass', fs=fs, output='sos')


def apply_filter(sos, x, zero_phase):
    if sos is None:
        return x
    return sosfiltfilt(sos, x, axis=-1) if zero_phase else sosfilt(sos, x, axis=-1)


def bipolar_pairs(mode):
    """List of electrode pairs (a, b, b2): bipole b - a; for max_orthogonal also a -> b2."""
    along  = [(r * 4 + c, r * 4 + c + 1) for r in range(4) for c in range(3)]
    across = [(r * 4 + c, (r + 1) * 4 + c) for r in range(3) for c in range(4)]
    if mode == 'along':
        return [(a, b, None) for a, b in along]
    if mode == 'across':
        return [(a, b, None) for a, b in across]
    if mode == 'both':
        return [(a, b, None) for a, b in along + across]
    if mode == 'max_orthogonal':                 # orthogonal pair sharing vertex (r, c)
        return [(r * 4 + c, r * 4 + c + 1, (r + 1) * 4 + c) for r in range(3) for c in range(3)]
    sys.exit('unknown bipolar_mode ' + mode)


def omni_cliques():
    """(vertex, x-neighbour, y-neighbour, sx, sy) for the 4 right-angle cliques of each square."""
    out = []
    for r in range(3):
        for c in range(3):
            sq = {(0, 0): r*4 + c, (1, 0): r*4 + c + 1, (0, 1): (r+1)*4 + c, (1, 1): (r+1)*4 + c + 1}
            for (vx, vy), v in sq.items():
                out.append((v, sq[(1 - vx, vy)], sq[(vx, 1 - vy)], 1 - 2 * vx, 1 - 2 * vy))
    return out


def recordings(uf, df, xy, cfg, omni=True):
    """
    uf: unipolar after unipolar filter (P,16,T); df: unipolar after differential filter.
    Returns list of (kind, label, positions (P,2), traces (P,T)).
    """
    out = [('uni', 'E{}'.format(e), xy[:, e], uf[:, e]) for e in range(16)]
    for a, b, b2 in bipolar_pairs(cfg['bipolar_mode']):
        s1 = df[:, b] - df[:, a]
        if b2 is None:
            out.append(('bi', 'E{}-E{}'.format(a, b), (xy[:, a] + xy[:, b]) / 2, s1))
        else:                                    # keep the larger peak-to-peak of the pair
            s2 = df[:, b2] - df[:, a]
            pick = (np.ptp(s1, axis=-1) >= np.ptp(s2, axis=-1))[:, None]
            out.append(('bi', 'E{}-E{}|E{}'.format(a, b, b2),
                        (xy[:, a] + xy[:, b] + xy[:, b2]) / 3, np.where(pick, s1, s2)))
    if not omni:
        return out
    ang = np.deg2rad(np.arange(0, 180, cfg['omni_angle_step_deg']))
    cos_a = np.cos(ang).astype(np.float32)[None, :, None]
    sin_a = np.sin(ang).astype(np.float32)[None, :, None]
    for v, hx, hy, sx, sy in omni_cliques():
        bx = (sx * (df[:, hx] - df[:, v])).astype(np.float32)
        by = (sy * (df[:, hy] - df[:, v])).astype(np.float32)
        proj = cos_a * bx[:, None] + sin_a * by[:, None]             # (P, angles, T)
        k = np.argmax(np.ptp(proj, axis=-1), axis=1)
        out.append(('omni', 'E{}-E{}-E{}'.format(v, hx, hy),
                    (xy[:, v] + xy[:, hx] + xy[:, hy]) / 3, proj[np.arange(len(k)), k]))
    return out


def woi_slice(t, woi):
    if woi is None:
        return slice(None)
    i = np.flatnonzero((t >= woi[0]) & (t <= woi[1]))
    return slice(i[0], i[-1] + 1)


def process_run(run, cfg, gain, sd, rng):
    fs = 1000.0 / float(np.median(np.diff(run['t'])))
    F = cfg['filters']
    uni = gain * run['uni']
    if sd > 0:
        uni = uni + rng.normal(0.0, sd, uni.shape)
    uf = apply_filter(make_filter(F['unipolar_hz'], fs, F['order']), uni, F['zero_phase'])
    df = apply_filter(make_filter(F['differential_hz'], fs, F['order']), uni, F['zero_phase'])
    import pandas as pd
    w = woi_slice(run['t'], cfg['woi_ms'])
    parts = []
    for kind, label, pos, tr in recordings(uf[..., w], df[..., w], run['xy'], cfg):
        parts.append(pd.DataFrame({'placement': np.arange(len(pos)), 'kind': kind, 'label': label,
                                   'x_mm': pos[:, 0], 'y_mm': pos[:, 1], 'd_mm': pos[:, 0],
                                   'PP_mV': np.ptp(tr, axis=-1)}))
    out = pd.concat(parts, ignore_index=True)
    for k, v in run['meta'].items():
        out.insert(0, k, v)
    return out


# ---------------- calibration ----------------
def median_bipolar_pp(runs, cfg, unit_noise=False, rng=None):
    """Median filtered bipolar peak-to-peak: of the signal (pre runs) or of unit-SD noise."""
    vals = []
    for run in runs:
        fs = 1000.0 / float(np.median(np.diff(run['t'])))
        x = rng.normal(0.0, 1.0, run['uni'].shape) if unit_noise else run['uni']
        df = apply_filter(make_filter(cfg['filters']['differential_hz'], fs, cfg['filters']['order']),
                          x, cfg['filters']['zero_phase'])[..., woi_slice(run['t'], cfg['woi_ms'])]
        for kind, _, _, tr in recordings(df, df, run['xy'], cfg, omni=False):
            if kind == 'bi':
                vals.append(np.ptp(tr, axis=-1))
        if unit_noise:
            break                                  # one run is plenty for the noise statistic
    return float(np.median(np.concatenate(vals)))


def calibrate(runs, cfg):
    g = cfg['gain']
    if g['mode'] == 'fixed':
        gain = float(g['value'])
    else:
        pre = [r for r in runs if r['meta']['state'] == 'pre']
        if not pre:
            sys.exit('gain.mode = auto needs pre runs')
        gain = g['target_bipolar_healthy_mV'] / median_bipolar_pp(pre, cfg)
    n = cfg['noise']
    if n['mode'] == 'sd':
        sd = float(n['sd_mV'])
    elif n['mode'] == 'none':
        sd = 0.0
    else:
        unit = median_bipolar_pp(runs, cfg, unit_noise=True, rng=np.random.default_rng(12345))
        sd = n['floor_bipolar_mV'] / unit
    return gain, sd


# ---------------- export ----------------
def export_wang(df, cfg, out):
    import pandas as pd
    W = cfg['wang_export']
    pats = {v: i + 1 for i, v in enumerate(sorted(df[W['patient_field']].unique()))}
    loc_map = {str(k): v for k, v in W['location_map'].items()}
    wang = pd.DataFrame({
        'patientID': df[W['patient_field']].map(pats),
        'inf0_roof1': df[W['location_field']].astype(str).map(loc_map),
        'uni1_bi2_omni3': df['kind'].map({'uni': 1, 'bi': 2, 'omni': 3}),
        'DistAcrossPath': df['d_mm'] - W['boundary_offset_mm'],
        'PP': df['PP_mV'],
        'PFreq': np.nan,
        'state': df['state']})
    if wang['inf0_roof1'].isna().any():
        sys.exit('location_map does not cover all values of ' + W['location_field'])
    post, pre = wang[wang.state == 'post'].drop(columns='state'), wang[wang.state == 'pre'].drop(columns='state')
    post.to_excel(os.path.join(out, 'merged_rectangle_summary_euclidean.xlsx'), index=False)
    pre.to_excel(os.path.join(out, 'merged_rectangle_summary_euclidean_pre.xlsx'), index=False)
    # Flat sheet: geodesic = Euclidean. Placeholder validation files so the Rmd runs unchanged.
    import shutil
    shutil.copy(os.path.join(out, 'merged_rectangle_summary_euclidean.xlsx'),
                os.path.join(out, 'merged_rectangle_summary_geodesic.xlsx'))
    v = np.linspace(-2.5, 9.5, 50)
    np.savetxt(os.path.join(out, 'auto_y.txt'), v, fmt='%.3f')
    np.savetxt(os.path.join(out, 'manual_y.txt'), np.round(v), fmt='%.0f')
    with open(os.path.join(out, 'patient_map.json'), 'w') as f:
        json.dump({str(k): v for k, v in pats.items()}, f, indent=1)


def run_config(cfg):
    import pandas as pd
    runs = load_runs(cfg)
    gain, sd = calibrate(runs, cfg)
    out = os.path.join(REPO, cfg['out_dir'], cfg['name'])
    os.makedirs(out, exist_ok=True)
    print('[{}] {} runs | gain {:.4g} | noise SD {:.4g} mV'.format(cfg['name'], len(runs), gain, sd))

    parts = []
    for run in runs:
        seed = int(hashlib.md5('{}|{}'.format(cfg['seed'], run['meta']['run']).encode()).hexdigest()[:8], 16)
        parts.append(process_run(run, cfg, gain, sd, np.random.default_rng(seed)))
    df = pd.concat(parts, ignore_index=True)
    df.to_csv(os.path.join(out, 'points.csv'), index=False)

    healthy = df[(df.state == 'pre')].groupby('kind').PP_mV.median().to_dict()
    summary = dict(config=cfg, gain=gain, noise_sd_mV=sd, n_points=len(df),
                   pre_median_PP_mV=healthy,
                   wang_table2_inferior_healthy_mV={'uni': 1.83, 'bi': 1.44, 'omni': 2.05})
    with open(os.path.join(out, 'run_summary.json'), 'w') as f:
        json.dump(summary, f, indent=1, default=str)
    if cfg['wang_export']['enabled']:
        export_wang(df, cfg, out)
    print('[{}] {} points -> {}  | pre medians (mV): {}'.format(
        cfg['name'], len(df), out, {k: round(v, 3) for k, v in healthy.items()}))


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('config', nargs='+', help='JSON config file(s)')
    a = p.parse_args()
    for path in a.config:
        cfg = deep_update(DEFAULTS, json.load(open(path)))
        for c in expand_sweep(cfg):
            run_config(c)


if __name__ == '__main__':
    main()