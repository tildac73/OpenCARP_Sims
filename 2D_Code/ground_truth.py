#!/usr/bin/env python3
"""
groundtruth - lead-field ground truth: which tissue actually contributes to each recording.

Same physics as the electrograms (openCARP phie recovery, infinite homogeneous medium):
    phi(t) = sum_k c_k(t),   c_k(t) = S_k(t) . grad Z(x_k)
    S_k    = sigma_i,k * w_k * grad Vm_k(t)          (w_k = node area/volume)
    Z(x)   = 1 / (4 pi sigma_b |r - x|)              (lead field of a point electrode)
Bipoles and omnipoles are the same linear combinations of electrode lead fields that
egmpipe uses, with the same filters, so ground truth and the Wang-format points match.

Per recording it reports
    fov80_res_mm   smallest radius R (in the tissue plane, around the recording site) such
                   that rebuilding the signal from tissue within R leaves <= 20% of the
                   signal energy unexplained, for every larger R
    fov80_node_mm  radius containing 80% of the summed per-node contribution energy
    r_vs_recovery  (unipolar only) correlation of sum_k c_k with openCARP's recovered phi
                   at the same point - consistency check, should be ~1
    scale_vs_recovery  (unipolar only) amplitude ratio; should be the same for every
                   electrode (absolute scale differs only by sigma_b / thickness)

Usage:
    python3 groundtruth.py configs/clin_default.json
Writes results/<name>/groundtruth.csv and, if points.csv exists, points_with_gt.csv.

Settings (optional "groundtruth" block in the config):
    "placements_xc_mm_pre":  catheter centres for pre runs  (default [0]; translation-invariant)
    "placements_xc_mm_post": catheter centres for post runs (default -10..15 step 2.5)
    "fraction": 0.8
"""
import argparse
import csv
import glob
import json
import os
import sys

import numpy as np
from scipy import sparse

import egm as ep

GT_DEFAULTS = {"placements_xc_mm_pre": [0.0],
               "placements_xc_mm_post": list(np.arange(-10.0, 15.0 + 1e-9, 2.5)),
               "fraction": 0.8}


# ---------------- inputs ----------------
def read_igb(path):
    """openCARP .igb -> (T, N) float32."""
    with open(path, 'rb') as f:
        raw = f.read()
    kv = dict(t.split(':', 1) for t in raw[:1024].decode('latin-1').split() if ':' in t)
    n = int(kv['x']) * int(kv.get('y', 1)) * int(kv.get('z', 1))
    nt = int(kv['t'])
    end = '>' if 'big' in kv.get('systeme', 'little') else '<'
    return np.frombuffer(raw[len(raw) - 4 * n * nt:], dtype=end + 'f4').reshape(nt, n)


def run_conductivity(folder, meta):
    """gi (S/m) and lesion description, from meta.json or tuned_cv.csv + folder name."""
    gi = meta.get('gi')
    if gi is None:
        with open(os.path.join(ep.REPO, 'tuned_cv.csv')) as f:
            for row in csv.DictReader(f):
                if abs(float(row['target_cv']) - meta['cv']) < 1e-9:
                    gi = float(row['gi'])
    if gi is None:
        sys.exit('No gi for ' + folder)
    m = ep.RUN_RE.search(meta['run'] + '_')
    lw = meta.get('lesion_width', float(m.group(2)) if m and m.group(2) else None)
    lg = meta.get('lesion_g', float(m.group(3)) if m and m.group(3) else None)
    return gi, lw, lg


def source_density(folder, meta):
    """S (ncomp, N, T) = sigma_i * w * grad Vm on the regular node grid, and node xyz (N, 3) in mm."""
    with open(os.path.join(folder, 'meshname.txt')) as f:
        pts = np.loadtxt(f.read().strip() + '.pts', skiprows=1) / 1000.0
    vm = read_igb(glob.glob(os.path.join(folder, 'vm.igb'))[0])
    gi, lw, lg = run_conductivity(folder, meta)

    axes = [a for a in range(3) if np.ptp(pts[:, a]) > 0]          # 2 for a sheet, 3 for a slab
    h = min(np.min(np.diff(np.unique(pts[:, a]))) for a in axes)
    idx = [np.rint((pts[:, a] - pts[:, a].min()) / h).astype(int) for a in axes]
    shape = [i.max() + 1 for i in idx]
    if np.prod(shape) != len(pts):
        sys.exit('Ground truth needs a regular grid mesh (carputils Block)')
    flat = np.ravel_multi_index(idx[::-1], shape[::-1])             # last axis fastest
    order = np.empty(len(pts), int)
    order[flat] = np.arange(len(pts))
    xyz = pts[order]

    T = vm.shape[0]
    v = vm[:, order].reshape([T] + shape[::-1])
    grads = np.gradient(v, h, axis=tuple(range(1, len(shape) + 1)))  # in reversed-axis order
    grads = grads[::-1] if len(axes) > 1 else [grads]

    sig = np.full(len(xyz), gi)
    if meta.get('state') == 'post' and lw is not None:
        x = xyz[:, 0]
        sig[(x > 1e-6) & (x < lw - 1e-6)] = gi * lg
    w = np.full(shape[::-1], h ** len(axes))                         # trapezoid weights
    for ax in range(len(shape)):
        sl = [slice(None)] * len(shape)
        for end in (0, -1):
            sl[ax] = end
            w[tuple(sl)] *= 0.5
    w = (sig * w.ravel()).astype(np.float32)

    S = np.stack([(g.reshape(T, -1) * w).T for g in grads]).astype(np.float32)  # (ncomp, N, T)
    comps = axes
    return S, xyz, comps, h


# ---------------- lead fields ----------------
def grad_Z(site, xyz, comps):
    """grad_x (1/|r - x|) at every node for a point electrode at site (3,), -> (ncomp, N)."""
    d = site[None, :] - xyz
    inv_r3 = np.sum(d * d, axis=1) ** -1.5
    return np.stack([d[:, a] * inv_r3 for a in comps]).astype(np.float32)


def recording_leadfields(sites, cfg, Sf, comps, xyz):
    """
    Lead-field gradients (ncomp, N) and site positions for every recording of one placement,
    in egmpipe's order and with egmpipe's labels. Sf = differentially filtered S (for the
    max_orthogonal and omnipolar choices, made on the noise-free filtered signal).
    """
    G = [grad_Z(s, xyz, comps) for s in sites]                      # 16 electrodes
    sig = lambda g: np.einsum('cn,cnt->t', g, Sf)                   # noqa: E731
    out = [('uni', 'E{}'.format(e), sites[e], G[e]) for e in range(16)]
    for a, b, b2 in ep.bipolar_pairs(cfg['bipolar_mode']):
        g1 = G[b] - G[a]
        if b2 is None:
            out.append(('bi', 'E{}-E{}'.format(a, b), (sites[a] + sites[b]) / 2, g1))
        else:
            g2 = G[b2] - G[a]
            g = g1 if np.ptp(sig(g1)) >= np.ptp(sig(g2)) else g2
            out.append(('bi', 'E{}-E{}|E{}'.format(a, b, b2), (sites[a] + sites[b] + sites[b2]) / 3, g))
    ang = np.deg2rad(np.arange(0, 180, cfg['omni_angle_step_deg']))
    for v, hx, hy, sx, sy in ep.omni_cliques():
        gx, gy = sx * (G[hx] - G[v]), sy * (G[hy] - G[v])
        bx, by = sig(gx), sig(gy)
        k = int(np.argmax([np.ptp(np.cos(a) * bx + np.sin(a) * by) for a in ang]))
        out.append(('omni', 'E{}-E{}-E{}'.format(v, hx, hy), (sites[v] + sites[hx] + sites[hy]) / 3,
                    np.cos(ang[k]) * gx + np.sin(ang[k]) * gy))
    return out


def fov_metrics(g, S, energy_terms, site, xyz, h, frac):
    """fov80 by signal reconstruction and by per-node energy, around site in the tissue plane."""
    rho = np.hypot(xyz[:, 0] - site[0], xyz[:, 1] - site[1])
    b = np.minimum((rho / h).astype(int), int(rho.max() / h))
    nb = b.max() + 1
    full = np.einsum('cn,cnt->t', g, S)
    total = float(np.sum(full ** 2))
    if total == 0:
        return np.nan, np.nan, full
    rows = np.arange(len(b))
    ring = sum(sparse.csr_matrix((g[c], (b, rows)), shape=(nb, len(b))) @ S[c] for c in range(len(g)))
    partial = np.cumsum(np.asarray(ring), axis=0)                   # (nb, T) signal from rho <= R
    resid = np.sum((full[None, :] - partial) ** 2, axis=1) / total
    above = np.flatnonzero(resid > 1 - frac)
    fov_res = (above[-1] + 2) * h if above.size else h

    Sxx = energy_terms                                              # (ncomp, ncomp, N)
    e = np.einsum('cn,dn,cdn->n', g, g, Sxx)
    ce = np.cumsum(np.bincount(b, weights=e, minlength=nb))
    fov_node = (np.searchsorted(ce, frac * ce[-1]) + 1) * h
    return float(fov_res), float(fov_node), full


# ---------------- per run ----------------
def process_run(folder, cfg, gt):
    meta = ep.run_meta(folder)
    z = np.load(os.path.join(folder, 'egm.npz'))
    standoff = float(z['standoff_mm']) if 'standoff_mm' in z.files else 0.5
    t = z['t_ms']
    fs = 1000.0 / float(np.median(np.diff(t)))
    S, xyz, comps, h = source_density(folder, meta)
    top = xyz[:, 2].max()

    F = cfg['filters']
    w = ep.woi_slice(t, cfg['woi_ms'])
    Sd = ep.apply_filter(ep.make_filter(F['differential_hz'], fs, F['order']), S, F['zero_phase'])[..., w]
    Su = ep.apply_filter(ep.make_filter(F['unipolar_hz'], fs, F['order']), S, F['zero_phase'])[..., w]
    En = {k: np.einsum('cnt,dnt->cdn', X, X) for k, X in (('d', Sd), ('u', Su))}

    xc_list = gt['placements_xc_mm_pre'] if meta['state'] == 'pre' else gt['placements_xc_mm_post']
    rows = []
    for xc in xc_list:
        p = int(np.argmin(np.abs(z['xc_mm'] - xc)))
        sites = np.c_[z['uni_xy'][p][:, :2], np.full(16, top + standoff)]
        for kind, label, site, g in recording_leadfields(sites, cfg, Sd, comps, xyz):
            X, E = (Su, En['u']) if kind == 'uni' else (Sd, En['d'])
            fr, fn, full = fov_metrics(g, X, E, site, xyz, h, gt['fraction'])
            row = dict(meta, placement=p, xc_mm=float(z['xc_mm'][p]), kind=kind, label=label,
                       x_mm=float(site[0]), y_mm=float(site[1]), d_mm=float(site[0]),
                       fov80_res_mm=fr, fov80_node_mm=fn)
            if kind == 'uni':                                       # check against openCARP recovery
                e = int(label[1:])
                rec = z['uni'][p, e].astype(np.float64)
                rec = ep.apply_filter(ep.make_filter(F['unipolar_hz'], fs, F['order']), rec, F['zero_phase'])[w]
                mine = full - full.mean()
                rec = rec - rec.mean()
                row['r_vs_recovery'] = float(np.corrcoef(mine, rec)[0, 1])
                row['scale_vs_recovery'] = float(np.ptp(rec) / np.ptp(mine)) if np.ptp(mine) > 0 else np.nan
            rows.append(row)
    print('{}: {} recordings'.format(meta['run'], len(rows)), flush=True)
    return rows


def main():
    import pandas as pd
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('config', help='same JSON config as egmpipe')
    a = p.parse_args()
    cfg = ep.deep_update(ep.DEFAULTS, json.load(open(a.config)))
    gt = dict(GT_DEFAULTS, **cfg.get('groundtruth', {}))
    folders = sorted(f for f in glob.glob(os.path.join(ep.REPO, cfg['runs']))
                     if os.path.exists(os.path.join(f, 'egm.npz')) and glob.glob(os.path.join(f, 'vm.igb')))
    if not folders:
        sys.exit('No run folders with egm.npz and vm.igb match ' + cfg['runs'])

    rows = []
    for f in folders:
        rows += process_run(f, cfg, gt)
    df = pd.DataFrame(rows)
    out = os.path.join(ep.REPO, cfg['out_dir'], cfg['name'])
    os.makedirs(out, exist_ok=True)
    df.to_csv(os.path.join(out, 'groundtruth.csv'), index=False)

    u = df[df.kind == 'uni']
    print('Consistency with openCARP recovery: r median {:.4f} (min {:.4f}); scale CV {:.2%}'.format(
        u.r_vs_recovery.median(), u.r_vs_recovery.min(), u.scale_vs_recovery.std() / u.scale_vs_recovery.mean()))
    print(df.groupby(['state', 'kind'])[['fov80_res_mm', 'fov80_node_mm']].median().round(2).to_string())

    pts = os.path.join(out, 'points.csv')
    if os.path.exists(pts):
        keys = ['run', 'placement', 'kind', 'label']
        merged = pd.read_csv(pts).merge(df[keys + ['fov80_res_mm', 'fov80_node_mm']], on=keys, how='left')
        merged.to_csv(os.path.join(out, 'points_with_gt.csv'), index=False)
    print('-> ' + out)


if __name__ == '__main__':
    main()