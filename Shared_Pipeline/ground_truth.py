#!/usr/bin/env python3
"""
groundtruth - lead-field ground truth: which tissue actually contributes to each recording.

Same physics as the electrograms (openCARP phie recovery, infinite homogeneous medium):
    phi(t) = sum_k c_k(t),   c_k(t) = S_k(t) . grad Z(x_k)
    S_k    = sigma_i,k * w_k * grad Vm_k(t)          (w_k = node area/volume)
    Z(x)   = 1 / (4 pi sigma_b |r - x|)              (lead field of a point electrode)
Bipoles and omnipoles are the same linear combinations of electrode lead fields that
egmpipe uses, with the same filters, so ground truth and the Wang-format points match.
Finite electrodes (egm.npz uni_sample_xyz) use the mean lead field of their sample points.

Works for any regular-grid run (2D sheet or 3D slab): sigma_i per node comes from the
run's own parameters.par (gregion conductivities) and the mesh .elem tags, with fibres
along x (l = x, t = y, n = z).

Per recording it reports
    fov80_res_mm   smallest radius R (in the tissue plane, around the recording site) such
                   that rebuilding the signal from tissue within R leaves <= 20% of the
                   signal energy unexplained, for every larger R
    fov80_node_mm  radius containing 80% of the summed per-node contribution energy
    share_tag<k>   share of the contribution energy from tissue with mesh tag k (a node on a
                   region boundary is split by the conductivity each region gives it)
    r_vs_recovery  (unipolar only) correlation of sum_k c_k with openCARP's recovered phi
                   at the same point - consistency check, should be ~1
    scale_vs_recovery  (unipolar only) amplitude ratio; should be the same for every
                   electrode (absolute scale differs only by sigma_b / thickness)

Usage:
    python3 ground_truth.py configs/clin_default.json
Writes results/<name>/groundtruth.csv and, if points.csv exists, points_with_gt.csv.

Settings (optional "groundtruth" block in the config):
    "placements_xc_mm_pre":  catheter centres x (y = 0) for pre runs (default [0]; translation-invariant)
    "placements_xc_mm_post": catheter centres x (y = 0) for post runs (default -10..15 step 2.5)
    "placements_xy_mm_pre" / "placements_xy_mm_post": [[x, y], ...] instead of the above
                             (every placement at the nearest centre, e.g. both gaps, is used)
    "window_last_ms": only the last this many ms of each run (default: the whole run)
    "fraction": 0.8
"""
import argparse
import glob
import json
import os
import sys

import numpy as np
from scipy import sparse

import egm as ep

GT_DEFAULTS = {"placements_xc_mm_pre": [0.0],
               "placements_xc_mm_post": list(np.arange(-10.0, 15.0 + 1e-9, 2.5)),
               "window_last_ms": None,
               "fraction": 0.8}
G_DEFAULT = {'l': 0.174, 't': 0.019, 'n': 0.019}      # openCARP gregion g_i* defaults


# ---------------- inputs ----------------
def igb_header(path):
    """(nodes, frames, byte order) of an openCARP .igb file."""
    with open(path, 'rb') as f:
        head = f.read(1024)
    kv = dict(t.split(':', 1) for t in head.decode('latin-1').split() if ':' in t)
    n = int(kv['x']) * int(kv.get('y', 1)) * int(kv.get('z', 1))
    return n, int(kv['t']), '>' if 'big' in kv.get('systeme', 'little') else '<'


def read_igb(path, frames=slice(None)):
    """openCARP .igb -> (T, N) float32, only the requested frames read from disk."""
    n, nt, end = igb_header(path)
    data = np.memmap(path, dtype=end + 'f4', mode='r', offset=os.path.getsize(path) - 4 * n * nt,
                     shape=(nt, n))
    return np.array(data[frames], dtype=np.float32)


def read_par(path):
    """openCARP parameters.par -> {name: value string}."""
    par = {}
    for line in open(path):
        line = line.split('#', 1)[0]
        if '=' in line:
            k, v = line.split('=', 1)
            par[k.strip()] = v.strip().strip('"')
    return par


def node_tissue(folder, meshname, npts):
    """
    sigma_i (N, 3) along (x, y, z) per node and region weights {tag: (N,)}, from the run's
    parameters.par and <mesh>.elem. A node gets the mean conductivity of its elements;
    its weight for a tag is the share of that conductivity coming from the tag's elements.
    Fibres must lie along x.
    """
    import pandas as pd
    par = read_par(os.path.join(folder, 'parameters.par'))
    g_tag, g0 = {}, None
    for r in range(int(par.get('num_gregions', 1))):
        g = [float(par.get('gregion[{}].g_i{}'.format(r, a), G_DEFAULT[a])) for a in 'ltn']
        g0 = g if r == 0 else g0                      # untagged elements fall back to region 0
        for k in range(int(par.get('gregion[{}].num_IDs'.format(r), 0))):
            g_tag[int(par['gregion[{}].ID[{}]'.format(r, k)])] = g

    el = pd.read_csv(meshname + '.elem', sep=r'\s+', header=None, skiprows=1)
    nodes, tags = el.iloc[:, 1:-1].to_numpy(int), el.iloc[:, -1].to_numpy(int)
    lon = pd.read_csv(meshname + '.lon', sep=r'\s+', header=None, skiprows=1).to_numpy()
    if not np.allclose(np.abs(lon[:, 0]), 1.0, atol=1e-3):
        sys.exit('Ground truth assumes fibres along x: ' + meshname + '.lon')

    g_el = np.array([g_tag.get(t, g0) for t in np.unique(tags)])[np.searchsorted(np.unique(tags), tags)]
    flat = nodes.ravel()
    count = np.bincount(flat, minlength=npts)
    sig = np.stack([np.bincount(flat, weights=np.repeat(g_el[:, a], nodes.shape[1]), minlength=npts)
                    for a in range(3)], axis=1) / np.maximum(count, 1)[:, None]
    regions = {t: np.bincount(flat, weights=np.repeat(np.where(tags == t, g_el[:, 0], 0.0), nodes.shape[1]),
                              minlength=npts) for t in np.unique(tags)}
    total = np.maximum(sum(regions.values()), 1e-30)
    return sig.astype(np.float32), {t: (r / total).astype(np.float32) for t, r in regions.items()}


def source_density(folder, frames=slice(None)):
    """
    S (ncomp, N, T) = sigma_i * w * grad Vm on the regular node grid for the given frames,
    node xyz (N, 3) in mm, components (axes), grid spacing h, region weights {tag: (N,)}.
    """
    with open(os.path.join(folder, 'meshname.txt')) as f:
        meshname = f.read().strip()
    pts = np.loadtxt(meshname + '.pts', skiprows=1) / 1000.0
    sig, regions = node_tissue(folder, meshname, len(pts))
    vm = read_igb(glob.glob(os.path.join(folder, 'vm.igb'))[0], frames)

    axes = [a for a in range(3) if np.ptp(pts[:, a]) > 0]          # 2 for a sheet, 3 for a slab
    h = min(np.min(np.diff(np.unique(pts[:, a]))) for a in axes)
    idx = [np.rint((pts[:, a] - pts[:, a].min()) / h).astype(int) for a in axes]
    shape = [i.max() + 1 for i in idx]
    if np.prod(shape) != len(pts):
        sys.exit('Ground truth needs a regular grid mesh (carputils Block / mesher)')
    flat = np.ravel_multi_index(idx[::-1], shape[::-1])             # last axis fastest
    order = np.empty(len(pts), int)
    order[flat] = np.arange(len(pts))
    xyz, sig = pts[order], sig[order]
    regions = {t: r[order] for t, r in regions.items()}

    T = vm.shape[0]
    v = vm[:, order].reshape([T] + shape[::-1])
    del vm
    grads = np.gradient(v, h, axis=tuple(range(1, len(shape) + 1)))  # in reversed-axis order
    grads = grads[::-1] if len(axes) > 1 else [grads]
    del v

    w = np.full(shape[::-1], h ** len(axes), np.float32)            # trapezoid weights
    for ax in range(len(shape)):
        sl = [slice(None)] * len(shape)
        for end in (0, -1):
            sl[ax] = end
            w[tuple(sl)] *= 0.5
    w = w.ravel()

    S = np.empty((len(axes), len(xyz), T), np.float32)
    for c, a in enumerate(axes):
        S[c] = (grads[c].reshape(T, -1) * (sig[:, a] * w)).T
    return S, xyz, axes, h, regions


def filtered(S, band, fs, order, zero_phase, chunk=4000):
    """egmpipe's filter along time, in node chunks and float32 (a float64 copy of S is GBs)."""
    sos = ep.make_filter(band, fs, order)
    if sos is None:
        return S
    out = np.empty_like(S)
    for c in range(S.shape[0]):
        for i in range(0, S.shape[1], chunk):
            out[c, i:i + chunk] = ep.apply_filter(sos, S[c, i:i + chunk], zero_phase)
    return out


def energy_terms(X):
    """sum_t X_a X_b per node -> (ncomp, ncomp, N)."""
    n = X.shape[0]
    E = np.empty((n, n, X.shape[1]))
    for a in range(n):
        for b in range(a, n):
            E[a, b] = E[b, a] = np.einsum('nt,nt->n', X[a], X[b], dtype=np.float64)
    return E


# ---------------- placements and electrodes ----------------
def placements(z, gt, state, key='placements'):
    """Placement indices at the configured catheter centres (all placements sharing a centre)."""
    centres = z['centre_xy_mm'] if 'centre_xy_mm' in z.files else np.c_[z['xc_mm'], 0 * z['xc_mm']]
    targets = gt.get('{}_xy_mm_{}'.format(key, state)) or \
        [[xc, 0.0] for xc in gt['{}_xc_mm_{}'.format(key, state)]]
    out = []
    for target in targets:
        dist = np.hypot(*(centres - np.asarray(target, float)).T)
        out += list(np.flatnonzero(dist <= dist.min() + 1e-6))
    return sorted(set(out))


def electrode_sites(z, p, top):
    """
    (16, 4) electrode x, y, z, d in mm and their sample points (16, S, 3) or None.
    z defaults to standoff_mm above the tissue top; d to x.
    """
    xy = z['uni_xy'][p]
    if xy.shape[1] > 2:
        zc = xy[:, 2]
    else:
        zc = np.full(16, top + (float(z['standoff_mm']) if 'standoff_mm' in z.files else 0.5))
    d = z['d_mm'][p] if 'd_mm' in z.files else xy[:, 0]
    samples = z['uni_sample_xyz'][p] if 'uni_sample_xyz' in z.files else None
    return np.c_[xy[:, :2], zc, d], samples


# ---------------- lead fields ----------------
def grad_Z(site, xyz, comps):
    """grad_x (1/|r - x|) at every node for a point electrode at site (3,), -> (ncomp, N)."""
    d = site[None, :3] - xyz
    inv_r3 = np.sum(d * d, axis=1) ** -1.5
    return np.stack([d[:, a] * inv_r3 for a in comps]).astype(np.float32)


def recording_leadfields(sites, cfg, Sf, comps, xyz, samples=None):
    """
    Lead-field gradients (ncomp, N) and site positions (x, y, z, d) for every recording of
    one placement, in egmpipe's order and with egmpipe's labels. Sf = differentially filtered
    S (for the max_orthogonal and omnipolar choices, made on the noise-free filtered signal).
    samples (16, S, 3): points averaged into each electrode (finite electrodes).
    """
    if samples is None:
        G = [grad_Z(s, xyz, comps) for s in sites]                  # 16 electrodes
    else:
        G = [np.mean([grad_Z(q, xyz, comps) for q in s], axis=0) for s in samples]
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


def node_energy(g, energy_terms):
    """Per-node contribution energy sum_t c_k(t)^2 of a recording with lead-field gradient g."""
    g = g.astype(np.float64)
    return np.einsum('cn,dn,cdn->n', g, g, energy_terms)


def region_shares(e, regions):
    """{'share_tag<k>': share of the energy e (N,) from each mesh region}."""
    total = e.sum()
    return {'share_tag{}'.format(t): float(e @ r / total) if total > 0 else np.nan
            for t, r in regions.items()}


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

    e = node_energy(g, energy_terms)
    ce = np.cumsum(np.bincount(b, weights=e, minlength=nb))
    fov_node = (np.searchsorted(ce, frac * ce[-1]) + 1) * h
    return float(fov_res), float(fov_node), full


# ---------------- per run ----------------
def load_tissue(folder, cfg, gt):
    """
    Everything the lead fields need for one run, over the analysed window:
    dict(meta, z, S filtered for unipolar (Su) and differential (Sd) recordings, their
    energy terms, xyz, comps, h, region weights, frames of egm.npz analysed (egm), woi slice
    (w), fs).
    """
    meta = ep.run_meta(folder)
    z = np.load(os.path.join(folder, 'egm.npz'))
    t = z['t_ms']
    fs = 1000.0 / float(np.median(np.diff(t)))
    k = len(t) if not gt.get('window_last_ms') else min(len(t), int(round(gt['window_last_ms'] * fs / 1000)))
    nt = igb_header(glob.glob(os.path.join(folder, 'vm.igb'))[0])[1]
    S, xyz, comps, h, regions = source_density(folder, slice(nt - k, nt))   # last k frames of Vm

    F = cfg['filters']
    Sd = filtered(S, F['differential_hz'], fs, F['order'], F['zero_phase'])
    Su = filtered(S, F['unipolar_hz'], fs, F['order'], F['zero_phase'])
    del S
    w = ep.woi_slice(t[-k:], cfg['woi_ms'])
    Sd, Su = Sd[..., w], Su[..., w]
    return dict(meta=meta, z=z, fs=fs, egm=slice(len(t) - k, len(t)), w=w, Sd=Sd, Su=Su,
                E={'u': energy_terms(Su), 'd': energy_terms(Sd)},
                xyz=xyz, comps=comps, h=h, regions=regions)


def process_run(folder, cfg, gt):
    T = load_tissue(folder, cfg, gt)
    meta, z, xyz, h = T['meta'], T['z'], T['xyz'], T['h']
    F = cfg['filters']
    top = xyz[:, 2].max()
    pmeta = {k[2:]: z[k] for k in z.files if k.startswith('p_')}

    rows = []
    for p in placements(z, gt, meta['state']):
        sites, samples = electrode_sites(z, p, top)
        for kind, label, site, g in recording_leadfields(sites, cfg, T['Sd'], T['comps'], xyz, samples):
            X, E = (T['Su'], T['E']['u']) if kind == 'uni' else (T['Sd'], T['E']['d'])
            fr, fn, full = fov_metrics(g, X, E, site, xyz, h, gt['fraction'])
            row = dict(meta, **{k: v[p].item() for k, v in pmeta.items()}, placement=p, kind=kind,
                       label=label, x_mm=float(site[0]), y_mm=float(site[1]), z_mm=float(site[2]),
                       d_mm=float(site[3]), fov80_res_mm=fr, fov80_node_mm=fn)
            row.update(region_shares(node_energy(g, E), T['regions']))
            if kind == 'uni':                                       # check against openCARP recovery
                rec = z['uni'][p, int(label[1:])][T['egm']].astype(np.float64)
                rec = ep.apply_filter(ep.make_filter(F['unipolar_hz'], T['fs'], F['order']), rec,
                                      F['zero_phase'])[T['w']]
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
    print('Consistency with openCARP recovery: |r| median {:.4f} (min {:.4f}); scale CV {:.2%}'.format(
        u.r_vs_recovery.abs().median(), u.r_vs_recovery.abs().min(),
        u.scale_vs_recovery.std() / u.scale_vs_recovery.mean()))
    print(df.groupby(['state', 'kind'])[['fov80_res_mm', 'fov80_node_mm']].median().round(2).to_string())

    pts = os.path.join(out, 'points.csv')
    if os.path.exists(pts):
        keys = ['run', 'placement', 'kind', 'label']
        merged = pd.read_csv(pts).merge(df[keys + ['fov80_res_mm', 'fov80_node_mm']], on=keys, how='left')
        merged.to_csv(os.path.join(out, 'points_with_gt.csv'), index=False)
    print('-> ' + out)


if __name__ == '__main__':
    main()
