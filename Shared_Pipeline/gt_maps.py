#!/usr/bin/env python3
"""
gt_maps - ground-truth contribution maps: which tissue each recording actually sees.

Uses ground_truth.py's lead fields and filters. For a recording with lead-field gradient g,
node k contributes c_k(t) = g_k . S_k(t); the map is the per-node contribution energy
sum_t c_k(t)^2, as a share of the recording's total (summed through the wall for a slab).
One central recording per kind (uni / bi / omni) at each catheter position.

Usage:
    python3 gt_maps.py config.json <out dir> <run folder> [<run folder> ...]
Writes <out dir>/<run>_maps.npz, <run>_maps.png and <run>_maps_summary.csv.

Settings (optional, in the config's "groundtruth" block):
    "map_xc_mm_post": catheter centres for post runs (default -8, -4, -2, 0, 2, 4, 8)
    "map_xc_mm_pre":  catheter centres for pre runs  (default 0; translation-invariant)
    "map_lim_mm":     half-width of the plotted window (default 14)
"""
import json
import os
import sys

import numpy as np
from scipy.signal import sosfilt

import egm as ep
import ground_truth as gt

MAP_DEFAULTS = {"map_xc_mm_post": [-8.0, -4.0, -2.0, 0.0, 2.0, 4.0, 8.0],
                "map_xc_mm_pre": [0.0],
                "map_lim_mm": 14.0}
KINDS = {'uni': 'Unipolar', 'bi': 'Bipolar', 'omni': 'Omnipolar'}


def filtered(S, band, fs, order, zero_phase, chunk=4000):
    """Differential filter applied in node chunks, kept float32 (a full float64 copy is GBs)."""
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


def process_run(folder, cfg, mp, out):
    meta = ep.run_meta(folder)
    z = np.load(os.path.join(folder, 'egm.npz'))
    t = z['t_ms']
    fs = 1000.0 / float(np.median(np.diff(t)))
    standoff = float(z['standoff_mm']) if 'standoff_mm' in z.files else 0.5

    S, xyz, comps, h = gt.source_density(folder, meta)
    _, lw, _ = gt.run_conductivity(folder, meta)
    if meta['state'] != 'post':
        lw = None
    F = cfg['filters']
    w = ep.woi_slice(t, cfg['woi_ms'])
    Sd = filtered(S, F['differential_hz'], fs, F['order'], F['zero_phase'])[..., w]
    Su = filtered(S, F['unipolar_hz'], fs, F['order'], F['zero_phase'])[..., w]
    del S
    En = {'u': energy_terms(Su), 'd': energy_terms(Sd)}

    # grid layout of xyz (source_density orders nodes x fastest, then y, then z)
    xs, ys = np.unique(xyz[:, 0]), np.unique(xyz[:, 1])
    grid = (-1, len(ys), len(xs))
    top = xyz[:, 2].max()
    x = xyz[:, 0]

    maps, rows = {}, []
    xc_list = mp['map_xc_mm_post'] if meta['state'] == 'post' else mp['map_xc_mm_pre']
    for xc in xc_list:
        p = int(np.argmin(np.abs(z['xc_mm'] - xc)))
        sites = np.c_[z['uni_xy'][p][:, :2], np.full(16, top + standoff)]
        recs = gt.recording_leadfields(sites, cfg, Sd, comps, xyz)
        for kind in KINDS:
            # the recording of this kind closest to the catheter centre
            _, label, site, g = min((r for r in recs if r[0] == kind),
                                    key=lambda r: np.hypot(r[2][0] - z['xc_mm'][p], r[2][1]))
            X, E = (Su, En['u']) if kind == 'uni' else (Sd, En['d'])
            fr, fn, _ = gt.fov_metrics(g, X, E, site, xyz, h, cfg['groundtruth'].get('fraction', 0.8))
            g64 = g.astype(np.float64)
            e = np.einsum('cn,dn,cdn->n', g64, g64, E)
            share = e / e.sum() if e.sum() > 0 else e
            maps['{}_xc{:g}'.format(kind, z['xc_mm'][p])] = share.reshape(grid).sum(axis=0).astype(np.float32)
            row = dict(run=meta['run'], state=meta['state'], cv=meta.get('cv'),
                       stim_angle=meta.get('stim_angle'), placement=p, xc_mm=float(z['xc_mm'][p]),
                       kind=kind, label=label, site_x=float(site[0]), site_y=float(site[1]),
                       fov80_res_mm=fr, fov80_node_mm=fn)
            if lw is not None:
                row['lesion_energy_frac'] = float(share[(x > 1e-6) & (x < lw - 1e-6)].sum())
                row['iso_wall_energy_frac'] = float(share[x >= lw - 1e-6].sum())
            rows.append(row)

    import pandas as pd
    base = os.path.join(out, meta['run'] + '_maps')
    pd.DataFrame(rows).to_csv(base + '_summary.csv', index=False)
    np.savez_compressed(base + '.npz', x_mm=xs, y_mm=ys, lesion_width_mm=lw if lw else 0.0,
                        uni_xy=z['uni_xy'][..., :2], xc_mm=z['xc_mm'], **maps)
    plot(base + '.png', meta, rows, maps, xs, ys, lw, z, mp['map_lim_mm'])
    print('{}: maps for {} catheter positions -> {}'.format(meta['run'], len(xc_list), base),
          flush=True)


def plot(path, meta, rows, maps, xs, ys, lw, z, lim):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm

    xcs = sorted({r['xc_mm'] for r in rows})
    plt.rcParams.update({'font.size': 8, 'axes.edgecolor': '#9a9a9a', 'axes.linewidth': 0.6,
                         'xtick.color': '#555555', 'ytick.color': '#555555'})
    fig, axes = plt.subplots(3, len(xcs), figsize=(2.05 * len(xcs) + 1.4, 6.9), squeeze=False,
                             sharex=True, sharey=True, constrained_layout=True)
    im = None
    for j, xc in enumerate(xcs):
        p = next(r['placement'] for r in rows if r['xc_mm'] == xc)
        for i, kind in enumerate(KINDS):
            ax = axes[i, j]
            r = next(r for r in rows if r['xc_mm'] == xc and r['kind'] == kind)
            m = maps['{}_xc{:g}'.format(kind, xc)]
            if m.max() > 0:
                im = ax.pcolormesh(xs, ys, np.maximum(m, 1e-7), norm=LogNorm(1e-6, m.max()),
                                   cmap='Blues', shading='nearest', rasterized=True)
                s = np.sort(m.ravel())[::-1]                    # smallest area holding 80%
                thr = s[min(np.searchsorted(np.cumsum(s), 0.8), s.size - 1)]
                ax.contour(xs, ys, m, levels=[thr], colors='#1a1a1a', linewidths=1.0)
            if lw:
                ax.axvspan(0, lw, color='#c0392b', alpha=0.10, lw=0)
                ax.axvline(0, color='#c0392b', lw=0.8, ls='--')
            if np.isfinite(r['fov80_res_mm']):
                ax.add_patch(plt.Circle((r['site_x'], r['site_y']), r['fov80_res_mm'], fill=False,
                                        ec='#e67e22', lw=1.0, ls=':'))
            ax.plot(z['uni_xy'][p][:, 0], z['uni_xy'][p][:, 1], 'o', ms=2.2,
                    mfc='white', mec='#333333', mew=0.5)
            ax.plot(r['site_x'], r['site_y'], 'x', color='#e67e22', ms=5, mew=1.2)
            ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim); ax.set_aspect('equal')
            txt = 'fov80 {:.1f} mm'.format(r['fov80_res_mm'])
            if 'lesion_energy_frac' in r:
                txt += '\nscar {:.0%}'.format(r['lesion_energy_frac'] + r['iso_wall_energy_frac'])
            ax.text(0.03, 0.97, txt, transform=ax.transAxes, va='top', fontsize=6.5, color='#222222',
                    bbox=dict(fc='white', ec='none', alpha=0.75, pad=1.5))
            if i == 0:
                ax.set_title('catheter centre x = {:g} mm'.format(xc), fontsize=8)
            if j == 0:
                ax.set_ylabel('{}\ny (mm)'.format(KINDS[kind]))
            if i == 2:
                ax.set_xlabel('x (mm)  [scar > 0]')
    if im is not None:
        cb = fig.colorbar(im, ax=axes, shrink=0.6, pad=0.01)
        cb.set_label("share of recording's contribution energy per node (log)")
    legend = 'black contour: tissue holding 80% of contribution energy | orange circle: fov80'
    if lw:
        legend += ' | red band: lesion (0-{:g} mm)'.format(lw)
    fig.suptitle('Ground-truth contributing tissue - {}\n{}'.format(meta['run'], legend), fontsize=8.5)
    fig.savefig(path, dpi=200)
    plt.close(fig)


def main():
    if len(sys.argv) < 4:
        sys.exit(__doc__)
    cfg = ep.deep_update(ep.DEFAULTS, json.load(open(sys.argv[1])))
    cfg['groundtruth'] = dict(gt.GT_DEFAULTS, **cfg.get('groundtruth', {}))
    mp = dict(MAP_DEFAULTS, **cfg['groundtruth'])
    out = sys.argv[2]
    os.makedirs(out, exist_ok=True)
    for folder in sys.argv[3:]:
        process_run(folder.rstrip('/'), cfg, mp, out)


if __name__ == '__main__':
    main()
