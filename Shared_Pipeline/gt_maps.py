#!/usr/bin/env python3
"""
gt_maps - ground-truth contribution maps: which tissue each recording actually sees.

Uses ground_truth.py's lead fields and filters. For a recording with lead-field gradient g,
node k contributes c_k(t) = g_k . S_k(t); the map is the per-node contribution energy
sum_t c_k(t)^2, as a share of the recording's total.
One central recording per kind (uni / bi / omni) for each placement at the map centres.

Usage:
    python3 gt_maps.py config.json <out dir> <run folder> [<run folder> ...]
Writes to <out dir>:
    <run>_maps.png          top view (summed through the wall for a slab)
    <run>_maps_summary.csv  per recording: fov80, region shares, thr80 (see below)
    <run>_maps.npz          the plotted maps
  3D runs also:
    <run>_maps_depth.png    side view, x-z section summed across y
    <run>_p<P>.vtk          3D volume per placement (ParaView): point arrays uni, bi, omni
                            (share per node) and region (main mesh tag per node). The tissue
                            holding 80% of a recording's energy is where its array >= thr80.
    <run>_p<P>_electrodes.vtk  the placement's 16 electrode centres

Settings (optional, in the config's "groundtruth" block):
    "map_xc_mm_post": catheter centres x (y = 0) for post runs (default -8, -4, -2, 0, 2, 4, 8)
    "map_xc_mm_pre":  catheter centres x (y = 0) for pre runs  (default 0)
    "map_xy_mm_post" / "map_xy_mm_pre": [[x, y], ...] instead of the above
    "map_lim_mm":     half-width of the plotted window (default 14)
    "window_last_ms": as in ground_truth.py
"""
import json
import os
import sys

import numpy as np

import egm as ep
import ground_truth as gt

MAP_DEFAULTS = {"map_xc_mm_post": [-8.0, -4.0, -2.0, 0.0, 2.0, 4.0, 8.0],
                "map_xc_mm_pre": [0.0],
                "map_lim_mm": 14.0}
KINDS = {'uni': 'Unipolar', 'bi': 'Bipolar', 'omni': 'Omnipolar'}


def process_run(folder, cfg, mp, out):
    T = gt.load_tissue(folder, cfg, mp)
    meta, z, xyz, h, regions = T['meta'], T['z'], T['xyz'], T['h'], T['regions']

    # grid layout of xyz (source_density orders nodes x fastest, then y, then z)
    xs, ys, zs = np.unique(xyz[:, 0]), np.unique(xyz[:, 1]), np.unique(xyz[:, 2])
    grid = (len(zs), len(ys), len(xs))
    ids = np.array(sorted(regions))
    main = ids[np.argmax(np.stack([regions[t] for t in ids]), axis=0)]   # each node's main region
    tag_map = main.reshape(grid).max(axis=0)                     # regions seen from above
    tag_xz = main.reshape(grid).max(axis=1)                      # regions seen from the side
    pmeta = {k[2:]: z[k] for k in z.files if k.startswith('p_')}
    top = xyz[:, 2].max()
    slab = len(zs) > 1
    base = os.path.join(out, meta['run'].replace('/', '_'))

    maps, maps_xz, rows = {}, {}, []
    for p in gt.placements(z, mp, meta['state'], key='map'):
        sites, samples = gt.electrode_sites(z, p, top)
        centre = sites[:, :2].mean(axis=0)
        recs = gt.recording_leadfields(sites, cfg, T['Sd'], T['comps'], xyz, samples)
        volume = {}
        for kind in KINDS:
            # the recording of this kind closest to the catheter centre
            _, label, site, g = min((r for r in recs if r[0] == kind),
                                    key=lambda r: np.hypot(*(r[2][:2] - centre)))
            X, E = (T['Su'], T['E']['u']) if kind == 'uni' else (T['Sd'], T['E']['d'])
            fr, fn, _ = gt.fov_metrics(g, X, E, site, xyz, h, mp['fraction'])
            e = gt.node_energy(g, E)
            share = (e / e.sum() if e.sum() > 0 else e).astype(np.float32)
            vol = share.reshape(grid)
            key = '{}_p{}'.format(kind, p)
            maps[key], maps_xz[key], volume[kind] = vol.sum(axis=0), vol.sum(axis=1), share
            row = dict(run=meta['run'], state=meta['state'], placement=p,
                       **{k: v[p].item() for k, v in pmeta.items()},
                       centre_x_mm=float(centre[0]), centre_y_mm=float(centre[1]), kind=kind,
                       label=label, site_x=float(site[0]), site_y=float(site[1]),
                       site_z=float(site[2]), fov80_res_mm=fr, fov80_node_mm=fn,
                       thr80=threshold(share, 0.8))
            row.update(gt.region_shares(e, regions))
            rows.append(row)
        if slab:
            write_vtk('{}_p{}.vtk'.format(base, p), xs, ys, zs, dict(volume, region=main.astype(np.float32)))
            write_points_vtk('{}_p{}_electrodes.vtk'.format(base, p), sites[:, :3])

    import pandas as pd
    pd.DataFrame(rows).to_csv(base + '_maps_summary.csv', index=False)
    np.savez_compressed(base + '_maps.npz', x_mm=xs, y_mm=ys, z_mm=zs, tag_map=tag_map, tag_xz=tag_xz,
                        **maps, **{k + '_xz': v for k, v in maps_xz.items()})
    lim = mp['map_lim_mm']
    plot(base + '_maps.png', meta, rows, maps, xs, ys, tag_map, z, pmeta,
         view=dict(v='y', xlim=lambda r: (-lim, lim), ylim=(-lim, lim), circle=True,
                   title='top view (summed through the wall)'))
    if slab:
        plot(base + '_maps_depth.png', meta, rows, maps_xz, xs, zs, tag_xz, z, pmeta,
             view=dict(v='z', xlim=lambda r: (r['centre_x_mm'] - 8, r['centre_x_mm'] + 8),
                       ylim=(zs[0], z['uni_xy'][..., 2].max() + 0.75 if z['uni_xy'].shape[-1] > 2 else top + 1),
                       circle=False, surface=top, title='side view, x-z (summed across y)'))
    print('{}: maps for {} placements -> {}_maps*'.format(meta['run'], len(rows) // 3, base), flush=True)


def threshold(share, frac):
    """Share value above which the nodes hold frac of the energy (the 80% region's edge)."""
    s = np.sort(share.ravel())[::-1]
    return float(s[min(np.searchsorted(np.cumsum(s), frac), s.size - 1)])


def write_vtk(path, xs, ys, zs, arrays):
    """Legacy binary VTK structured points (x fastest, as the node grid), one point array per key."""
    with open(path, 'wb') as f:
        f.write('# vtk DataFile Version 3.0\nground-truth contribution share\nBINARY\n'
                'DATASET STRUCTURED_POINTS\nDIMENSIONS {} {} {}\nORIGIN {} {} {}\n'
                'SPACING {} {} {}\nPOINT_DATA {}\n'.format(
                    len(xs), len(ys), len(zs), xs[0], ys[0], zs[0], xs[1] - xs[0], ys[1] - ys[0],
                    zs[1] - zs[0], len(xs) * len(ys) * len(zs)).encode())
        for name, a in arrays.items():
            f.write('SCALARS {} float 1\nLOOKUP_TABLE default\n'.format(name).encode())
            f.write(np.asarray(a, '>f4').tobytes() + b'\n')


def write_points_vtk(path, pts):
    """Legacy ASCII VTK point cloud (electrode centres, mm) as vertex cells."""
    with open(path, 'w') as f:
        f.write('# vtk DataFile Version 3.0\nelectrodes\nASCII\nDATASET UNSTRUCTURED_GRID\n'
                'POINTS {} float\n'.format(len(pts)))
        f.write('\n'.join('{:.4f} {:.4f} {:.4f}'.format(*q) for q in pts) + '\n')
        f.write('CELLS {} {}\n'.format(len(pts), 2 * len(pts)))
        f.write('\n'.join('1 {}'.format(i) for i in range(len(pts))) + '\n')
        f.write('CELL_TYPES {}\n'.format(len(pts)) + '1\n' * len(pts))       # 1 = VTK_VERTEX


def plot(path, meta, rows, maps, xs, ys, tag_map, z, pmeta, view):
    """
    Rows = recording kind, columns = placements. view: v ('y' top view or 'z' side view),
    xlim (per row of the summary), ylim, circle (draw fov80), surface (tissue top, side view).
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm

    ps = sorted({r['placement'] for r in rows})
    base_tag = np.bincount(tag_map.ravel()).argmax()                # the bulk tissue
    plt.rcParams.update({'font.size': 8, 'axes.edgecolor': '#9a9a9a', 'axes.linewidth': 0.6,
                         'xtick.color': '#555555', 'ytick.color': '#555555'})
    side = view['v'] == 'z'
    height = 4.2 if side else 6.9
    fig, axes = plt.subplots(3, len(ps), figsize=(2.05 * len(ps) + 1.4, height), squeeze=False,
                             sharex=not side, sharey=True, constrained_layout=True)
    im = None
    for j, p in enumerate(ps):
        for i, kind in enumerate(KINDS):
            ax = axes[i, j]
            r = next(r for r in rows if r['placement'] == p and r['kind'] == kind)
            m = maps['{}_p{}'.format(kind, p)]
            if m.max() > 0:
                im = ax.pcolormesh(xs, ys, np.maximum(m, 1e-7), norm=LogNorm(1e-6, m.max()),
                                   cmap='Blues', shading='nearest', rasterized=True)
                s = np.sort(m.ravel())[::-1]                    # smallest area holding 80%
                thr = s[min(np.searchsorted(np.cumsum(s), 0.8), s.size - 1)]
                ax.contour(xs, ys, m, levels=[thr], colors='#1a1a1a', linewidths=1.0)
            for tg in np.unique(tag_map):                       # mesh regions, outlined
                if tg != base_tag:
                    ax.contour(xs, ys, (tag_map == tg).astype(float), levels=[0.5],
                               colors='#c0392b', linewidths=0.6, linestyles='--')
            if view['circle'] and np.isfinite(r['fov80_res_mm']):
                ax.add_patch(plt.Circle((r['site_x'], r['site_y']), r['fov80_res_mm'], fill=False,
                                        ec='#e67e22', lw=1.0, ls=':'))
            if side:
                ax.axhline(view['surface'], color='#888888', lw=0.6)   # endocardium
            ev = z['uni_xy'][p][:, 2] if side else z['uni_xy'][p][:, 1]
            ax.plot(z['uni_xy'][p][:, 0], ev, 'o', ms=2.2, mfc='white', mec='#333333', mew=0.5)
            ax.plot(r['site_x'], r['site_z'] if side else r['site_y'], 'x', color='#e67e22', ms=5, mew=1.2)
            ax.set_xlim(*view['xlim'](r)); ax.set_ylim(*view['ylim']); ax.set_aspect('equal')
            shares = sorted(((v, k[9:]) for k, v in r.items() if k.startswith('share_tag')), reverse=True)
            txt = ('' if side else 'fov80 {:.1f} mm\n'.format(r['fov80_res_mm'])) + \
                  '  '.join('tag {} {:.0%}'.format(t, v) for v, t in shares[:3] if v >= 0.005)
            ax.text(0.03, 0.04 if side else 0.97, txt, transform=ax.transAxes,  # side: clear of electrodes
                    va='bottom' if side else 'top', fontsize=6.5, color='#222222',
                    bbox=dict(fc='white', ec='none', alpha=0.75, pad=1.5))
            if i == 0:
                extra = ''.join(', {} {}'.format(k, pmeta[k][p]) for k in pmeta)
                ax.set_title('centre ({:g}, {:g}) mm{}'.format(r['centre_x_mm'], r['centre_y_mm'], extra),
                             fontsize=7.5)
            if j == 0:
                ax.set_ylabel('{}\n{} (mm)'.format(KINDS[kind], view['v']))
            if i == 2:
                ax.set_xlabel('x (mm)')
    if im is not None:
        cb = fig.colorbar(im, ax=axes, shrink=0.6, pad=0.01)
        cb.set_label("share of recording's contribution energy per node (log)")
    fig.suptitle('Ground-truth contributing tissue - {} - {}\nblack contour: tissue holding 80% of '
                 'contribution energy{} | red dashed: mesh regions (tags other than {})'.format(
                     meta['run'], view['title'], ' | orange circle: fov80' if view['circle'] else '',
                     base_tag), fontsize=8.5)
    fig.savefig(path, dpi=200)
    plt.close(fig)


def main():
    if len(sys.argv) < 4:
        sys.exit(__doc__)
    cfg = ep.deep_update(ep.DEFAULTS, json.load(open(sys.argv[1])))
    mp = {**gt.GT_DEFAULTS, **MAP_DEFAULTS, **cfg.get('groundtruth', {})}
    out = sys.argv[2]
    os.makedirs(out, exist_ok=True)
    for folder in sys.argv[3:]:
        process_run(folder.rstrip('/'), cfg, mp, out)


if __name__ == '__main__':
    main()
