#!/usr/bin/env python3
"""
thesis_figures - figures, tables and captions for the 3D slab EFOV pipeline.

Runs after egm.py and ground_truth.py (efov_3d.sh calls it last). Reads the stage
folders (vm.igb, egm.npz, meta.json, mesh) and the results folder (points.csv,
groundtruth.csv, groundtruth_curves.npz, run_summary.json), and writes to
<results>/figures/:

    NN_<name>.png / .pdf   one figure each (PNG 300 dpi for slides, PDF vector for LaTeX)
    FIGURES.md             caption and measured numbers for every figure
    figure_stats.json      every number quoted in the captions
    tables/*.csv, *.tex    summary tables (PP, field of view, region shares); electrophysiology
                           numbers (CL, CV, APD90, T_LIFT window) are in figure_stats.json
    failures.log           traceback of any figure that could not be made (others continue)
    cache/                 Vm planes and probe traces extracted once from vm.igb

Usage:
    python3 thesis_figures.py --results <results dir> --config <job config.json> <stage dirs...>
    [--only 04,09] [--no-vm] [--strict]

Vm figures describe what the simulation did; they do not decide whether reentry or
termination happened "correctly". Activation is an upward crossing of -20 mV (as
lats[0] in common.par).
"""
import argparse
import json
import os
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap, ListedColormap, LogNorm, TwoSlopeNorm  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Circle, Patch, Rectangle  # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "Shared_Pipeline"))
import egm as ep  # noqa: E402
import igb_io  # noqa: E402

# ---------------- style ----------------
INK, INK2, MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#8a8984", "#e6e5e1", "#ffffff"
# Stage colours (validated all-pairs for CVD: aqua, violet, blue, orange); colour follows the stage
STAGE_COLOURS = {
    "s1_pace": "#1baf7a",
    "s2_block": "#4a3aa7",
    "s3_reentry": "#2a78d6",
    "s4_ablate_terminate": "#eb6834",
    "post_s1": "#e87ba4",
    "post_s2": "#008300",
    "post_s3": "#e34948",
}
STAGE_LABELS = {
    "s1_pace": "s1 pacing",
    "s2_block": "s2 exit block",
    "s3_reentry": "s3 block lifted",
    "s4_ablate_terminate": "s4 post-ablation",
    "post_s1": "post s1 pacing",
    "post_s2": "post s2 exit block",
    "post_s3": "post s3 block lifted",
}
TAGS = {  # mesh tag -> (name, colour)
    1: ("wall", "#e9e8e4"),
    2: ("bundle", "#2a78d6"),
    3: ("exit segment", "#eb6834"),
    4: ("sheath", "#1baf7a"),
    5: ("lesion", "#3a3936"),
}
KINDS = {"uni": "Unipolar", "bi": "Bipolar", "omni": "Omnipolar"}
GAP_STYLE = {0.25: "-", 1.0: "--"}
BLUE_RAMP = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
SEQ = LinearSegmentedColormap.from_list("seq_blue", BLUE_RAMP)
# Vm: one-hue orange ramp (rest light, depolarised dark), separate from the magnitude blue
VM_CMAP = LinearSegmentedColormap.from_list(
    "vm_orange", ["#fdf1ea", "#f8c9ad", "#f09a6c", "#eb6834", "#c24d1c", "#8c3410", "#561c05"]
)
DIV = LinearSegmentedColormap.from_list(
    "div", ["#104281", "#3987e5", "#9ec5f4", "#f0efec", "#f2a5a4", "#e34948", "#8f1d1d"]
)
VM_LIM = (-85.0, 25.0)
THRESH_MV = -20.0
WIDTH = 6.5  # inches, thesis text width

plt.rcParams.update(
    {
        "font.size": 8,
        "axes.titlesize": 8.5,
        "axes.labelsize": 8,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        "legend.fontsize": 7,
        "legend.frameon": False,
        "axes.edgecolor": MUTED,
        "axes.labelcolor": INK,
        "axes.titlecolor": INK,
        "text.color": INK,
        "xtick.color": INK2,
        "ytick.color": INK2,
        "axes.linewidth": 0.6,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": False,
        "grid.color": GRID,
        "grid.linewidth": 0.5,
        "lines.linewidth": 1.2,
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "savefig.facecolor": SURFACE,
        "savefig.dpi": 300,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "image.interpolation": "nearest",
    }
)


def stage_colour(name):
    return STAGE_COLOURS.get(name, MUTED)


def stage_label(name):
    return STAGE_LABELS.get(name, name)


def fmt(v, digits=2):
    if v is None or (isinstance(v, float) and not np.isfinite(v)):
        return "n/a"
    return f"{v:.{digits}f}"


# ---------------- inputs ----------------
class Grid:
    """Regular node grid of a mesh, its voxel tags, and node lookup by (iz, iy, ix)."""

    def __init__(self, meshbase):
        pts = pd.read_csv(meshbase + ".pts", sep=r"\s+", skiprows=1, header=None).to_numpy() / 1000.0
        self.h = min(np.min(np.diff(np.unique(pts[:, a]))) for a in range(3) if np.ptp(pts[:, a]) > 0)
        lo = pts.min(axis=0)
        idx = np.rint((pts - lo) / self.h).astype(int)
        self.shape = tuple(idx[:, ::-1].max(axis=0) + 1)  # (nz, ny, nx)
        self.node = np.full(self.shape, -1, int)
        self.node[idx[:, 2], idx[:, 1], idx[:, 0]] = np.arange(len(pts))
        if (self.node < 0).any():
            raise ValueError(meshbase + ": figures need a regular Cartesian node grid")
        self.xs = lo[0] + self.h * np.arange(self.shape[2])
        self.ys = lo[1] + self.h * np.arange(self.shape[1])
        self.zs = lo[2] + self.h * np.arange(self.shape[0])
        self.n_nodes = len(pts)
        el = pd.read_csv(meshbase + ".elem", sep=r"\s+", skiprows=1, header=None)
        nodes = el.iloc[:, 1:-1].to_numpy(int)
        tags = el.iloc[:, -1].to_numpy(int)
        cent = pts[nodes].mean(axis=1)
        vox = np.floor((cent - lo) / self.h).astype(int)
        vshape = tuple(max(1, s - 1) for s in self.shape)
        vox = np.minimum(vox[:, ::-1], np.array(vshape) - 1)
        self.vtag = np.zeros(vshape, int)
        np.maximum.at(self.vtag, (vox[:, 0], vox[:, 1], vox[:, 2]), tags)
        self.vx = self.xs[: vshape[2]] + self.h / 2
        self.vy = self.ys[: vshape[1]] + self.h / 2
        self.vz = self.zs[: vshape[0]] + self.h / 2
        # node tag: the voxel at or below the node (adequate for region masks and outlines)
        iz = np.minimum(np.arange(self.shape[0]), vshape[0] - 1)
        iy = np.minimum(np.arange(self.shape[1]), vshape[1] - 1)
        ix = np.minimum(np.arange(self.shape[2]), vshape[2] - 1)
        self.ntag = self.vtag[np.ix_(iz, iy, ix)]

    def iz(self, z):
        return int(np.argmin(np.abs(self.zs - z)))

    def iy(self, y):
        return int(np.argmin(np.abs(self.ys - y)))

    def ix(self, x):
        return int(np.argmin(np.abs(self.xs - x)))

    def nearest(self, xyz):
        return int(self.node[self.iz(xyz[2]), self.iy(xyz[1]), self.ix(xyz[0])])

    def planes(self):
        """Node index arrays of the planes cached from Vm."""
        top = self.shape[0] - 1
        mid = self.iz(min(1.25, self.zs[-1]))
        deep = self.iz(max(-1.0, self.zs[0]))
        return {
            "top": self.node[top],
            "mid": self.node[mid],
            "deep": self.node[deep],
            "xz": self.node[:, self.iy(0.0), :],
            "yz": self.node[:, :, self.ix(0.0)],
        }

    def plane_z(self):
        return {
            "top": self.zs[-1],
            "mid": self.zs[self.iz(min(1.25, self.zs[-1]))],
            "deep": self.zs[self.iz(max(-1.0, self.zs[0]))],
        }


PROBES = [  # (key, label, (x, y, z) mm); nearest node is used
    ("pace", "Pacing site (wall)", (8.5, 0.0, 1.25)),
    ("wall_exit", "Wall beside exit", (6.5, -3.5, 1.25)),
    ("exit", "Exit segment", (6.5, 0.0, 1.25)),
    ("interface", "Bundle/exit interface", (5.25, 0.0, 1.25)),
    ("mid", "Bundle middle (lesion site)", (0.0, 0.0, 1.25)),
    ("entry", "Bundle entry", (-6.5, 0.0, 1.25)),
    ("wall_entry", "Wall beyond entry", (-9.0, 0.0, 1.25)),
    ("remote", "Remote wall", (0.0, -10.0, 1.25)),
    ("below", "Wall beneath bundle", (0.0, 0.0, -1.0)),
]


class Stage:
    def __init__(self, folder):
        self.folder = Path(folder)
        self.name = self.folder.name
        self.meta = ep.run_meta(str(self.folder))
        self.state = self.meta.get("state", "pre")
        self.meshbase = (self.folder / "meshname.txt").read_text().strip()
        with np.load(self.folder / "egm.npz") as z:
            self.t = z["t_ms"].astype(float)
        self.vm = None  # filled by load_vm

    @property
    def colour(self):
        return stage_colour(self.name)

    @property
    def label(self):
        return stage_label(self.name)


def load_vm(stage, grid, cache_dir, chunk_bytes=2e8):
    """Planes (T, ...) float16 and probe traces (T, P) float32 from vm.igb, cached."""
    vm_path = stage.folder / "vm.igb"
    cache = Path(cache_dir) / f"{stage.name}_vm_planes.npz"
    planes = grid.planes()
    probe_idx = np.array([grid.nearest(p[2]) for p in PROBES])
    if cache.exists() and (not vm_path.exists() or cache.stat().st_mtime >= vm_path.stat().st_mtime):
        with np.load(cache) as c:
            data = {k: c[k] for k in c.files}
        if data["probes"].shape[0] == len(stage.t):
            return data
    fields, nodes, count, dtype = igb_io.header(vm_path)
    if nodes != grid.n_nodes:
        raise ValueError(f"{vm_path}: {nodes} nodes, mesh has {grid.n_nodes}")
    if count != len(stage.t):
        raise ValueError(f"{vm_path}: {count} frames, egm.npz has {len(stage.t)}")
    scale, zero = float(fields.get("facteur", 1)), float(fields.get("zero", 0))
    mm = np.memmap(vm_path, dtype=dtype, mode="r", offset=1024, shape=(count, nodes))
    flat = {k: v.ravel() for k, v in planes.items()}
    out = {k: np.empty((count, v.size), np.float16) for k, v in flat.items()}
    probes = np.empty((count, len(probe_idx)), np.float32)
    step = max(1, int(chunk_bytes // (nodes * dtype.itemsize)))
    for i in range(0, count, step):
        block = np.asarray(mm[i : i + step], np.float32) * scale + zero
        for k, v in flat.items():
            out[k][i : i + step] = block[:, v]
        probes[i : i + step] = block[:, probe_idx]
    data = {k: out[k].reshape((count,) + planes[k].shape) for k in out}
    data["probes"] = probes
    data["t"] = stage.t
    os.makedirs(cache.parent, exist_ok=True)
    np.savez(cache, **data)
    return data


# ---------------- signal helpers ----------------
def crossings(trace, t, thr=THRESH_MV, refractory=50.0):
    """Upward threshold crossings of a 1D trace (linear interpolation), with refractory gap."""
    v = np.asarray(trace, float)
    i = np.flatnonzero((v[:-1] < thr) & (v[1:] >= thr))
    times = t[i] + (thr - v[i]) / (v[i + 1] - v[i]) * (t[i + 1] - t[i])
    keep, last = [], -np.inf
    for tc in times:
        if tc - last >= refractory:
            keep.append(tc)
            last = tc
    return np.array(keep)


def first_crossing(v, t, start, stop, thr=THRESH_MV):
    """First upward crossing time in [start, stop) per node of v (T, ...); NaN if none."""
    sel = np.flatnonzero((t >= start - 1e-9) & (t < stop))
    if len(sel) < 2:
        return np.full(v.shape[1:], np.nan)
    w = np.asarray(v[sel], np.float32)
    tw = t[sel]
    up = (w[:-1] < thr) & (w[1:] >= thr)
    has = up.any(axis=0)
    k = np.argmax(up, axis=0)
    a = np.take_along_axis(w, k[None], 0)[0]
    b = np.take_along_axis(w, (k + 1)[None], 0)[0]
    frac = np.where(b != a, (thr - a) / np.where(b != a, b - a, 1), 0)
    out = tw[k] + frac * (tw[np.minimum(k + 1, len(tw) - 1)] - tw[k])
    return np.where(has, out, np.nan)


def chain(stages, names):
    """Concatenate non-overlapping consecutive stages (later stage wins at shared frames)."""
    parts = [s for s in stages if s.name in names and s.vm is not None]
    parts.sort(key=lambda s: s.t[0])
    if not parts:
        return None
    t_all, probes, planes, owner = [], [], {}, []
    for j, s in enumerate(parts):
        nxt = parts[j + 1].t[0] if j + 1 < len(parts) else np.inf
        keep = s.t < nxt - 1e-9
        t_all.append(s.t[keep])
        probes.append(s.vm["probes"][keep])
        owner += [s.name] * int(keep.sum())
        for k in ("top", "mid", "deep", "xz", "yz"):
            planes.setdefault(k, []).append(s.vm[k][keep])
    out = {k: np.concatenate(v) for k, v in planes.items()}
    out.update(t=np.concatenate(t_all), probes=np.concatenate(probes), owner=np.array(owner))
    return out


def apd90(trace, t, t_act):
    """APD90 of the action potential starting at t_act (ms), or NaN."""
    pre = (t >= t_act - 15) & (t < t_act)
    post = (t >= t_act) & (t < t_act + 150)
    if not pre.any() or not post.any():
        return np.nan
    rest, peak_i = np.min(trace[pre]), np.flatnonzero(post)[np.argmax(trace[post])]
    v90 = trace[peak_i] - 0.9 * (trace[peak_i] - rest)
    after = np.flatnonzero((np.arange(len(t)) > peak_i) & (trace < v90))
    return float(t[after[0]] - t_act) if after.size else np.nan


# ---------------- plotting helpers ----------------
def outline(ax, grid, layer="top", section=None, colours=True, lw=0.6, colour=None):
    """Region outlines (non-wall tags) of a top-view voxel layer or an x-z / y-z section."""
    if section is None:
        iz = grid.vtag.shape[0] - 1 if layer == "top" else min(grid.iz(layer), grid.vtag.shape[0] - 1)
        tags, X, Y = grid.vtag[iz], grid.vx, grid.vy
    elif section == "xz":
        tags, X, Y = grid.vtag[:, min(grid.iy(0.0), grid.vtag.shape[1] - 1), :], grid.vx, grid.vz
    else:
        tags, X, Y = grid.vtag[:, :, min(grid.ix(0.0), grid.vtag.shape[2] - 1)], grid.vy, grid.vz
    for tag in np.unique(tags):
        if tag == 1 or tags.shape[0] < 2 or tags.shape[1] < 2:
            continue
        c = colour or (TAGS.get(tag, ("", INK))[1] if colours else INK)
        ax.contour(X, Y, (tags == tag).astype(float), levels=[0.5], colors=[c], linewidths=lw)


def tag_image(ax, tags, X, Y):
    ids = sorted(set(np.unique(tags)) | set(TAGS))
    lut = {t: i for i, t in enumerate(ids)}
    cmap = ListedColormap([TAGS.get(t, (str(t), MUTED))[1] for t in ids])
    ax.pcolormesh(
        X, Y, np.vectorize(lut.get)(tags), cmap=cmap, vmin=-0.5, vmax=len(ids) - 0.5,
        shading="nearest", rasterized=True,
    )


def panel_label(ax, letter):
    ax.text(-0.02, 1.02, letter, transform=ax.transAxes, fontweight="bold", fontsize=9,
            ha="right", va="bottom")


def stage_legend(ax_or_fig, names, loc="upper right", ncol=None, **kw):
    handles = [Line2D([], [], color=stage_colour(n), lw=2) for n in names]
    labels = [stage_label(n) for n in names]
    return ax_or_fig.legend(handles, labels, loc=loc, ncol=ncol or len(names), **kw)


# ---------------- figure context ----------------
class Ctx:
    def __init__(self, args):
        self.out = Path(args.results) / "figures"
        self.cache = self.out / "cache"
        self.tables = self.out / "tables"
        for d in (self.out, self.cache, self.tables):
            d.mkdir(parents=True, exist_ok=True)
        with open(args.config) as f:
            self.cfg = ep.deep_update(ep.DEFAULTS, json.load(f))
        self.results = Path(args.results)
        self.stages = [Stage(f) for f in args.stages]
        self.by_name = {s.name: s for s in self.stages}
        self.grids = {}
        for s in self.stages:
            if s.meshbase not in self.grids:
                self.grids[s.meshbase] = Grid(s.meshbase)
        self.stats = {}
        self.captions = []
        self.t_lift = next(
            (s.meta.get("t_lift_ms") for s in self.stages if s.meta.get("t_lift_ms")), None
        )
        if self.t_lift is None and "s2_block" in self.by_name:
            self.t_lift = float(self.by_name["s2_block"].t[-1])
        summ = self.results / "run_summary.json"
        self.summary = json.loads(summ.read_text()) if summ.exists() else {}
        self.gain = float(self.summary.get("gain", 1.0))
        self.points = self._csv("points.csv")
        self.gt = self._csv("groundtruth.csv")
        curves = self.results / "groundtruth_curves.npz"
        self.curves = dict(np.load(curves)) if curves.exists() else None
        self.vm_ok = False
        self.load_vm = not args.no_vm
        self._egm = {}

    def _csv(self, name):
        path = self.results / name
        if not path.exists():
            return None
        df = pd.read_csv(path)
        df["stage_name"] = df["run"].astype(str).str.split("/").str[-1]
        return df

    def grid(self, stage=None):
        return self.grids[(stage or self.stages[0]).meshbase]

    def pre_grid(self):
        pre = [s for s in self.stages if s.state == "pre"]
        return self.grid(pre[0] if pre else self.stages[0])

    def post_grid(self):
        post = [s for s in self.stages if s.state == "post"]
        return self.grid(post[0]) if post else None

    def ensure_vm(self):
        if self.vm_ok or not self.load_vm:
            return self.vm_ok
        for s in self.stages:
            if s.vm is None:
                if not (s.folder / "vm.igb").exists() and not (
                    self.cache / f"{s.name}_vm_planes.npz"
                ).exists():
                    print(f"  (no vm.igb for {s.name}; its Vm figures are skipped)")
                    continue
                started = time.time()
                s.vm = load_vm(s, self.grid(s), self.cache)
                print(f"  Vm planes for {s.name}: {time.time() - started:.0f} s", flush=True)
        self.vm_ok = any(s.vm is not None for s in self.stages)
        return self.vm_ok

    def pre_chain(self):
        names = [n for n in ("s1_pace", "s2_block", "s3_reentry") if n in self.by_name]
        return chain(self.stages, names)

    def egm_signals(self, stage):
        """Gain-scaled, noise-free, filtered unipolar traces over the analysis window."""
        if stage.name in self._egm:
            return self._egm[stage.name]
        with np.load(stage.folder / "egm.npz") as z:
            t = z["t_ms"]
            raw, window = ep.analysis_slices(t, self.cfg)
            uni = self.gain * z["uni"][..., raw].astype(np.float64)
            info = dict(
                centres=z["centre_xy_mm"] if "centre_xy_mm" in z.files else np.zeros((len(uni), 2)),
                gaps=z["p_gap_mm"] if "p_gap_mm" in z.files else np.zeros(len(uni)),
                xy=z["uni_xy"],
            )
        fs = 1000.0 / float(np.median(np.diff(t)))
        F = self.cfg["filters"]
        uf = ep.apply_filter(ep.make_filter(F["unipolar_hz"], fs, F["order"]), uni, F["zero_phase"])
        df = ep.apply_filter(ep.make_filter(F["differential_hz"], fs, F["order"]), uni, F["zero_phase"])
        info.update(t=t[raw][window], uf=uf[..., window], df=df[..., window])
        self._egm[stage.name] = info
        return info

    def placement(self, stage, centre, gap):
        e = self.egm_signals(stage)
        d = np.hypot(*(e["centres"] - np.asarray(centre, float)).T) + 100 * np.abs(e["gaps"] - gap)
        return int(np.argmin(d))

    def trace(self, stage, p, kind, label):
        """Recording trace using the exact electrode weights saved in points.csv."""
        e = self.egm_signals(stage)
        rows = self.points[
            (self.points.stage_name == stage.name)
            & (self.points.placement == p)
            & (self.points.kind == kind)
            & (self.points.label == label)
        ]
        w = rows.iloc[0][[f"weight_E{i}" for i in range(16)]].to_numpy(float)
        return e["t"], w @ (e["uf"][p] if kind == "uni" else e["df"][p])

    def save(self, fig, number, name, title, caption, stats=None):
        base = self.out / f"{number:02d}_{name}"
        fig.savefig(str(base) + ".png", dpi=300, bbox_inches="tight")
        fig.savefig(str(base) + ".pdf", bbox_inches="tight")
        plt.close(fig)
        self.captions.append(dict(number=number, file=base.name, title=title, caption=caption,
                                  stats=stats or {}))
        if stats:
            self.stats[base.name] = stats
        print(f"  {base.name}", flush=True)

    def compare_pair(self):
        """(baseline, comparison) stage names for pre/post comparisons."""
        names = [s.name for s in self.stages]
        pre = next((n for n in ("s3_reentry", "s2_block", "s1_pace") if n in names), None)
        post = next((n for n in ("s4_ablate_terminate", "post_s3") if n in names), None)
        return pre, post


# ======================================================================
# A. Model and protocol
# ======================================================================
def fig_geometry(ctx, number):
    pre, post = ctx.pre_grid(), ctx.post_grid()
    grids = [("Pre-ablation", pre)] + ([("Post-ablation", post)] if post is not None else [])
    fig = plt.figure(figsize=(WIDTH, 6.2), constrained_layout=True)
    gs = fig.add_gridspec(3, len(grids), height_ratios=[3.2, 1.3, 0.9])
    stim = stim_box(ctx)
    for j, (title, g) in enumerate(grids):
        ax = fig.add_subplot(gs[0, j])
        tag_image(ax, g.vtag[-1], g.vx, g.vy)
        if stim is not None:
            (x0, y0, _), (x1, y1, _) = stim
            ax.add_patch(Rectangle((x0, y0), x1 - x0, y1 - y0, fill=False, ec=INK, lw=0.8, ls="--"))
            ax.annotate("stimulus", (x1, y1), (x1 + 1.2, y1 + 3.0), fontsize=6.5, color=INK2,
                        arrowprops=dict(arrowstyle="-", color=INK2, lw=0.5))
        for k, (_, _, xyz) in enumerate(PROBES):
            if xyz[2] > 0:
                ax.plot(xyz[0], xyz[1], "o", ms=3.2, mfc=SURFACE, mec=INK, mew=0.6)
                ax.text(xyz[0], xyz[1] + 0.8, str(k + 1), fontsize=5.5, ha="center", color=INK)
        ax.set_aspect("equal")
        ax.set(title=f"{title}: top view (z = {g.zs[-1]:g} mm)", xlabel="x (mm)", ylabel="y (mm)")
        panel_label(ax, "ab"[j])
        ax2 = fig.add_subplot(gs[1, j])
        ixv = min(g.ix(0.0), g.vtag.shape[2] - 1)
        tag_image(ax2, g.vtag[:, :, ixv], g.vy, g.vz)
        ax2.set_aspect("equal")
        ax2.set_xlim(-5, 5)
        ax2.set(title="y-z section at x = 0", xlabel="y (mm)", ylabel="z (mm)")
        panel_label(ax2, "cd"[j])
    ax3 = fig.add_subplot(gs[2, :])
    g = post if post is not None else pre
    tag_image(ax3, g.vtag[:, min(g.iy(0.0), g.vtag.shape[1] - 1), :], g.vx, g.vz)
    ax3.set_aspect("equal")
    ax3.set(title=f"x-z section at y = 0 ({'post' if post is not None else 'pre'}-ablation)",
            xlabel="x (mm)", ylabel="z (mm)")
    panel_label(ax3, "e" if post is not None else "c")
    present = sorted(set(np.unique(pre.vtag)) | (set(np.unique(post.vtag)) if post is not None else set()))
    handles = [Patch(fc=TAGS.get(t, (str(t), MUTED))[1], ec=MUTED, lw=0.4,
                     label=f"{t}: {TAGS.get(t, (f'tag {t}',))[0]}") for t in present]
    fig.legend(handles=handles, loc="lower center", ncol=len(handles), bbox_to_anchor=(0.5, -0.04))
    counts = {TAGS.get(int(t), (f"tag {t}",))[0]: int(c) for t, c in zip(*np.unique(g.vtag, return_counts=True))}
    stats = dict(nodes=pre.n_nodes, grid_shape_zyx=[int(v) for v in pre.shape], spacing_mm=float(pre.h),
                 voxels_per_region=counts)
    probe_list = "; ".join(f"{k + 1} = {p[1]}" for k, p in enumerate(PROBES) if p[2][2] > 0)
    ctx.save(fig, number, "geometry", "Slab geometry and mesh regions",
             f"Mesh regions of the {np.ptp(pre.xs):g} x {np.ptp(pre.ys):g} x {np.ptp(pre.zs):g} mm slab "
             f"({pre.n_nodes:,} nodes, {pre.h:g} mm spacing). The insulated myobundle runs along x "
             "at the top surface; the exit segment (tag 3) is blocked only during s2. The lesion "
             "(tag 5) is a transmural cylinder through the bundle midpoint, present only in "
             "post-ablation stages. Dashed box: pacing stimulus. Numbered circles: Vm probe sites "
             f"used in later figures ({probe_list}).", stats)


def stim_box(ctx):
    par = ctx.stages[0].folder / "parameters.par"
    if not par.exists():
        return None
    import ground_truth as gtm

    p = gtm.read_par(str(par))
    try:
        p0 = [float(p[f"stim[0].elec.p0[{i}]"]) / 1000 for i in range(3)]
        p1 = [float(p[f"stim[0].elec.p1[{i}]"]) / 1000 for i in range(3)]
        return p0, p1
    except KeyError:
        return None


def pacing_times(ctx):
    par = ctx.stages[0].folder / "parameters.par"
    try:
        import ground_truth as gtm

        p = gtm.read_par(str(par))
        start, bcl, n = float(p["stim[0].ptcl.start"]), float(p["stim[0].ptcl.bcl"]), int(p["stim[0].ptcl.npls"])
    except (OSError, KeyError, ValueError):
        start, bcl, n = 0.0, 200.0, 10
    return start + bcl * np.arange(n)


def fig_catheter(ctx, number):
    s = ctx.stages[0]
    g = ctx.grid(s)
    with np.load(s.folder / "egm.npz") as z:
        files = z.files
        xyz = z["uni_xy"]
        centres = z["centre_xy_mm"] if "centre_xy_mm" in files else np.zeros((len(xyz), 2))
        gaps = z["p_gap_mm"] if "p_gap_mm" in files else np.zeros(len(xyz))
        samples = z["uni_sample_xyz"] if "uni_sample_xyz" in files else None
        lat_x = z["x_mm"] if "x_mm" in files else np.unique(xyz[..., 0])
        lat_y = z["y_mm"] if "y_mm" in files else np.unique(xyz[..., 1])
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(WIDTH, 3.4), constrained_layout=True,
                                  gridspec_kw=dict(width_ratios=[1.25, 1]))
    outline(ax, g, colours=True, lw=0.8)
    post = ctx.post_grid()
    if post is not None:
        outline(ax, post, colours=False, colour=TAGS[5][1], lw=0.8)
    LX, LY = np.meshgrid(lat_x, lat_y)
    ax.plot(LX.ravel(), LY.ravel(), ".", ms=1.2, color=MUTED, label="electrode lattice (1 mm)")
    uc = np.unique(np.round(centres, 6), axis=0)
    ax.add_patch(Rectangle(uc.min(axis=0), *(np.ptp(uc, axis=0)), fill=False, ec=INK2, lw=0.6, ls=":"))
    p0 = int(np.argmin(np.hypot(*centres.T) + 100 * np.abs(gaps - gaps.min())))
    e = xyz[p0]
    for a, b, b2 in ep.bipolar_pairs(ctx.cfg["bipolar_mode"]):
        for q in [b] + ([b2] if b2 is not None else []):
            ax.plot(e[[a, q], 0], e[[a, q], 1], color=STAGE_COLOURS["s3_reentry"], lw=0.6, alpha=0.6)
    ax.plot(e[:, 0], e[:, 1], "o", ms=6, mfc=SURFACE, mec=INK, mew=0.7)
    for k in range(16):
        ax.text(e[k, 0], e[k, 1], str(k), fontsize=4.5, ha="center", va="center")
    gtc = np.array(ctx.cfg["groundtruth"].get("placements_xy_mm_pre") or [[0, 0]], float)
    ax.plot(gtc[:, 0], gtc[:, 1], "x", color=INK, ms=4, mew=0.8, label="ground-truth centres")
    mpc = np.array(ctx.cfg["groundtruth"].get("map_xy_mm_pre") or [[0, 0]], float)
    ax.plot(mpc[:, 0], mpc[:, 1], "s", ms=5, mfc="none", mec=STAGE_COLOURS["s4_ablate_terminate"],
            mew=0.9, label="contribution-map centres")
    ax.set_aspect("equal")
    lim = max(np.abs(lat_x).max(), np.abs(lat_y).max()) + 1.5
    ax.set(xlim=(-lim, lim), ylim=(-lim, lim), xlabel="x (mm)", ylabel="y (mm)",
           title="HD Grid placement over the slab (top view)")
    ax.legend(loc="lower left", fontsize=6, bbox_to_anchor=(0, -0.02))
    panel_label(ax, "a")
    # side view y-z through x = 0 with the electrode cylinders
    ixv = min(g.ix(0.0), g.vtag.shape[2] - 1)
    tag_image(ax2, g.vtag[:, :, ixv], g.vy, g.vz)
    for gap in np.unique(gaps):
        p = int(np.argmin(np.hypot(*centres.T) + 100 * np.abs(gaps - gap)))
        row = xyz[p][xyz[p][:, 0] == xyz[p][:, 0].min()] if xyz.shape[-1] > 2 else None
        if row is None:
            continue
        r = 0.5
        for q in row:
            ax2.add_patch(Circle((q[1], q[2]), r, fill=False, ec=INK, lw=0.8, ls=GAP_STYLE.get(float(gap), "-")))
        if samples is not None:
            smp = samples[p][xyz[p][:, 0] == xyz[p][:, 0].min()].reshape(-1, 3)
            ax2.plot(smp[:, 1], smp[:, 2], ".", ms=1.5, color=INK2)
        ax2.text(row[:, 1].max() + 0.9, row[0, 2], f"gap {gap:g} mm", fontsize=6.5, va="center")
    ax2.set_aspect("equal")
    ax2.set(xlim=(-8.5, 10.5), ylim=(g.zs[0] - 0.2, g.zs[-1] + 2.6), xlabel="y (mm)", ylabel="z (mm)",
            title="Side view: electrodes above the surface")
    panel_label(ax2, "b")
    n_centres = len(uc)
    ctx.save(fig, number, "catheter_layout", "Virtual HD Grid catheter",
             f"(a) The 4 x 4 HD Grid (4 mm electrode spacing, electrode numbers shown) at the central "
             f"placement. It is stepped across {n_centres} catheter centres (dotted box) at "
             f"{len(np.unique(gaps))} electrode-tissue gaps ({', '.join(f'{v:g}' for v in np.unique(gaps))} mm), "
             f"giving {len(centres)} placements. Lines: orthogonal bipolar pairs (max_orthogonal: "
             "the larger of the along/across bipoles sharing a vertex). Crosses: placements with "
             "lead-field ground truth; squares: contribution-map placements. (b) Electrodes are "
             "cylinders of radius 0.5 mm; dots are the 8 surface samples averaged per electrode.",
             dict(placements=int(len(centres)), catheter_centres=int(n_centres),
                  gaps_mm=[float(v) for v in np.unique(gaps)]))


def fig_protocol(ctx, number):
    fig, ax = plt.subplots(figsize=(WIDTH, 2.4), constrained_layout=True)
    order = [s for s in ctx.stages]
    pacing = pacing_times(ctx)
    for row, s in enumerate(order[::-1]):
        y = row
        ax.barh(y, s.t[-1] - s.t[0], left=s.t[0], height=0.5, color=s.colour, alpha=0.35, lw=0)
        if ctx.points is not None:
            w = ctx.points[ctx.points.stage_name == s.name]
            if len(w):
                a, b = w.window_start_ms.iloc[0], w.window_end_ms.iloc[0]
                ax.barh(y, b - a, left=a, height=0.5, color=s.colour, lw=0)
        ax.text(s.t[0] - 40, y, s.label, ha="right", va="center", fontsize=7)
        paced = pacing[(pacing >= s.t[0]) & (pacing <= s.t[-1])]
        ax.plot(paced, np.full(len(paced), y + 0.38), "|", color=INK, ms=5, mew=0.8)
    ax.set_yticks([])
    ax.spines["left"].set_visible(False)
    marks = [(1795, "checkpoint 1795")]
    if ctx.t_lift:
        marks.append((ctx.t_lift, f"block lifted {ctx.t_lift:g}"))
    if any(s.name == "s4_ablate_terminate" for s in ctx.stages):
        marks.append((3500, "ablation 3500"))
    for k, (t, label) in enumerate(marks):
        ax.axvline(t, color=INK2, lw=0.6, ls=":")
        ax.text(t + 15, len(order) - 0.3 + 0.32 * (k % 2), label, fontsize=6.5, ha="left", color=INK2)
    ax.set_xlim(-1500, max(s.t[-1] for s in ctx.stages) + 100)
    ax.set_ylim(-0.6, len(order) + 0.4)
    ax.set_xlabel("time (ms)")
    ax.set_xticks(np.arange(0, max(s.t[-1] for s in ctx.stages) + 1, 1000))
    fig.legend(handles=[Patch(fc=MUTED, alpha=0.35, label="simulated interval"),
                        Patch(fc=MUTED, label="EGM / ground-truth analysis window"),
                        Line2D([], [], marker="|", ls="none", color=INK, label="pacing stimulus")],
               loc="lower center", ncol=3, fontsize=6.5, bbox_to_anchor=(0.5, -0.1))
    windows = {}
    if ctx.points is not None:
        for s in ctx.stages:
            w = ctx.points[ctx.points.stage_name == s.name]
            if len(w):
                windows[s.name] = [float(w.window_start_ms.iloc[0]), float(w.window_end_ms.iloc[0])]
    ctx.save(fig, number, "protocol_timeline", "Simulation protocol",
             "Checkpointed stage sequence. Pale bars: simulated interval; solid bars: the EGM and "
             "ground-truth analysis window (last "
             f"{ctx.cfg.get('analysis_window_last_ms')} ms). Ticks: pacing stimuli (CL "
             f"{np.diff(pacing).mean() if len(pacing) > 1 else float('nan'):g} ms). The post-ablation "
             "stage restarts from the 3500 ms checkpoint of s3, so it overlaps s3 between 3500 and "
             "4000 ms and is a branch, not a continuation.",
             dict(stage_intervals_ms={s.name: [float(s.t[0]), float(s.t[-1])] for s in ctx.stages},
                  analysis_windows_ms=windows, pacing_ms=[float(v) for v in pacing]))


# ======================================================================
# B. Electrophysiology from Vm
# ======================================================================
def needs_vm(fn):
    def wrapper(ctx, number, *args):
        if not ctx.ensure_vm():
            raise RuntimeError("Vm not available (vm.igb missing or --no-vm)")
        return fn(ctx, number, *args)

    wrapper.__name__ = fn.__name__
    return wrapper


def probe_series(ctx):
    """[(stage-name-or-'pre', t, probes, owner)] for the pre chain and each post branch."""
    out = []
    pre = ctx.pre_chain()
    if pre is not None:
        out.append(("pre", pre))
    for s in ctx.stages:
        if s.name not in ("s1_pace", "s2_block", "s3_reentry") and s.vm is not None:
            out.append((s.name, dict(t=s.t, probes=s.vm["probes"], owner=np.array([s.name] * len(s.t)),
                                     **{k: s.vm[k] for k in ("top", "mid", "deep", "xz", "yz")})))
    return out


def coloured_line(ax, t, v, owner, lw=0.6):
    for name in pd.unique(owner):
        m = owner == name
        ax.plot(t[m], v[m], color=stage_colour(name), lw=lw)


@needs_vm
def fig_vm_traces(ctx, number):
    series = probe_series(ctx)
    keys = [0, 2, 4, 5, 7, 8]  # pacing site, exit, bundle middle, entry, remote wall, below
    fig, axes = plt.subplots(len(keys), 1, figsize=(WIDTH, 6.5), sharex=True, constrained_layout=True)
    for ax, k in zip(axes, keys):
        for _, d in series:
            coloured_line(ax, d["t"], d["probes"][:, k], d["owner"])
        ax.set_ylim(*VM_LIM)
        ax.set_yticks([-80, -40, 0])
        ax.text(0.005, 0.92, f"{k + 1}. {PROBES[k][1]}", transform=ax.transAxes, fontsize=7,
                va="top", bbox=dict(fc=SURFACE, ec="none", alpha=0.8, pad=1))
        for t, ls in ((ctx.t_lift, ":"), (3500, "--")):
            if t:
                ax.axvline(t, color=INK2, lw=0.5, ls=ls)
    axes[len(keys) // 2].set_ylabel("Vm (mV)")
    axes[-1].set_xlabel("time (ms)")
    stage_legend(axes[0], [s.name for s in ctx.stages], loc="lower right", bbox_to_anchor=(1, 1.0))
    ctx.save(fig, number, "vm_probe_traces", "Transmembrane potential at probe sites",
             "Vm at the probe sites numbered in the geometry figure, over the whole protocol. The "
             "pre-ablation stages are continuous (s1 -> s2 -> s3); the post-ablation stage is a "
             "branch from 3500 ms and is overlaid in its own colour. Dotted line: block lifted; "
             "dashed line: lesion applied.")


@needs_vm
def fig_tlift_window(ctx, number):
    pre = ctx.pre_chain()
    if pre is None or not ctx.t_lift:
        raise RuntimeError("needs the pre-ablation chain with s2")
    t, P = pre["t"], pre["probes"]
    pacing = pacing_times(ctx)
    beat = float(pacing[-1]) if len(pacing) else 1800.0
    names = ["pace", "wall_exit", "exit", "interface", "mid", "entry"]
    idx = [next(i for i, p in enumerate(PROBES) if p[0] == n) for n in names]
    acts = {}
    for i in idx:
        c = crossings(P[:, i], t)
        after = c[c >= beat - 1]
        acts[PROBES[i][0]] = float(after[0]) if after.size else np.nan
    apd = apd90(P[:, idx[1]], t, acts["wall_exit"]) if np.isfinite(acts["wall_exit"]) else np.nan
    lower = beat + apd + 10 if np.isfinite(apd) else np.nan
    upper = acts["interface"] - 10 if np.isfinite(acts["interface"]) else np.nan
    inside = bool(np.isfinite(lower) and np.isfinite(upper) and lower < ctx.t_lift < upper)
    t_end = min(t[-1], beat + 600)
    m = (t >= beat - 20) & (t <= t_end)
    fig, ax = plt.subplots(figsize=(WIDTH, 3.2), constrained_layout=True)
    colours = ["#1baf7a", "#4a3aa7", "#eb6834", "#2a78d6", "#3a3936", "#8a8984"]
    for c, i in zip(colours, idx):
        ax.plot(t[m], P[m, i], color=c, lw=1.0, label=f"{i + 1}. {PROBES[i][1]}")
    if np.isfinite(lower) and np.isfinite(upper) and upper > lower:
        ax.axvspan(lower, upper, color=GRID, lw=0, zorder=0, label="admissible T_LIFT window")
    ax.axvline(ctx.t_lift, color=INK, lw=0.9, ls="--", label=f"T_LIFT = {ctx.t_lift:g} ms")
    ax.set(xlabel="time (ms)", ylabel="Vm (mV)", ylim=VM_LIM, title="Beat 10 and the exit-block lift")
    ax.legend(loc="upper center", fontsize=6.3, ncol=4, bbox_to_anchor=(0.5, -0.17))
    stats = dict(beat10_ms=beat, activation_ms={k: fmt(v, 1) for k, v in acts.items()},
                 apd90_wall_beside_exit_ms=fmt(apd, 1), window_ms=[fmt(lower, 1), fmt(upper, 1)],
                 t_lift_ms=ctx.t_lift, t_lift_inside_window=inside)
    ctx.save(fig, number, "tlift_window", "Unidirectional block: timing of the block lift",
             f"Vm around the last paced beat ({beat:g} ms). The block-lift time must fall after the "
             "wall beside the exit has repolarised (beat + APD90 + 10 ms) and before the "
             "orthodromic wave reaches the bundle/exit interface (arrival - 10 ms). Measured APD90 "
             f"beside the exit {fmt(apd, 1)} ms; arrival at the interface {fmt(acts['interface'], 1)} ms; "
             f"window {fmt(lower, 0)}-{fmt(upper, 0)} ms. T_LIFT = {ctx.t_lift:g} ms is "
             f"{'inside' if inside else 'OUTSIDE'} this window.", stats)


def snapshot_times(ctx, stage, which):
    t0, t1 = stage.t[0], stage.t[-1]
    if which == "early":
        span = (t0 + (5 if t0 == 0 else 0), min(t1, t0 + (150 if stage.name.endswith("s1_pace") or stage.name == "post_s1" else 350)))
    else:
        span = (max(t0, t1 - 300), t1)
    return np.linspace(span[0], span[1], 8)


@needs_vm
def fig_snapshots(ctx, number, stage_name, which):
    s = ctx.by_name[stage_name]
    if s.vm is None:
        raise RuntimeError("no Vm for " + stage_name)
    g = ctx.grid(s)
    times = snapshot_times(ctx, s, which)
    fig, axes = plt.subplots(2, 4, figsize=(WIDTH, 3.5), sharex=True, sharey=True, constrained_layout=True)
    im = None
    for ax, tq in zip(axes.ravel(), times):
        k = int(np.argmin(np.abs(s.t - tq)))
        im = ax.pcolormesh(g.xs, g.ys, s.vm["top"][k].astype(np.float32), cmap=VM_CMAP,
                           vmin=VM_LIM[0], vmax=VM_LIM[1], shading="nearest", rasterized=True)
        outline(ax, g, colours=False, colour=INK, lw=0.4)
        ax.set_aspect("equal")
        ax.set_title(f"{s.t[k]:.0f} ms", fontsize=7.5)
    for ax in axes[-1]:
        ax.set_xlabel("x (mm)")
    for ax in axes[:, 0]:
        ax.set_ylabel("y (mm)")
    cb = fig.colorbar(im, ax=axes, shrink=0.75, pad=0.01)
    cb.set_label("Vm (mV)")
    label = {"early": "start", "late": "end"}[which]
    ctx.save(fig, number, f"snapshots_{stage_name}_{which}", f"Vm snapshots: {s.label} ({label})",
             f"Top-surface Vm (z = {g.zs[-1]:g} mm, the surface under the catheter) at eight times near "
             f"the {label} of {s.label} ({s.t[0]:g}-{s.t[-1]:g} ms). Dark = depolarised. Outlines: "
             "bundle, sheath, exit segment and (post-ablation) lesion.")


@needs_vm
def fig_depth_snapshots(ctx, number, stage_name):
    s = ctx.by_name[stage_name]
    if s.vm is None:
        raise RuntimeError("no Vm for " + stage_name)
    g = ctx.grid(s)
    times = snapshot_times(ctx, s, "early")[::2][:4]
    fig, axes = plt.subplots(len(times), 1, figsize=(WIDTH, 0.95 * len(times) + 0.6), sharex=True,
                             constrained_layout=True)
    im = None
    for ax, tq in zip(np.atleast_1d(axes), times):
        k = int(np.argmin(np.abs(s.t - tq)))
        im = ax.pcolormesh(g.xs, g.zs, s.vm["xz"][k].astype(np.float32), cmap=VM_CMAP,
                           vmin=VM_LIM[0], vmax=VM_LIM[1], shading="nearest", rasterized=True)
        outline(ax, g, section="xz", colours=False, colour=INK, lw=0.4)
        ax.set_aspect("equal")
        ax.set_ylabel("z (mm)")
        ax.text(1.005, 0.5, f"{s.t[k]:.0f} ms", transform=ax.transAxes, fontsize=7, va="center")
    np.atleast_1d(axes)[-1].set_xlabel("x (mm)")
    cb = fig.colorbar(im, ax=axes, shrink=0.8, pad=0.06)
    cb.set_label("Vm (mV)")
    ctx.save(fig, number, f"depth_snapshots_{stage_name}", f"Transmural Vm: {s.label}",
             f"Vm on the x-z section through the bundle axis (y = 0) during {s.label}, showing "
             "propagation through the wall thickness and along the insulated bundle. Outlines: "
             "mesh regions.")


@needs_vm
def fig_time_space(ctx, number):
    series = probe_series(ctx)
    g = ctx.pre_grid()
    iy_b, iy_w = g.iy(0.0), g.iy(-4.0)
    fig, axes = plt.subplots(2 * len(series), 1, figsize=(WIDTH, 1.45 * 2 * len(series) + 0.6),
                             constrained_layout=True, squeeze=False)
    axes = axes[:, 0]
    im = None
    for j, (name, d) in enumerate(series):
        for k, (iy, where) in enumerate(((iy_b, "bundle axis, y = 0"), (iy_w, "wall, y = -4 mm"))):
            ax = axes[2 * j + k]
            v = d["mid"][:, iy, :].astype(np.float32).T  # (x, t)
            im = ax.imshow(v, aspect="auto", origin="lower", cmap=VM_CMAP, vmin=VM_LIM[0], vmax=VM_LIM[1],
                           extent=(d["t"][0], d["t"][-1], g.xs[0], g.xs[-1]), interpolation="nearest",
                           rasterized=True)
            title = "pre-ablation chain (s1-s3)" if name == "pre" else stage_label(name)
            ax.set_title(f"{title}: {where}, z = {g.plane_z()['mid']:g} mm", fontsize=7.5, loc="left")
            ax.set_ylabel("x (mm)")
            for t, ls in ((ctx.t_lift, ":"), (3500, "--")):
                if t and d["t"][0] < t < d["t"][-1]:
                    ax.axvline(t, color=INK, lw=0.5, ls=ls)
    axes[-1].set_xlabel("time (ms)")
    cb = fig.colorbar(im, ax=axes, shrink=0.5, pad=0.01)
    cb.set_label("Vm (mV)")
    ctx.save(fig, number, "time_space", "Time-space plots",
             "Vm along x (vertical) against time (horizontal) on two lines at mid bundle depth: the "
             "bundle axis and a parallel line in the wall 4 mm away. Sloped bands are propagating "
             "wavefronts (slope = 1/velocity); repeating bands after the block lift indicate "
             "recurring activation. Dotted: block lifted; dashed: lesion applied.")


@needs_vm
def fig_time_space_zoom(ctx, number):
    pre, post = ctx.compare_pair()
    picks = [n for n in (pre, post) if n and ctx.by_name[n].vm is not None]
    if not picks:
        raise RuntimeError("no stage with Vm")
    g = ctx.pre_grid()
    fig, axes = plt.subplots(2, len(picks), figsize=(WIDTH, 3.6), constrained_layout=True, squeeze=False,
                             sharey=True)
    im = None
    for j, name in enumerate(picks):
        s = ctx.by_name[name]
        span = (s.t[-1] - 600, s.t[-1]) if name == pre else (s.t[0], s.t[0] + 600)
        m = (s.t >= span[0]) & (s.t <= span[1])
        for k, (y, where) in enumerate(((0.0, "bundle axis"), (-4.0, "wall, y = -4 mm"))):
            ax = axes[k, j]
            v = s.vm["mid"][m][:, g.iy(y), :].astype(np.float32).T
            im = ax.imshow(v, aspect="auto", origin="lower", cmap=VM_CMAP, vmin=VM_LIM[0], vmax=VM_LIM[1],
                           extent=(s.t[m][0], s.t[m][-1], g.xs[0], g.xs[-1]), rasterized=True)
            ax.set_title(f"{s.label}: {where}", fontsize=7.5, loc="left")
            if k == 1:
                ax.set_xlabel("time (ms)")
            if j == 0:
                ax.set_ylabel("x (mm)")
    cb = fig.colorbar(im, ax=axes, shrink=0.7, pad=0.01)
    cb.set_label("Vm (mV)")
    ctx.save(fig, number, "time_space_zoom", "Time-space plots before and after ablation",
             "Zoom of the time-space plots: the last 600 ms before ablation (left) and the first 600 ms "
             "after the lesion is applied (right). Compare the recurrence of bands and their extent "
             "along the bundle.")


def cycle_window(ctx):
    """Median activation interval at the bundle middle in the last 1000 ms of the baseline stage."""
    pre, _ = ctx.compare_pair()
    s = ctx.by_name.get(pre)
    if s is None or s.vm is None:
        return None
    c = crossings(s.vm["probes"][:, 4], s.t)
    c = c[c > s.t[-1] - 1000]
    return float(np.median(np.diff(c))) if len(c) > 2 else None


@needs_vm
def fig_activation_maps(ctx, number):
    pre_chain = ctx.pre_chain()
    pacing = pacing_times(ctx)
    panels = []
    if pre_chain is not None and pre_chain["t"][0] <= pacing[0] + 1:
        panels.append(("Beat 1 (first paced)", pre_chain, pacing[0], pacing[0] + 200))
    if pre_chain is not None and len(pacing) and pre_chain["t"][-1] > pacing[-1] + 50:
        panels.append((f"Beat 10 (blocked exit, from {pacing[-1]:g} ms)", pre_chain, pacing[-1], pacing[-1] + 350))
    cl = cycle_window(ctx) or 300.0
    pre, post = ctx.compare_pair()
    if pre and ctx.by_name[pre].vm is not None:
        s = ctx.by_name[pre]
        a = s.t[-1] - 2 * cl
        panels.append((f"{s.label}: {cl:.0f} ms from {a:.0f} ms", dict(t=s.t, top=s.vm["top"]), a, a + cl))
    if post and ctx.by_name[post].vm is not None:
        s = ctx.by_name[post]
        panels.append((f"{s.label}: first {cl:.0f} ms", dict(t=s.t, top=s.vm["top"]), s.t[0], s.t[0] + cl))
    if not panels:
        raise RuntimeError("no activation windows available")
    g = ctx.pre_grid()
    cols = 2
    rows = int(np.ceil(len(panels) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(WIDTH, 3.0 * rows + 0.3), constrained_layout=True,
                             squeeze=False)
    stats = {}
    for ax, (title, d, a, b) in zip(axes.ravel(), panels):
        lat = first_crossing(d["top"], d["t"], a, b) - a
        masked = np.ma.masked_invalid(lat)
        ax.set_facecolor(GRID)
        im = ax.pcolormesh(g.xs, g.ys, masked, cmap=SEQ, shading="nearest", rasterized=True,
                           vmin=0, vmax=b - a)
        if np.isfinite(lat).sum() > 4 and np.nanmax(lat) > 10:
            ax.contour(g.xs, g.ys, np.where(np.isfinite(lat), lat, np.nan),
                       levels=np.arange(10, b - a, 10), colors=SURFACE, linewidths=0.35)
        outline(ax, g, colours=False, colour=INK, lw=0.5)
        ax.set_aspect("equal")
        ax.set_title(title, fontsize=7.5)
        cb = fig.colorbar(im, ax=ax, shrink=0.85, pad=0.01)
        cb.set_label("activation time (ms)")
        stats[title] = dict(activated_fraction=float(np.isfinite(lat).mean()),
                            total_activation_time_ms=fmt(float(np.nanmax(lat)) if np.isfinite(lat).any() else np.nan, 1))
    for ax in axes.ravel()[len(panels):]:
        ax.axis("off")
    ctx.save(fig, number, "activation_maps", "Activation maps",
             "Local activation time (first upward crossing of -20 mV) on the top surface within each "
             "window, relative to the window start; white isochrones every 10 ms; grey = not "
             "activated in the window. Reentrant windows have a line of block or a wrap-around "
             "discontinuity where the latest and earliest times meet.", stats)


@needs_vm
def fig_cycle_length(ctx, number):
    series = probe_series(ctx)
    probe_keys = [4, 5, 0, 7]
    fig, axes = plt.subplots(3, 1, figsize=(WIDTH, 5.4), constrained_layout=True,
                             gridspec_kw=dict(height_ratios=[1.2, 1, 1]))
    markers = ["o", "s", "^", "D"]
    stats = {}
    # (a) activation raster
    ax = axes[0]
    ylabels = []
    row = 0
    for name, d in series:
        for k in probe_keys:
            c = crossings(d["probes"][:, k], d["t"])
            owner = d["owner"][np.clip(np.searchsorted(d["t"], c), 0, len(d["t"]) - 1)]
            for o in pd.unique(owner):
                ax.plot(c[owner == o], np.full((owner == o).sum(), row), "|", ms=6, mew=1.0,
                        color=stage_colour(o))
            tag = "pre" if name == "pre" else stage_label(name)
            ylabels.append(f"{PROBES[k][1].split(' (')[0]} [{tag}]")
            stats.setdefault(name, {})[PROBES[k][0]] = dict(
                n_activations=int(len(c)), last_activation_ms=fmt(float(c[-1]) if len(c) else np.nan, 1))
            row += 1
    ax.set_yticks(range(row))
    ax.set_yticklabels(ylabels, fontsize=6)
    ax.invert_yaxis()
    ax.set_title("Activation times (upward -20 mV crossings)", loc="left")
    panel_label(ax, "a")
    # (b) instantaneous cycle length
    ax = axes[1]
    for name, d in series:
        for mk, k in zip(markers, probe_keys):
            c = crossings(d["probes"][:, k], d["t"])
            if len(c) < 2:
                continue
            owner = d["owner"][np.clip(np.searchsorted(d["t"], c[1:]), 0, len(d["t"]) - 1)]
            for o in pd.unique(owner):
                m = owner == o
                ax.plot(c[1:][m], np.diff(c)[m], mk, ms=3, mfc="none", mew=0.7, color=stage_colour(o))
    ax.set(ylabel="interval (ms)", title="Activation interval at each probe", xlabel="time (ms)")
    ax.legend([Line2D([], [], marker=m, ls="none", mfc="none", color=INK) for m in markers],
              [PROBES[k][1] for k in probe_keys], fontsize=6, ncol=4, loc="upper right")
    panel_label(ax, "b")
    # (c) fraction of the surface activating per 10 ms
    ax = axes[2]
    for name, d in series:
        top = d["top"].reshape(len(d["t"]), -1)
        up = (top[:-1] < THRESH_MV) & (top[1:] >= THRESH_MV)
        frac = up.mean(axis=1)
        edges = np.arange(d["t"][0], d["t"][-1] + 10, 10)
        hist, _ = np.histogram(d["t"][1:], bins=edges, weights=frac)
        mid = edges[:-1] + 5
        owner = d["owner"][np.clip(np.searchsorted(d["t"], mid), 0, len(d["t"]) - 1)]
        for o in pd.unique(owner):
            m = owner == o
            ax.plot(mid[m], hist[m], color=stage_colour(o), lw=0.9)
        active = mid[hist > 0.001]
        stats.setdefault(name, {})["last_surface_activation_ms"] = fmt(float(active[-1]) if active.size else np.nan, 0)
    ax.set(xlabel="time (ms)", ylabel="share of surface\nactivating / 10 ms", title="Global surface activity")
    panel_label(ax, "c")
    for a in axes:
        for t, ls in ((ctx.t_lift, ":"), (3500, "--")):
            if t:
                a.axvline(t, color=INK2, lw=0.5, ls=ls)
    stage_legend(axes[2], [s.name for s in ctx.stages], loc="upper right", ncol=2)
    cl = cycle_window(ctx)
    stats["baseline_cycle_length_ms_bundle_middle_last_1000ms"] = fmt(cl, 1) if cl else "n/a"
    post_last = {k: v.get("last_surface_activation_ms") for k, v in stats.items() if isinstance(v, dict) and k != "pre"}
    ctx.save(fig, number, "cycle_length_activity", "Activation sequence, cycle length and termination",
             "(a) Activation times at four probes for the pre-ablation chain and the post-ablation "
             "branch. (b) Interval between successive activations; 200 ms during pacing. (c) Share "
             "of top-surface nodes activating in each 10 ms bin. Median baseline cycle length at "
             f"the bundle middle (last 1000 ms of the baseline stage): {fmt(cl, 1) if cl else 'n/a'} ms. "
             f"Last surface activation in each post-ablation branch: {post_last or 'n/a'} ms. "
             "Activity ending well before the stage end indicates termination.", stats)


@needs_vm
def fig_conduction_velocity(ctx, number):
    pre_chain = ctx.pre_chain()
    pacing = pacing_times(ctx)
    if pre_chain is None or pre_chain["t"][0] > pacing[0] + 1:
        raise RuntimeError("needs s1 from t = 0")
    g = ctx.pre_grid()
    a, b = pacing[0], pacing[0] + 200
    stim = stim_box(ctx)
    sx = np.mean([stim[0][0], stim[1][0]]) if stim else 8.5
    sy = np.mean([stim[0][1], stim[1][1]]) if stim else 0.0
    out, results = {}, {}
    planes = [("deep", "wall (z = {:g} mm)".format(g.plane_z()["deep"]), [1]),
              ("mid", "bundle (z = {:g} mm)".format(g.plane_z()["mid"]), [2])]
    fig, axes = plt.subplots(2, 2, figsize=(WIDTH, 5.0), constrained_layout=True,
                             gridspec_kw=dict(width_ratios=[1.4, 1]))
    iz = {"deep": g.iz(g.plane_z()["deep"]), "mid": g.iz(g.plane_z()["mid"])}
    for row, (plane, title, tags) in enumerate(planes):
        lat = first_crossing(pre_chain[plane], pre_chain["t"], a, b)
        gy, gx = np.gradient(lat, g.h)
        grad = np.hypot(gx, gy)
        with np.errstate(divide="ignore", invalid="ignore"):
            speed = 1.0 / grad  # mm/ms = m/s
        X, Y = np.meshgrid(g.xs, g.ys)
        region = np.isin(g.ntag[iz[plane]], tags)
        # interior of the region only (all 4 neighbours in region), away from stimulus and edges
        inner = region.copy()
        inner[1:-1, 1:-1] &= region[2:, 1:-1] & region[:-2, 1:-1] & region[1:-1, 2:] & region[1:-1, :-2]
        inner[[0, -1], :] = inner[:, [0, -1]] = False
        ok = inner & np.isfinite(speed) & (np.hypot(X - sx, Y - sy) > 2.5) & (speed < 3.0) & (speed > 0.02)
        ang = np.degrees(np.arctan2(np.abs(gy), np.abs(gx)))
        cv_l, cv_t = speed[ok & (ang < 20)], speed[ok & (ang > 70)]
        ax = axes[row, 0]
        im = ax.pcolormesh(g.xs, g.ys, np.ma.masked_where(~ok, speed), cmap=SEQ, shading="nearest",
                           rasterized=True, vmin=0, vmax=np.nanpercentile(speed[ok], 98) if ok.any() else 1)
        ax.set_facecolor(GRID)
        outline(ax, g, colours=False, colour=INK, lw=0.4)
        ax.set_aspect("equal")
        ax.set_title(f"Local conduction speed, beat 1: {title}", fontsize=7.5)
        fig.colorbar(im, ax=ax, shrink=0.85, pad=0.01).set_label("speed (m/s)")
        ax = axes[row, 1]
        bins = np.linspace(0, max(1.2, np.nanpercentile(speed[ok], 99) if ok.any() else 1.2), 40)
        for vals, lab, c in ((cv_l, "along fibres (x)", STAGE_COLOURS["s3_reentry"]),
                             (cv_t, "across fibres (y)", STAGE_COLOURS["s4_ablate_terminate"])):
            if len(vals):
                ax.hist(vals, bins=bins, color=c, alpha=0.75, label=f"{lab}: median {np.median(vals):.2f}",
                        rwidth=0.9)
        ax.set(xlabel="speed (m/s)", ylabel="nodes", title=title)
        ax.legend(fontsize=6.3, loc="upper right")
        med_l = float(np.median(cv_l)) if len(cv_l) else np.nan
        med_t = float(np.median(cv_t)) if len(cv_t) else np.nan
        results[plane] = dict(cv_longitudinal_m_s=fmt(med_l), cv_transverse_m_s=fmt(med_t),
                              ratio_l_t=fmt(med_l / med_t if med_t else np.nan), n_l=int(len(cv_l)),
                              n_t=int(len(cv_t)))
        out[plane] = (med_l, med_t)
    wl, wt = out["deep"]
    ctx.save(fig, number, "conduction_velocity", "Conduction velocity",
             "Local conduction speed (1/|grad LAT|) for the first paced beat, on a wall plane below the "
             "bundle and on a plane through the bundle; only region interiors >2.5 mm from the "
             "stimulus are shown. Histograms split nodes whose propagation is within 20 degrees of the "
             f"fibre (x) or cross-fibre (y) direction. Wall: {fmt(wl)} m/s along, {fmt(wt)} m/s across "
             f"(ratio {fmt(wl / wt if wt else np.nan)}; the paper's ratio is 1.36). Gradient estimates "
             "are biased near wavefront collisions and boundaries; medians are reported.", results)


# ======================================================================
# C. Electrograms
# ======================================================================
def needs_points(fn):
    def wrapper(ctx, number, *args):
        if ctx.points is None:
            raise RuntimeError("points.csv missing (run egm.py)")
        return fn(ctx, number, *args)

    wrapper.__name__ = fn.__name__
    return wrapper


def compare_stages(ctx):
    pre, post = ctx.compare_pair()
    return [n for n in (pre, post) if n]


@needs_points
def fig_egm_grid(ctx, number):
    names = compare_stages(ctx)
    gap = min(np.unique(ctx.points.gap_mm)) if "gap_mm" in ctx.points else 0.0
    fig, axes = plt.subplots(4, 4, figsize=(WIDTH, 5.0), sharex=True, sharey=True, constrained_layout=True)
    for name in names:
        s = ctx.by_name[name]
        p = ctx.placement(s, (0, 0), gap)
        for e in range(16):
            t, v = ctx.trace(s, p, "uni", f"E{e}")
            ax = axes[e // 4, e % 4]
            ax.plot(t - t[0], v, color=s.colour, lw=0.8)
            ax.text(0.03, 0.92, f"E{e}", transform=ax.transAxes, fontsize=6.5, va="top")
    for ax in axes[-1]:
        ax.set_xlabel("time in window (ms)")
    for ax in axes[:, 0]:
        ax.set_ylabel("mV")
    stage_legend(fig, names, loc="lower center", bbox_to_anchor=(0.5, -0.04))
    ctx.save(fig, number, "egm_grid_unipolar", "Unipolar electrograms on the HD Grid",
             f"Unipolar electrograms of all 16 electrodes at the central placement (catheter centre "
             f"(0, 0), gap {gap:g} mm), over the analysis window, before and after ablation. Row = "
             "across-spline position (y), column = along-spline position (x), as electrode "
             f"e = row x 4 + col. Traces are gain-scaled (x{ctx.gain:.3g}) and filtered as in egm.py "
             "but shown without the added measurement noise.")


@needs_points
def fig_egm_examples(ctx, number):
    names = compare_stages(ctx)
    centres = ctx.cfg["groundtruth"].get("map_xy_mm_pre") or [[0, 0]]
    gaps = sorted(np.unique(ctx.points.gap_mm)) if "gap_mm" in ctx.points else [0.0]
    fig, axes = plt.subplots(len(centres), 3, figsize=(WIDTH, 1.55 * len(centres) + 0.5),
                             sharex=True, constrained_layout=True, squeeze=False)
    for i, c in enumerate(centres):
        for j, kind in enumerate(KINDS):
            ax = axes[i, j]
            for name in names:
                s = ctx.by_name[name]
                for gap in gaps:
                    p = ctx.placement(s, c, gap)
                    rows = ctx.points[(ctx.points.stage_name == name) & (ctx.points.placement == p)
                                      & (ctx.points.kind == kind)]
                    if rows.empty:
                        continue
                    r = rows.iloc[np.argmin(np.hypot(rows.x_mm - c[0], rows.y_mm - c[1]))]
                    t, v = ctx.trace(s, p, kind, r.label)
                    ax.plot(t - t[0], v, color=s.colour, lw=0.8, ls=GAP_STYLE.get(float(gap), "-"))
            if i == 0:
                ax.set_title(KINDS[kind])
            if j == 0:
                ax.set_ylabel(f"centre ({c[0]:g}, {c[1]:g})\nmV")
    for ax in axes[-1]:
        ax.set_xlabel("time in window (ms)")
    handles = [Line2D([], [], color=stage_colour(n), lw=2) for n in names] + [
        Line2D([], [], color=INK, ls=GAP_STYLE.get(float(g), "-")) for g in gaps]
    fig.legend(handles, [stage_label(n) for n in names] + [f"gap {g:g} mm" for g in gaps],
               loc="lower center", ncol=len(handles), bbox_to_anchor=(0.5, -0.03))
    ctx.save(fig, number, "egm_examples", "Unipolar, bipolar and omnipolar electrograms",
             "The recording of each kind closest to the catheter centre, at the contribution-map "
             "placements (rows), before and after ablation and at both electrode gaps. Bipolar = "
             "max-orthogonal pair; omnipolar = clique direction maximising peak-to-peak. Noise-free, "
             "gain-scaled, filtered as in egm.py.")


def site_medians(df):
    d = df.assign(xr=df.x_mm.round(3), yr=df.y_mm.round(3))
    return d.groupby(["xr", "yr"], as_index=False).PP_mV.median()


@needs_points
def fig_voltage_maps(ctx, number, gap):
    df = ctx.points[ctx.points.gap_mm == gap] if "gap_mm" in ctx.points else ctx.points
    names = [s.name for s in ctx.stages if s.name in set(df.stage_name)]
    fig, axes = plt.subplots(3, len(names), figsize=(WIDTH, WIDTH / max(len(names), 1) * 3 * 0.92 + 0.5),
                             sharex=True, sharey=True,
                             constrained_layout=True, squeeze=False)
    stats = {}
    for i, kind in enumerate(KINDS):
        dk = df[df.kind == kind]
        pos = dk.PP_mV[dk.PP_mV > 0]
        if pos.empty:
            continue
        norm = LogNorm(max(pos.quantile(0.01), 1e-4), pos.quantile(0.995))
        sc = None
        for j, name in enumerate(names):
            ax = axes[i, j]
            m = site_medians(dk[dk.stage_name == name])
            spacing = {"uni": 1.0, "bi": 0.67, "omni": 0.67}[kind]
            size = (spacing * 72 * 1.9 / 30) ** 2 * 4
            sc = ax.scatter(m.xr, m.yr, c=np.maximum(m.PP_mV, norm.vmin), s=size, marker="s", cmap=SEQ,
                            norm=norm, linewidths=0, rasterized=True)
            outline(ax, ctx.grid(ctx.by_name[name]), colours=False, colour=INK, lw=0.5)
            ax.set_aspect("equal")
            if i == 0:
                ax.set_title(stage_label(name), fontsize=7.5)
            if j == 0:
                ax.set_ylabel(f"{KINDS[kind]}\ny (mm)")
            if i == 2:
                ax.set_xlabel("x (mm)")
            stats.setdefault(name, {})[kind] = fmt(float(m.PP_mV.median()), 3)
        fig.colorbar(sc, ax=axes[i, :], shrink=0.9, pad=0.01).set_label("PP (mV, log)")
    ctx.save(fig, number, f"voltage_maps_gap{gap:g}", f"Voltage maps, gap {gap:g} mm",
             f"Peak-to-peak voltage (median over all placements recording at a site) at gap {gap:g} mm "
             "for each stage (columns) and recording kind (rows), on a log scale shared within a row. "
             "Each square is a recording site: unipoles at electrode positions, bipoles and omnipoles "
             "at the centroid of their electrodes. Outlines: bundle, sheath and lesion. Includes the "
             "simulated measurement noise.", dict(median_PP_mV=stats))


@needs_points
def fig_ratio_map(ctx, number):
    pre, post = ctx.compare_pair()
    if not (pre and post):
        raise RuntimeError("needs a baseline and a post-ablation stage")
    gaps = sorted(np.unique(ctx.points.gap_mm))
    fig, axes = plt.subplots(len(gaps), 3, figsize=(WIDTH, 2.1 * len(gaps) + 0.5), sharex=True, sharey=True,
                             constrained_layout=True, squeeze=False)
    stats = {}
    sc = None
    for i, gap in enumerate(gaps):
        for j, kind in enumerate(KINDS):
            d = ctx.points[(ctx.points.gap_mm == gap) & (ctx.points.kind == kind)]
            a = site_medians(d[d.stage_name == pre]).set_index(["xr", "yr"]).PP_mV
            b = site_medians(d[d.stage_name == post]).set_index(["xr", "yr"]).PP_mV
            r = np.log2(b / a).replace([np.inf, -np.inf], np.nan).dropna()
            ax = axes[i, j]
            sc = ax.scatter(r.index.get_level_values(0), r.index.get_level_values(1), c=r.values,
                            s=9, marker="s", cmap=DIV, norm=TwoSlopeNorm(0, -3, 3), linewidths=0,
                            rasterized=True)
            outline(ax, ctx.post_grid() or ctx.pre_grid(), colours=False, colour=INK, lw=0.5)
            ax.set_aspect("equal")
            if i == 0:
                ax.set_title(KINDS[kind])
            if j == 0:
                ax.set_ylabel(f"gap {gap:g} mm\ny (mm)")
            if i == len(gaps) - 1:
                ax.set_xlabel("x (mm)")
            stats[f"{kind}_gap{gap:g}"] = dict(median_log2_ratio=fmt(float(r.median())),
                                               share_below_half=fmt(float((r < -1).mean())))
    fig.colorbar(sc, ax=axes, shrink=0.8, pad=0.01).set_label(f"log2 PP ratio ({stage_label(post)} / {stage_label(pre)})")
    ctx.save(fig, number, "pp_ratio_map", "Change in voltage after ablation",
             f"log2 of the peak-to-peak ratio {stage_label(post)} / {stage_label(pre)} at every recording "
             "site (-1 = halved, +1 = doubled; clipped at +/-3). Blue = lower after ablation. The two "
             "stages are in different electrical states (activation pattern as well as substrate "
             "differ), so the ratio is not a pure substrate effect.", stats)


def binned(x, y, edges):
    mid, med, q1, q3 = [], [], [], []
    for a, b in zip(edges[:-1], edges[1:]):
        v = y[(x >= a) & (x < b)]
        if len(v) >= 3:
            mid.append((a + b) / 2)
            med.append(np.median(v))
            q1.append(np.percentile(v, 25))
            q3.append(np.percentile(v, 75))
    return map(np.array, (mid, med, q1, q3))


@needs_points
def fig_pp_vs_distance(ctx, number):
    gaps = sorted(np.unique(ctx.points.gap_mm))
    names = [s.name for s in ctx.stages]
    fig, axes = plt.subplots(len(gaps), 3, figsize=(WIDTH, 2.2 * len(gaps) + 0.5), sharex=True,
                             constrained_layout=True, squeeze=False)
    edges = np.arange(-3.0, 16.0, 1.0)
    stats = {}
    for i, gap in enumerate(gaps):
        for j, kind in enumerate(KINDS):
            ax = axes[i, j]
            for name in names:
                d = ctx.points[(ctx.points.gap_mm == gap) & (ctx.points.kind == kind) & (ctx.points.stage_name == name)]
                if d.empty:
                    continue
                mid, med, q1, q3 = binned(-d.d_mm.to_numpy(), d.PP_mV.to_numpy(), edges)
                ax.fill_between(mid, q1, q3, color=stage_colour(name), alpha=0.15, lw=0)
                ax.plot(mid, med, color=stage_colour(name), lw=1.4)
                near = d[(-d.d_mm) < 2].PP_mV
                stats.setdefault(name, {})[f"{kind}_gap{gap:g}_median_within_2mm_of_lesion_edge"] = fmt(float(near.median()) if len(near) else np.nan, 3)
            ax.axvline(0, color=INK2, lw=0.5, ls=":")
            ax.set_yscale("log")
            if i == 0:
                ax.set_title(KINDS[kind])
            if j == 0:
                ax.set_ylabel(f"gap {gap:g} mm\nPP (mV)")
            if i == len(gaps) - 1:
                ax.set_xlabel("distance outside lesion edge (mm)")
    stage_legend(fig, names, loc="lower center", bbox_to_anchor=(0.5, -0.04))
    ctx.save(fig, number, "pp_vs_distance", "Voltage against distance from the lesion",
             "Median (line) and interquartile range (band) of peak-to-peak voltage in 1 mm bins of "
             "the recording site's distance outside the lesion boundary (negative = above the "
             "lesion; the dotted line is the boundary, radius 2.5 mm). Pre-ablation stages use the "
             "same geometric distance to where the lesion will be.", stats)


@needs_points
def fig_gap_effect(ctx, number):
    gaps = sorted(np.unique(ctx.points.gap_mm))
    if len(gaps) < 2:
        raise RuntimeError("needs two gaps")
    g0, g1 = gaps[0], gaps[-1]
    names = compare_stages(ctx) or [s.name for s in ctx.stages][:3]
    fig, axes = plt.subplots(1, 3, figsize=(WIDTH, 2.4), constrained_layout=True)
    stats = {}
    for ax, kind in zip(axes, KINDS):
        lo, hi = np.inf, 0
        for name in names:
            d = ctx.points[(ctx.points.kind == kind) & (ctx.points.stage_name == name)].assign(
                xr=lambda f: f.x_mm.round(3), yr=lambda f: f.y_mm.round(3))
            keys = ["label", "xr", "yr"]
            a = d[d.gap_mm == g0].groupby(keys).PP_mV.median()
            b = d[d.gap_mm == g1].groupby(keys).PP_mV.median()
            j = pd.concat([a.rename("a"), b.rename("b")], axis=1).dropna()
            j = j[(j.a > 0) & (j.b > 0)]
            ax.scatter(j.a, j.b, s=4, color=stage_colour(name), alpha=0.5, linewidths=0, rasterized=True)
            lo, hi = min(lo, j.min().min()), max(hi, j.max().max())
            stats.setdefault(name, {})[kind] = fmt(float(np.median(j.b / j.a)), 3)
        if np.isfinite(lo):
            ax.plot([lo, hi], [lo, hi], color=INK2, lw=0.6, ls=":")
        ax.set(xscale="log", yscale="log", title=KINDS[kind], xlabel=f"PP at gap {g0:g} mm (mV)")
        ax.text(0.04, 0.96, "\n".join(f"{stage_label(n)}: x{stats[n][kind]}" for n in names),
                transform=ax.transAxes, fontsize=6.3, va="top")
    axes[0].set_ylabel(f"PP at gap {g1:g} mm (mV)")
    stage_legend(fig, names, loc="lower center", bbox_to_anchor=(0.5, -0.07))
    ctx.save(fig, number, "gap_effect", "Effect of electrode-tissue distance",
             f"Peak-to-peak voltage at the same recording site with the catheter at {g1:g} mm against "
             f"{g0:g} mm. Dotted line = identity; the text gives the median ratio (gap {g1:g} / {g0:g}). "
             "Ratios below 1 show the attenuation with distance; differences between kinds reflect "
             "how far-field each recording type is.", dict(median_ratio=stats))


@needs_points
def fig_pp_distributions(ctx, number):
    names = [s.name for s in ctx.stages]
    gaps = sorted(np.unique(ctx.points.gap_mm))
    fig, axes = plt.subplots(1, 3, figsize=(WIDTH, 2.6), sharey=True, constrained_layout=True)
    rows = []
    for ax, kind in zip(axes, KINDS):
        pos = 0
        ticks = []
        for name in names:
            for k, gap in enumerate(gaps):
                v = ctx.points[(ctx.points.kind == kind) & (ctx.points.stage_name == name) & (ctx.points.gap_mm == gap)].PP_mV
                v = v[v > 0]
                if v.empty:
                    continue
                bp = ax.boxplot([v], positions=[pos], widths=0.7, patch_artist=True, showfliers=False,
                                medianprops=dict(color=INK, lw=1.0), whiskerprops=dict(color=MUTED, lw=0.6),
                                capprops=dict(color=MUTED, lw=0.6))
                for b in bp["boxes"]:
                    b.set(facecolor=stage_colour(name) if k == 0 else SURFACE, edgecolor=stage_colour(name), lw=0.9)
                rows.append(dict(stage=name, kind=kind, gap_mm=gap, n=len(v), median=v.median(),
                                 q1=v.quantile(0.25), q3=v.quantile(0.75)))
                pos += 1
            ticks.append(pos - len(gaps) / 2 - 0.5)
            pos += 0.6
        ax.set_xticks(ticks)
        ax.set_xticklabels([n.split("_")[0] for n in names])
        ax.set(yscale="log", title=KINDS[kind])
    axes[0].set_ylabel("PP (mV)")
    fig.legend([Patch(fc=MUTED, ec=MUTED), Patch(fc=SURFACE, ec=MUTED)], [f"gap {g:g} mm" for g in gaps],
               loc="lower center", ncol=2, bbox_to_anchor=(0.5, -0.06))
    table = pd.DataFrame(rows)
    table.to_csv(ctx.tables / "pp_summary.csv", index=False)
    write_tex(ctx.tables / "pp_summary.tex", table, "Peak-to-peak voltage (mV): median [IQR]", "median")
    ctx.save(fig, number, "pp_distributions", "Voltage distributions",
             "Peak-to-peak voltage of every recording by stage (s1-s4) and kind; filled boxes = "
             f"gap {gaps[0]:g} mm, open boxes = gap {gaps[-1]:g} mm. Boxes are the interquartile range, "
             "whiskers 1.5 IQR. The auto gain is calibrated so the median baseline bipolar PP is "
             f"{ctx.cfg['gain'].get('target_bipolar_healthy_mV', 'n/a')} mV. Table: tables/pp_summary.*")


@needs_points
def fig_peak_frequency(ctx, number):
    d = ctx.points[ctx.points.kind != "uni"]
    names = [s.name for s in ctx.stages]
    fig, axes = plt.subplots(1, 3, figsize=(WIDTH, 2.5), constrained_layout=True,
                             gridspec_kw=dict(width_ratios=[1, 1, 0.8]))
    stats = {}
    band = ctx.cfg["peak_freq"]["band_hz"] or ctx.cfg["filters"]["differential_hz"]
    bins = np.linspace(band[0], band[1], 45)
    for ax, kind in zip(axes[:2], ("bi", "omni")):
        for name in names:
            v = d[(d.kind == kind) & (d.stage_name == name) & (d.PF_Hz > 0)].PF_Hz
            if len(v):
                ax.hist(v, bins=bins, histtype="step", color=stage_colour(name), lw=1.2, density=True)
                stats.setdefault(name, {})[f"{kind}_median_PF_Hz"] = fmt(float(v.median()), 0)
        ax.set(title=KINDS[kind], xlabel="peak frequency (Hz)")
    axes[0].set_ylabel("density")
    ax = axes[2]
    x = np.arange(len(names))
    for k, kind in enumerate(("bi", "omni")):
        share = [float((d[(d.kind == kind) & (d.stage_name == n)].PF_Hz == 0).mean()) for n in names]
        ax.bar(x + (k - 0.5) * 0.38, share, width=0.36, color=[stage_colour(n) for n in names],
               hatch=None if k == 0 else "////", edgecolor=SURFACE, lw=0.5)
        for n, s in zip(names, share):
            stats.setdefault(n, {})[f"{kind}_undetectable"] = fmt(s, 3)
    ax.set_xticks(x)
    ax.set_xticklabels([n.split("_")[0] for n in names])
    ax.set(ylabel="share undetectable", title="PF undetectable", ylim=(0, 1))
    ax.legend([Patch(fc=MUTED), Patch(fc=MUTED, hatch="////", ec=SURFACE)], ["bipolar", "omnipolar"], fontsize=6.3)
    stage_legend(fig, names, loc="lower center", bbox_to_anchor=(0.5, -0.07))
    ctx.save(fig, number, "peak_frequency", "Peak frequency",
             "Distribution of the peak (dominant) frequency of bipolar and omnipolar recordings within "
             f"a {ctx.cfg['peak_freq']['window_ms']} ms window around the largest deflection, search band "
             f"{band[0]}-{band[1]} Hz. Recordings below {ctx.cfg['peak_freq']['min_snr']} x the noise-only PP "
             "are undetectable (right).", stats)


# ======================================================================
# D. Ground truth
# ======================================================================
def needs_gt(fn):
    def wrapper(ctx, number, *args):
        if ctx.gt is None or ctx.gt.empty:
            raise RuntimeError("groundtruth.csv missing (run ground_truth.py)")
        return fn(ctx, number, *args)

    wrapper.__name__ = fn.__name__
    return wrapper


@needs_gt
def fig_fov_box(ctx, number):
    gt = ctx.gt
    names = [s.name for s in ctx.stages if s.name in set(gt.stage_name)]
    gaps = sorted(np.unique(gt.gap_mm)) if "gap_mm" in gt else [0.0]
    fig, axes = plt.subplots(1, 2, figsize=(WIDTH, 2.8), sharey=True, constrained_layout=True)
    rows = []
    rng = np.random.default_rng(0)
    for ax, (col, title) in zip(axes, (("fov80_res_mm", "Reconstruction radius (fov80_res)"),
                                       ("fov80_node_mm", "Energy radius (fov80_node)"))):
        pos, ticks = 0, []
        for kind in KINDS:
            start = pos
            for name in names:
                for k, gap in enumerate(gaps):
                    v = gt[(gt.kind == kind) & (gt.stage_name == name) & (gt.gap_mm == gap)][col].dropna()
                    if v.empty:
                        pos += 1
                        continue
                    bp = ax.boxplot([v], positions=[pos], widths=0.75, patch_artist=True, showfliers=False,
                                    medianprops=dict(color=INK, lw=1.0), whiskerprops=dict(color=MUTED, lw=0.6),
                                    capprops=dict(color=MUTED, lw=0.6))
                    for b in bp["boxes"]:
                        b.set(facecolor=stage_colour(name) if k == 0 else SURFACE,
                              edgecolor=stage_colour(name), lw=0.9)
                    ax.scatter(pos + rng.uniform(-0.2, 0.2, len(v)), v, s=2, color=INK2, alpha=0.4, linewidths=0)
                    rows.append(dict(metric=col, kind=kind, stage=name, gap_mm=gap, n=len(v), median=v.median(),
                                     q1=v.quantile(0.25), q3=v.quantile(0.75)))
                    pos += 1
            ticks.append((start + pos - 1) / 2)
            pos += 1
        ax.set_xticks(ticks)
        ax.set_xticklabels(list(KINDS.values()))
        ax.set_title(title)
        ax.grid(axis="y")
    axes[0].set_ylabel("radius holding 80% (mm)")
    handles = [Patch(fc=stage_colour(n), ec=stage_colour(n), label=stage_label(n)) for n in names] + [
        Patch(fc=MUTED, ec=MUTED, label=f"gap {gaps[0]:g} mm (filled)"),
        Patch(fc=SURFACE, ec=MUTED, label=f"gap {gaps[-1]:g} mm (open)")]
    fig.legend(handles=handles, loc="lower center", ncol=min(6, len(handles)), bbox_to_anchor=(0.5, -0.1))
    table = pd.DataFrame(rows)
    table.to_csv(ctx.tables / "fov80_summary.csv", index=False)
    write_tex(ctx.tables / "fov80_summary.tex", table, "Field-of-view radius (mm): median [IQR]", "median")
    med = table.groupby(["metric", "kind"]).median(numeric_only=True)["median"].round(2).to_dict()
    ctx.save(fig, number, "fov80_by_kind", "Ground-truth field of view by recording kind",
             "Lateral radius around each recording site that holds 80% of its signal, from the "
             "lead-field ground truth. Left: smallest radius beyond which rebuilding the signal from "
             "the enclosed tissue always leaves <= 20% of its energy unexplained. Right: radius "
             "holding 80% of the summed per-node contribution energy. Points = individual "
             "recordings. Table: tables/fov80_summary.*",
             {f"{m}|{k}": v for (m, k), v in med.items()})


@needs_gt
def fig_fov_curves(ctx, number):
    if ctx.curves is None:
        raise RuntimeError("groundtruth_curves.npz missing (re-run ground_truth.py)")
    c = ctx.curves
    run = c["run"].astype(str)
    gap_of = ctx.gt.drop_duplicates(["run", "placement"]).set_index(["run", "placement"]).gap_mm.to_dict() \
        if "gap_mm" in ctx.gt else {}
    gaps = np.array([gap_of.get((r, int(p)), 0.0) for r, p in zip(run, c["placement"])])
    stage = np.array([r.split("/")[-1] for r in run])
    names = [s.name for s in ctx.stages if s.name in set(stage)]
    ug = sorted(np.unique(gaps))
    R = c["radius_mm"]
    fig, axes = plt.subplots(len(ug), 3, figsize=(WIDTH, 2.1 * len(ug) + 0.5), sharex=True, sharey=True,
                             constrained_layout=True, squeeze=False)
    stats = {}
    for i, gap in enumerate(ug):
        for j, kind in enumerate(KINDS):
            ax = axes[i, j]
            for name in names:
                m = (c["kind"] == kind) & (stage == name) & (gaps == gap)
                if not m.any():
                    continue
                y = np.clip(c["explained"][m], 0, 1)
                ax.fill_between(R, np.percentile(y, 25, axis=0), np.percentile(y, 75, axis=0),
                                color=stage_colour(name), alpha=0.15, lw=0)
                med = np.median(y, axis=0)
                ax.plot(R, med, color=stage_colour(name), lw=1.4)
                r80 = R[np.argmax(med >= 0.8)] if (med >= 0.8).any() else np.nan
                stats[f"{name}|{kind}|gap{gap:g}"] = fmt(float(r80), 2)
            ax.axhline(0.8, color=INK2, lw=0.5, ls=":")
            ax.set_xlim(0, min(R[-1], 15))
            ax.set_ylim(0, 1.02)
            if i == 0:
                ax.set_title(KINDS[kind])
            if j == 0:
                ax.set_ylabel(f"gap {gap:g} mm\nsignal energy explained")
            if i == len(ug) - 1:
                ax.set_xlabel("radius around recording site (mm)")
    stage_legend(fig, names, loc="lower center", bbox_to_anchor=(0.5, -0.04))
    ctx.save(fig, number, "fov_curves", "Field-of-view curves",
             "Share of each recording's signal energy reproduced by tissue within a lateral radius "
             "of the recording site (1 - normalised residual), median and IQR over recordings. The "
             "radius where the median reaches 0.8 (dotted) is the typical field of view; values can "
             "dip below 0 before the radius covers the dominant sources, so the axis is clipped. "
             "Median crossing radii (mm) are listed in figure_stats.json.", stats)


@needs_gt
def fig_region_shares(ctx, number):
    gt = ctx.gt
    tags = sorted(int(c[9:]) for c in gt if c.startswith("share_tag"))
    names = [s.name for s in ctx.stages if s.name in set(gt.stage_name)]
    gaps = sorted(np.unique(gt.gap_mm)) if "gap_mm" in gt else [0.0]
    rows = []
    for name in names:
        for kind in KINDS:
            for gap in gaps:
                d = gt[(gt.stage_name == name) & (gt.kind == kind) & (gt.gap_mm == gap)]
                if d.empty:
                    continue
                means = {t: float(d[f"share_tag{t}"].fillna(0).mean()) for t in tags}
                rows.append(dict(stage=name, kind=kind, gap_mm=gap, **{f"tag{t}": v for t, v in means.items()}))
    table = pd.DataFrame(rows)
    table.to_csv(ctx.tables / "region_shares.csv", index=False)
    fig, ax = plt.subplots(figsize=(WIDTH, 0.22 * len(table) + 1.1), constrained_layout=True)
    y = np.arange(len(table))[::-1]
    left = np.zeros(len(table))
    for t in tags:
        v = table[f"tag{t}"].to_numpy()
        ax.barh(y, v, left=left, height=0.75, color=TAGS.get(t, (str(t), MUTED))[1], edgecolor=SURFACE,
                lw=1.0, label=f"{TAGS.get(t, (f'tag {t}',))[0]}")
        left += v
    ax.set_yticks(y)
    ax.set_yticklabels([f"{r.stage.split('_')[0]}  {KINDS[r.kind]:<9}  gap {r.gap_mm:g}" for r in table.itertuples()],
                       fontsize=6.3, family="monospace")
    ax.set(xlim=(0, 1), xlabel="mean share of contribution energy")
    ax.legend(loc="lower center", ncol=len(tags), bbox_to_anchor=(0.5, 1.0))
    ctx.save(fig, number, "region_shares", "Which tissue each recording sees",
             "Mean share of each recording's contribution energy coming from each mesh region (wall, "
             "bundle, exit segment, sheath, lesion), from the ground-truth placements. The bundle "
             "is a small share of tissue volume; a larger energy share means the recording is "
             "dominated by bundle activity. Table: tables/region_shares.csv",
             {f"{r['stage']}|{r['kind']}|gap{r['gap_mm']:g}": {k: round(v, 3) for k, v in r.items() if k.startswith("tag")}
              for r in rows})


@needs_gt
def fig_gt_validation(ctx, number):
    u = ctx.gt[ctx.gt.kind == "uni"]
    names = [s.name for s in ctx.stages if s.name in set(u.stage_name)]
    fig, axes = plt.subplots(1, 2, figsize=(WIDTH, 2.4), constrained_layout=True)
    stats = {}
    for name in names:
        d = u[u.stage_name == name]
        r = d.r_vs_recovery.abs().dropna()
        sc = d.scale_vs_recovery.dropna()
        if len(r):
            axes[0].hist(r, bins=np.linspace(max(0.0, r.min() - 0.01), 1, 30), histtype="step",
                         color=stage_colour(name), lw=1.2)
        if len(sc):
            axes[1].hist(sc, bins=30, histtype="step", color=stage_colour(name), lw=1.2)
        stats[name] = dict(abs_r_median=fmt(float(r.median()) if len(r) else np.nan, 4),
                           abs_r_min=fmt(float(r.min()) if len(r) else np.nan, 4),
                           scale_cv=fmt(float(sc.std() / sc.mean()) if len(sc) > 1 else np.nan, 3))
    axes[0].axvline(0.95, color=INK2, lw=0.6, ls=":")
    axes[0].set(xlabel="|r| reconstruction vs openCARP recovery", ylabel="unipolar recordings",
                title="Waveform agreement")
    axes[1].set(xlabel="PP(recovery) / PP(reconstruction)", title="Amplitude scale")
    stage_legend(fig, names, loc="lower center", bbox_to_anchor=(0.5, -0.08))
    ctx.save(fig, number, "groundtruth_validation", "Validation of the ground-truth reconstruction",
             "Agreement between the regular-grid lead-field reconstruction and openCARP's own phie "
             "recovery for every unipolar recording at the ground-truth placements. Left: absolute "
             "correlation of the waveforms (dotted: 0.95 warning threshold). Right: amplitude ratio; "
             "a narrow distribution means the ground truth differs from openCARP only by a constant "
             "scale. Flat (quiescent) signals have no defined correlation and are omitted.", stats)


@needs_gt
def fig_fov_vs_position(ctx, number):
    gt = ctx.gt
    names = [s.name for s in ctx.stages if s.name in set(gt.stage_name)]
    fig, axes = plt.subplots(1, 3, figsize=(WIDTH, 2.5), sharey=True, constrained_layout=True)
    markers = {g: m for g, m in zip(sorted(np.unique(gt.gap_mm)), ["o", "^", "s"])} if "gap_mm" in gt else {0.0: "o"}
    for ax, kind in zip(axes, KINDS):
        for name in names:
            for gap, mk in markers.items():
                d = gt[(gt.kind == kind) & (gt.stage_name == name) & (gt.gap_mm == gap)]
                ax.scatter(-d.d_mm, d.fov80_res_mm, s=7, marker=mk, facecolors="none",
                           edgecolors=stage_colour(name), linewidths=0.6, alpha=0.8)
        ax.axvline(0, color=INK2, lw=0.5, ls=":")
        ax.set(title=KINDS[kind], xlabel="distance outside lesion edge (mm)")
    axes[0].set_ylabel("fov80_res (mm)")
    handles = [Line2D([], [], color=stage_colour(n), lw=2) for n in names] + [
        Line2D([], [], marker=m, ls="none", mfc="none", color=INK) for m in markers.values()]
    fig.legend(handles, [stage_label(n) for n in names] + [f"gap {g:g} mm" for g in markers],
               loc="lower center", ncol=len(handles), bbox_to_anchor=(0.5, -0.08))
    ctx.save(fig, number, "fov_vs_position", "Field of view against position",
             "Ground-truth field of view of each recording against its distance outside the lesion "
             "boundary (placements along the x and y axes through the lesion). Changes near the "
             "boundary show whether the recording's field of view depends on the local substrate.")


@needs_gt
def fig_fov_vs_pp(ctx, number):
    path = ctx.results / "points_with_gt.csv"
    if not path.exists():
        raise RuntimeError("points_with_gt.csv missing")
    d = pd.read_csv(path)
    d = d[d.groundtruth_computed.astype(bool)]
    d["stage_name"] = d.run.astype(str).str.split("/").str[-1]
    names = [s.name for s in ctx.stages if s.name in set(d.stage_name)]
    fig, axes = plt.subplots(1, 3, figsize=(WIDTH, 2.5), sharey=True, constrained_layout=True)
    stats = {}
    for ax, kind in zip(axes, KINDS):
        for name in names:
            v = d[(d.kind == kind) & (d.stage_name == name)].dropna(subset=["fov80_res_mm"])
            v = v[v.PP_mV > 0]
            ax.scatter(v.PP_mV, v.fov80_res_mm, s=6, color=stage_colour(name), alpha=0.6, linewidths=0)
            if len(v) > 3:
                rho = v[["PP_mV", "fov80_res_mm"]].corr(method="spearman").iloc[0, 1]
                stats[f"{name}|{kind}"] = fmt(float(rho), 2)
        ax.set(xscale="log", title=KINDS[kind], xlabel="PP (mV)")
    axes[0].set_ylabel("fov80_res (mm)")
    stage_legend(fig, names, loc="lower center", bbox_to_anchor=(0.5, -0.08))
    ctx.save(fig, number, "fov_vs_pp", "Field of view against measured voltage",
             "Ground-truth field of view against the measured peak-to-peak voltage of the same "
             "recording. Spearman correlations per stage and kind are listed in figure_stats.json.",
             dict(spearman_rho=stats))


# ---------------- tables ----------------
def write_tex(path, table, caption, value):
    """Simple LaTeX tabular of median [q1, q3], without pandas' jinja2 dependency."""
    if table.empty:
        return
    keys = [c for c in table.columns if c not in ("n", "median", "q1", "q3")]
    lines = ["\\begin{table}[ht]", "\\centering", f"\\caption{{{caption}}}",
             "\\begin{tabular}{" + "l" * len(keys) + "rr}", "\\hline",
             " & ".join(k.replace("_", "\\_") for k in keys) + " & n & median [IQR] \\\\", "\\hline"]
    for r in table.itertuples(index=False):
        d = r._asdict()
        cells = [str(d[k]).replace("_", "\\_") for k in keys]
        lines.append(" & ".join(cells) + f" & {d['n']} & {d['median']:.3g} [{d['q1']:.3g}, {d['q3']:.3g}] \\\\")
    lines += ["\\hline", "\\end{tabular}", "\\end{table}"]
    Path(path).write_text("\n".join(lines) + "\n")


def write_outputs(ctx, failures):
    with open(ctx.out / "figure_stats.json", "w") as f:
        json.dump(ctx.stats, f, indent=1, default=str)
    maps = sorted((ctx.results / "maps").glob("*.png")) if (ctx.results / "maps").exists() else []
    lines = ["# Thesis figures", "",
             f"Generated {time.strftime('%Y-%m-%d %H:%M')} from `{ctx.results}`.",
             f"Stages: {', '.join(s.name for s in ctx.stages)}. Gain x{ctx.gain:.4g}; "
             f"noise SD {ctx.summary.get('noise_sd_mV', 'n/a')} mV.", "",
             "Each figure exists as PNG (300 dpi) and PDF (vector). Numbers in captions are measured "
             "from this run; check them against the figure before quoting.", ""]
    for c in sorted(ctx.captions, key=lambda c: c["number"]):
        lines += [f"## {c['number']:02d}. {c['title']}", "", f"![{c['title']}]({c['file']}.png)", "",
                  c["caption"], ""]
        if c["stats"]:
            lines += ["<details><summary>Measured values</summary>", "", "```json",
                      json.dumps(c["stats"], indent=1, default=str), "```", "</details>", ""]
    if maps:
        lines += ["## Ground-truth contribution maps (from gt_maps.py)", "",
                  "Per stage: top view (`*_maps.png`, contribution energy summed through the wall, "
                  "black contour = tissue holding 80%, orange circle = fov80) and side view "
                  "(`*_maps_depth.png`). 3D volumes for ParaView are in `maps/*.vtk`.", ""]
        lines += [f"- `../maps/{m.name}`" for m in maps] + [""]
    if failures:
        lines += ["## Figures that could not be made", "", "See `failures.log`.", ""]
        lines += [f"- {name}: {msg}" for name, msg in failures] + [""]
    (ctx.out / "FIGURES.md").write_text("\n".join(lines))


# ---------------- main ----------------
def registry(ctx):
    figs = [
        ("geometry", fig_geometry),
        ("catheter", fig_catheter),
        ("protocol", fig_protocol),
        ("vm_traces", fig_vm_traces),
        ("tlift", fig_tlift_window),
    ]
    for name in [s.name for s in ctx.stages]:
        figs.append((f"snap_{name}_early", lambda c, n, s=name: fig_snapshots(c, n, s, "early")))
        if name in ("s3_reentry", "s4_ablate_terminate", "post_s3"):
            figs.append((f"snap_{name}_late", lambda c, n, s=name: fig_snapshots(c, n, s, "late")))
    for name in [n for n in ("s3_reentry", "s4_ablate_terminate") if n in ctx.by_name]:
        figs.append((f"depth_{name}", lambda c, n, s=name: fig_depth_snapshots(c, n, s)))
    figs += [
        ("time_space", fig_time_space),
        ("time_space_zoom", fig_time_space_zoom),
        ("activation", fig_activation_maps),
        ("cycle_length", fig_cycle_length),
        ("cv", fig_conduction_velocity),
        ("egm_grid", fig_egm_grid),
        ("egm_examples", fig_egm_examples),
    ]
    gaps = sorted(np.unique(ctx.points.gap_mm)) if ctx.points is not None and "gap_mm" in ctx.points else []
    for gap in gaps:
        figs.append((f"voltage_maps_gap{gap:g}", lambda c, n, g=gap: fig_voltage_maps(c, n, g)))
    figs += [
        ("ratio_map", fig_ratio_map),
        ("pp_vs_distance", fig_pp_vs_distance),
        ("gap_effect", fig_gap_effect),
        ("pp_distributions", fig_pp_distributions),
        ("peak_frequency", fig_peak_frequency),
        ("fov_box", fig_fov_box),
        ("fov_curves", fig_fov_curves),
        ("region_shares", fig_region_shares),
        ("gt_validation", fig_gt_validation),
        ("fov_vs_position", fig_fov_vs_position),
        ("fov_vs_pp", fig_fov_vs_pp),
    ]
    return figs


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--results", required=True, help="results folder of egm.py / ground_truth.py")
    p.add_argument("--config", required=True, help="the job's config JSON")
    p.add_argument("--only", default="", help="comma-separated figure numbers or names to make")
    p.add_argument("--no-vm", action="store_true", help="skip Vm-based figures (no vm.igb reads)")
    p.add_argument("--strict", action="store_true", help="exit non-zero if any figure fails")
    p.add_argument("stages", nargs="+", help="stage folders")
    a = p.parse_args()
    ctx = Ctx(a)
    only = {x.strip() for x in a.only.split(",") if x.strip()}
    failures = []
    log = ctx.out / "failures.log"
    if log.exists():
        log.unlink()
    for number, (name, fn) in enumerate(registry(ctx), start=1):
        if only and str(number) not in only and f"{number:02d}" not in only and name not in only:
            continue
        try:
            fn(ctx, number)
        except Exception as exc:  # one failed figure must not stop the rest
            plt.close("all")
            failures.append((name, f"{type(exc).__name__}: {exc}"))
            with open(log, "a") as f:
                f.write(f"=== {number:02d} {name}\n{traceback.format_exc()}\n")
            print(f"  FIGURE FAILED {number:02d} {name}: {type(exc).__name__}: {exc}", flush=True)
    write_outputs(ctx, failures)
    print(f"{len(ctx.captions)} figures -> {ctx.out} (captions: FIGURES.md)"
          + (f"; {len(failures)} failed, see failures.log" if failures else ""))
    if failures and a.strict:
        sys.exit(1)


if __name__ == "__main__":
    main()
