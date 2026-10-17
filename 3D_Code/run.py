#!/usr/bin/env python
"""
Micro-reentry in a 30 x 30 x 4 mm slab before and after ablation (setup after
PLOS Comp Biol, doi 10.1371/journal.pcbi.1014493). Physics, protocol and outputs live
in the layered .par files; this script builds the mesh and electrodes and chains the
checkpointed stages:

  pre       common + s1_pace -> s2_block -> s3_reentry            (baseline reentry)
  ablate    common + scar + s4_ablate_terminate, from the s3 checkpoint  (scenario A)
  reinduce  common + scar + s1 -> s2 -> s3, new output folders      (scenario B)

The default protocol runs pre then ablate (four stages). Reinduction is optional.

Only per-run values are passed on the command line: output/mesh/electrode paths,
restart checkpoints, T_LIFT, and optional validated conductivity overrides.
"""

import json
import os
import shutil
import re
import sys
import time
from pathlib import Path

import numpy as np

from carputils import settings, tools

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "Shared_Pipeline"))
import igb_io

# Slab mesh, hard-coded: mesher -size 3.0 3.0 0.4 (cm) -resolution 250 250 250 (um)
SLAB_MM = (30.0, 30.0, 4.0)  # x, y, thickness (mm); centred on the origin
DX_MM = 0.25  # mesh resolution (mm), all three axes

# Electrode lattice written by make_electrodes.py; point order gap, x, y, sample
EGM_GAPS_MM = (0.25, 1.0)
EGM_XY_MM = np.arange(-11, 12)  # 23 x 23 positions, 1 mm pitch
EGM_SAMPLES = 8  # surface points per cylindrical electrode, averaged

# Shared-pipeline export (Shared_Pipeline/egm.py, ground_truth.py): HD Grid 4 x 4 placements
# cut from the lattice at every centre where the whole grid fits, at both gaps
GRID_OFFSETS_MM = (-6, -2, 2, 6)  # 4 mm pitch, electrode e = row*4 + col (col along x)
SCAR_R_MM = 2.5  # lesion radius; d_mm = SCAR_R_MM - r (> 0 inside the lesion)

SPACEDT_MS = 1.0  # output step - must match spacedt in common.par

# Conductivity ratios kept when --gi / --ge rescale common.par's values
ANISO = 1.85  # longitudinal : transverse conductivity (paper CV ratio 1.36, CV ~ sqrt(g))
BUNDLE = 0.6  # myobundle (tags 2, 3) = wall x 0.6 (paper: -40%)

EXAMPLE_DIR = os.path.dirname(os.path.abspath(__file__))

# Stages per scenario: (stage .par file, output folder, folder whose checkpoint it restarts from)
SCENARIOS = {
    "pre": (
        False,
        [
            ("s1_pace", "s1_pace", None),
            ("s2_block", "s2_block", "s1_pace"),
            ("s3_reentry", "s3_reentry", "s2_block"),
        ],
    ),
    "ablate": (True, [("s4_ablate_terminate", "s4_ablate_terminate", "s3_reentry")]),
    "reinduce": (
        True,
        [
            ("s1_pace", "post_s1", None),
            ("s2_block", "post_s2", "post_s1"),
            ("s3_reentry", "post_s3", "post_s2"),
        ],
    ),
}


# define parameters exposed to the user on the commandline
def parser():
    parser = tools.standard_parser()
    group = parser.add_argument_group("experiment specific options")

    group.add_argument(
        "--scenario",
        default="protocol",
        choices=["protocol", "all"] + list(SCENARIOS),
        help="protocol = four stages (pre then ablate); all also runs reinduction. ablate needs the pre "
        "run's s3 checkpoint in the same output folder",
    )
    group.add_argument(
        "--t-lift",
        type=float,
        default=1950.0,
        help="Time the exit block is lifted (ms) = s2 tend and checkpoint. "
        "Set from a pilot simulation measuring APD90 and arrival at the exit",
    )
    group.add_argument(
        "--tune-cv",
        type=float,
        default=None,
        help="Disabled until the tuner supports the same cAF remodelling as common.par",
    )
    group.add_argument(
        "--resume",
        action="store_true",
        help="Reuse completed stages (meta.json marked complete); move incomplete or stale "
        "stage folders aside and rerun them. Without it, existing output is an error",
    )
    group.add_argument(
        "--gi",
        type=float,
        default=None,
        help="Validated longitudinal wall intracellular conductivity (S/m); also give --ge",
    )
    group.add_argument(
        "--ge",
        type=float,
        default=None,
        help="Validated longitudinal wall extracellular conductivity (S/m); also give --gi",
    )
    return parser


def conductivityOpts(gi, ge, stage):
    """
    Override common.par's conductivities from a longitudinal wall (gi, ge), keeping the
    1.85:1 anisotropy and the bundle = 0.6 x wall ratio. The exit segment (gregion[2])
    is left alone in s2_block, where the .par blocks it.
    """
    opts = []
    for r, scale in ((0, 1.0), (1, BUNDLE), (2, BUNDLE)):
        if r == 2 and stage == "s2_block":
            continue
        for ax, aniso in (("l", 1.0), ("t", ANISO), ("n", ANISO)):
            opts += [
                "-gregion[{}].g_i{}".format(r, ax),
                gi * scale / aniso,
                "-gregion[{}].g_e{}".format(r, ax),
                ge * scale / aniso,
            ]
    return opts


def jobID(args):
    """
    Generate name of top level output directory. No date, so a later --scenario ablate
    finds the pre run's checkpoints in the same folder. Under 3D_Code/runs/ wherever the
    script is started from, so the shared pipeline's run glob finds it.
    """
    cv = f"_cv{args.tune_cv:g}" if args.tune_cv else ""
    conductivities = (
        f"_gi{args.gi:g}_ge{args.ge:g}" if args.gi is not None and args.ge is not None else ""
    )
    return os.path.join(
        EXAMPLE_DIR, "runs", f"reentry_tlift{args.t_lift:g}{cv}{conductivities}_np{args.np}"
    )


def buildMesh(job):
    """
    mesher block -> mesh/slab_raw, retagged by element centroid (tags in the setup notes):
    mesh/slab_bundle (pre, tags 1-4) and mesh/slab_bundle_scar (post, + tag 5).
    Same .pts/.lon for both, so checkpoints restart across pre and post.
    Returns the two mesh base names. Reused if already built.
    """
    mdir = os.path.join(job.ID, "mesh")
    pre, post = os.path.join(mdir, "slab_bundle"), os.path.join(mdir, "slab_bundle_scar")
    if all(os.path.exists(base + ext) for base in (pre, post) for ext in (".pts", ".elem", ".lon")):
        return pre, post

    os.makedirs(mdir, exist_ok=True)
    raw = os.path.join(mdir, "slab_raw")
    cmd = [settings.execs.MESHER]
    for i in range(3):
        cmd += [
            "-size[{}]".format(i),
            SLAB_MM[i] / 10.0,  # mm -> cm
            "-resolution[{}]".format(i),
            DX_MM * 1000.0,
        ]  # mm -> um
    job.bash(cmd + ["-mesh", raw])
    missing = [raw + ext for ext in (".pts", ".elem", ".lon") if not os.path.isfile(raw + ext)]
    if missing:
        raise RuntimeError(
            "mesher did not write {}; check settings.execs.MESHER in your carputils settings".format(
                missing
            )
        )

    pts = np.loadtxt(raw + ".pts", skiprows=1)
    lo, hi = pts.min(axis=0) / 1000.0, pts.max(axis=0) / 1000.0
    if not (
        np.allclose(lo, -np.array(SLAB_MM) / 2, atol=0.01)
        and np.allclose(hi, np.array(SLAB_MM) / 2, atol=0.01)
    ):
        raise ValueError("mesher block is not centred on the origin: {} to {} mm".format(lo, hi))

    with open(raw + ".elem") as stream:
        lines = stream.read().splitlines()[1:]
    elems = [l.split() for l in lines if l.strip()]
    if any(e[0] != "Tt" or len(e) != 6 for e in elems):
        raise ValueError("Expected tagged tetrahedra from mesher")
    idx = np.array([[int(v) for v in e[1:5]] for e in elems])  # tetrahedra (Tt)
    c = pts[idx].mean(axis=1) / 1000.0  # centroids in mm
    x, y, z = c[:, 0], c[:, 1], c[:, 2]

    tag = np.ones(len(c), dtype=int)  # 1 = wall
    in_x = (x > -7.5) & (x < 7.5)
    sheath = (
        in_x
        & (z > 0.25)
        & (((np.abs(y) > 1.25) & (np.abs(y) < 1.5)) | ((z < 0.5) & (np.abs(y) < 1.5)))
    )
    bundle = in_x & (np.abs(y) < 1.25) & (z > 0.5)
    tag[sheath] = 4
    tag[bundle] = 2
    tag[bundle & (x > 5.5)] = 3  # exit segment
    tag_post = tag.copy()
    tag_post[np.hypot(x, y) < SCAR_R_MM] = 5  # transmural scar cylinder

    for base, t in ((pre, tag), (post, tag_post)):
        with open(base + ".elem", "w") as f:
            f.write(f"{len(elems)}\n")
            for e, ti in zip(elems, t):
                f.write(" ".join(e[:5]) + f" {ti}\n")
        for ext in (".pts", ".lon"):
            shutil.copy(raw + ext, base + ext)
    print("Mesh tags (pre): " + str(dict(zip(*np.unique(tag, return_counts=True)))))
    print("Scar elements: {}".format(int(np.sum(tag_post == 5))))
    return pre, post


def writeRecoveryPoints(job):
    """
    Run make_electrodes.py in the job folder (writes electrodes/egm_points.pts) and
    return the path without extension, as -phie_rec_ptf expects.
    """
    base = os.path.abspath(os.path.join(job.ID, "electrodes", "egm_points"))
    if not os.path.exists(base + ".pts"):
        cwd = os.getcwd()
        os.chdir(job.ID)
        try:
            job.bash([sys.executable, os.path.join(EXAMPLE_DIR, "make_electrodes.py")])
        finally:
            os.chdir(cwd)
    if not os.path.isfile(base + ".pts"):
        raise RuntimeError("make_electrodes.py did not write " + base + ".pts")
    return base


def stage_time_axis(path, t0, tend):
    """
    Absolute output times of a stage. Uses the IGB header when it spans t0..tend; if the
    header origin is missing or reset on restart, infers the axis from the frame count
    (first frame at t0, or one output step later) and reports the correction.
    """
    t = igb_io.time_axis(path)
    if np.isclose(t[0], t0, rtol=0, atol=1e-6) and np.isclose(t[-1], tend, rtol=0, atol=SPACEDT_MS):
        return t
    step = float(t[1] - t[0]) if len(t) > 1 else SPACEDT_MS
    for first in (t0, t0 + step):
        if np.isclose(first + (len(t) - 1) * step, tend, rtol=0, atol=0.5 * step + 1e-6):
            print(
                f"NOTE {path}: header times {t[0]:g}-{t[-1]:g} ms; using {first:g}-{tend:g} ms "
                "from the restart time and frame count"
            )
            return first + np.arange(len(t)) * step
    raise RuntimeError(
        f"{path}: {len(t)} frames ({t[0]:g}-{t[-1]:g} ms) do not cover {t0:g}-{tend:g} ms"
    )


def checkpoint(simdir, expected_ms):
    """Choose the explicitly requested restart time, never a lexicographic last file."""
    matches = []
    for path in Path(simdir).glob("state.*.roe"):
        match = re.fullmatch(r"state\.([0-9]+(?:\.[0-9]+)?)\.roe", path.name)
        if match and np.isclose(float(match[1]), expected_ms, rtol=0, atol=1e-6):
            matches.append(path)
    if len(matches) != 1:
        raise FileNotFoundError(
            f"Expected one checkpoint at {expected_ms:g} ms in {simdir}; found {len(matches)}"
        )
    return str(matches[0])[:-4], expected_ms


def buildEGMs(simdir, t0_ms, egm_points, t_ms=None):
    """
    Unipolar electrograms from phie_recovery.igb: average each electrode's surface
    samples. Saves <simdir>/egm.npz with
      lattice_uni (gap, ix, iy, T) in mV, gap_mm, x_mm, y_mm   the full 1 mm lattice
    and, in the shared-pipeline format (P = gap x grid centre placements of a 4 x 4 grid):
      uni (P, 16, T), uni_xy (P, 16, 3) electrode centres in mm, uni_sample_xyz (P, 16, 8, 3),
      d_mm (P, 16), centre_xy_mm (P, 2), p_gap_mm (P,), t_ms (T,) in absolute time
    Bipoles, omnipoles and LATs are made from these in post-processing.
    """
    path = os.path.join(simdir, "phie_recovery.igb")
    phi = igb_io.read(path).T  # (recovery points, time)
    header_t = igb_io.time_axis(path)
    vm_time = igb_io.time_axis(os.path.join(simdir, "vm.igb"))
    if header_t.shape != vm_time.shape or not np.allclose(np.diff(header_t), np.diff(vm_time)):
        raise ValueError("Vm and recovered electrograms have different time axes")
    if t_ms is None:  # trust the header
        t_ms = header_t
        if not np.isclose(t_ms[0], t0_ms):
            raise ValueError(f"IGB starts at {t_ms[0]} ms, expected restart at {t0_ms} ms")
    elif np.shape(t_ms) != header_t.shape:
        raise ValueError("Supplied time axis does not match the recovery frames")
    n = len(EGM_XY_MM)
    expected_points = len(EGM_GAPS_MM) * n * n * EGM_SAMPLES
    if phi.shape[0] != expected_points or not np.isfinite(phi).all():
        raise ValueError(
            f"Invalid recovery data: expected {expected_points} finite traces, got {phi.shape}"
        )
    lattice = phi.reshape(len(EGM_GAPS_MM), n, n, EGM_SAMPLES, -1).mean(axis=3)
    samples = (np.loadtxt(egm_points + ".pts", skiprows=1) / 1000.0).reshape(
        len(EGM_GAPS_MM), n, n, EGM_SAMPLES, 3
    )  # mm

    off = np.array(GRID_OFFSETS_MM)
    centres = EGM_XY_MM[
        (EGM_XY_MM + off.min() >= EGM_XY_MM[0]) & (EGM_XY_MM + off.max() <= EGM_XY_MM[-1])
    ]
    sel = [(g, cx, cy) for g in range(len(EGM_GAPS_MM)) for cx in centres for cy in centres]
    ix = lambda v: np.searchsorted(EGM_XY_MM, v)  # noqa: E731
    grid = [
        (ix(cx + off[c]), ix(cy + off[r])) for _, cx, cy in sel for r in range(4) for c in range(4)
    ]
    g16 = np.repeat([g for g, _, _ in sel], 16)
    i, j = np.array(grid).T
    uni = lattice[g16, i, j].reshape(len(sel), 16, -1)
    smp = samples[g16, i, j].reshape(len(sel), 16, EGM_SAMPLES, 3)
    xyz = smp.mean(axis=2)
    np.savez_compressed(
        os.path.join(simdir, "egm.npz"),
        lattice_uni=lattice,
        gap_mm=np.array(EGM_GAPS_MM),
        x_mm=EGM_XY_MM,
        y_mm=EGM_XY_MM,
        uni=uni,
        uni_xy=xyz,
        uni_sample_xyz=smp,
        d_mm=SCAR_R_MM - np.hypot(xyz[..., 0], xyz[..., 1]),
        centre_xy_mm=np.array([(cx, cy) for _, cx, cy in sel], float),
        p_gap_mm=np.array([EGM_GAPS_MM[g] for g, _, _ in sel]),
        t_ms=t_ms,
    )
    print("Electrograms saved: " + os.path.join(simdir, "egm.npz"))


# Checkpoint each stage must leave for the next (ms, or the block-lift time)
CHECKPOINT_FOR = {"s1_pace": 1795.0, "s2_block": "t_lift", "s3_reentry": 3500.0}


def stageTimes(stage, t_lift):
    """(start, end) in ms of a stage; restarts begin at the checkpoint they load."""
    return {
        "s1_pace": (0.0, 1795.0),
        "s2_block": (1795.0, float(t_lift)),
        "s3_reentry": (float(t_lift), 4000.0),
        "s4_ablate_terminate": (3500.0, 6000.0),
    }[stage]


def stageComplete(simdir, stage, t_lift):
    """True when a stage folder holds every output of a finished stage at the same T_LIFT."""
    try:
        with open(os.path.join(simdir, "meta.json")) as f:
            meta = json.load(f)
        if not meta.get("complete") or meta.get("stage") != stage:
            return False
        if not np.isclose(meta.get("t_lift_ms", t_lift), t_lift):
            return False
        for name in ("vm.igb", "phie_recovery.igb", "parameters.par", "egm.npz", "meshname.txt"):
            if not os.path.isfile(os.path.join(simdir, name)):
                return False
        needed = CHECKPOINT_FOR.get(stage)
        if needed is not None:
            checkpoint(simdir, t_lift if needed == "t_lift" else needed)
        igb_io.header(os.path.join(simdir, "vm.igb"))  # raises if truncated
        return True
    except (OSError, ValueError, KeyError):
        return False


# The runner owns its stage folders: carputils must not clean the parent on restart.
@tools.carpexample(parser, jobID, simple=True)
def run(args, job):

    # Validate before launching expensive simulation stages.
    if not np.isfinite(args.t_lift) or not 1802 < args.t_lift < 3500:
        raise ValueError(
            "--t-lift must be after the final stimulus (1802) and before the s3 checkpoint (3500 ms)"
        )
    if args.tune_cv is not None:
        raise ValueError(
            "--tune-cv is disabled: the existing tuner omits cAF remodelling. Use validated --gi/--ge or common.par."
        )
    if args.gi is not None and (not np.isfinite(args.gi) or args.gi <= 0):
        raise ValueError("--gi must be positive and finite")
    if args.ge is not None and (not np.isfinite(args.ge) or args.ge <= 0):
        raise ValueError("--ge must be positive and finite")
    if settings.platform.BATCH or getattr(args, "dry", False):
        raise ValueError(
            "This checkpoint chain needs immediate execution; use a local carputils platform (inside your compute allocation), without --dry"
        )

    # Conductivity overrides must be a complete intracellular/extracellular pair.
    if (args.gi is None) != (args.ge is None):
        raise ValueError("Give both --gi and --ge, or neither for common.par")

    # --------------- Mesh and electrodes (shared by every stage) ------------------
    os.makedirs(job.ID, exist_ok=True)
    mesh_pre, mesh_post = buildMesh(job)
    egm_points = writeRecoveryPoints(job)

    scenarios = (
        list(SCENARIOS)
        if args.scenario == "all"
        else ["pre", "ablate"] if args.scenario == "protocol" else [args.scenario]
    )
    # Do not silently mix a rerun with existing output or stale checkpoints. With --resume,
    # completed stages are reused and incomplete ones are moved aside and rerun.
    resume = getattr(args, "resume", False)
    if not resume:
        for scenario in scenarios:
            for _, out, _ in SCENARIOS[scenario][1]:
                destination = Path(job.ID) / out
                if destination.exists() and any(destination.iterdir()):
                    raise FileExistsError(
                        f"{destination} already contains output; archive it, use RESUME=1 "
                        "or SKIP_SIMS=1"
                    )
    rerun = set()  # stages (re)simulated in this call: anything restarting from them is stale
    for scenario in scenarios:
        scar, stages = SCENARIOS[scenario]
        for stage, out, restart in stages:
            simdir = os.path.join(job.ID, out)
            t0, tend = stageTimes(stage, args.t_lift)
            if resume and os.path.isdir(simdir) and os.listdir(simdir):
                if restart not in rerun and stageComplete(simdir, stage, args.t_lift):
                    print("=== {}: {} complete, reusing {}".format(scenario, stage, simdir))
                    continue
                aside = "{}.{}-{}".format(
                    simdir,
                    "stale" if restart in rerun else "incomplete",
                    time.strftime("%Y%m%d-%H%M%S"),
                )
                os.rename(simdir, aside)
                print("=== {}: moved previous {} output to {}".format(scenario, stage, aside))
            rerun.add(out)

            # Layered settings: later files override earlier ones
            cmd = tools.carp_cmd(os.path.join(EXAMPLE_DIR, "common.par"))
            if scar:
                cmd += ["+F", os.path.join(EXAMPLE_DIR, "scar.par")]
            cmd += ["+F", os.path.join(EXAMPLE_DIR, stage + ".par")]

            # Per-run values (override the .par files)
            cmd += [
                "-simID",
                simdir,
                "-meshname",
                mesh_post if scar else mesh_pre,
                "-phie_rec_ptf",
                egm_points,
            ]
            if restart:
                state, _ = checkpoint(os.path.join(job.ID, restart), t0)
                cmd += ["-start_statef", state]
            if stage == "s2_block":
                cmd += ["-tend", args.t_lift, "-tsav[0]", args.t_lift]
            if args.gi is not None:
                cmd += conductivityOpts(args.gi, args.ge, stage)

            print("=== {}: {} -> {}  ({:g}-{:g} ms)".format(scenario, stage, simdir, t0, tend))
            started = time.time()
            job.carp(cmd)
            elapsed = time.time() - started

            required = ["vm.igb", "phie_recovery.igb", "parameters.par"]
            missing = [name for name in required if not (Path(simdir) / name).is_file()]
            if missing:
                raise RuntimeError(f"{out}: missing simulation outputs: {missing}")
            t_ms = stage_time_axis(Path(simdir) / "vm.igb", t0, tend)
            needed = CHECKPOINT_FOR.get(stage)
            if needed is not None:
                checkpoint(simdir, args.t_lift if needed == "t_lift" else needed)

            with open(os.path.join(simdir, "meshname.txt"), "w") as f:
                f.write(os.path.abspath(mesh_post if scar else mesh_pre) + "\n")

            # Unipolar electrograms from the recovered potentials
            buildEGMs(simdir, t0, egm_points, t_ms)

            # Metadata for the shared pipeline (egm.py, ground_truth.py); written last, so a
            # stage with meta.json "complete" has every output
            with open(os.path.join(simdir, "meta.json"), "w") as f:
                json.dump(
                    {
                        "model": "slab3d",
                        "run": os.path.basename(job.ID) + "/" + out,
                        "state": "post" if scar else "pre",
                        "scenario": scenario,
                        "stage": stage,
                        "t_start_ms": float(t_ms[0]),
                        "t_end_ms": float(t_ms[-1]),
                        "t_lift_ms": float(args.t_lift),
                        "restart_from": restart,
                        "np": getattr(args, "np", None),
                        "wall_time_s": round(elapsed, 1),
                        "complete": True,
                    },
                    f,
                    indent=1,
                )
            print("=== {}: {} finished in {:.1f} min".format(scenario, stage, elapsed / 60))

    # Do visualization: Vm of the last stage run
    if args.visualize and not settings.platform.BATCH:
        job.meshalyzer(mesh_post if scar else mesh_pre, os.path.join(simdir, "vm.igb"))


if __name__ == "__main__":
    run()
