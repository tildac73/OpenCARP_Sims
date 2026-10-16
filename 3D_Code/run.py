#!/usr/bin/env python
"""
Micro-reentry in a 30 x 30 x 4 mm slab before and after ablation (setup after
PLOS Comp Biol, doi 10.1371/journal.pcbi.1014493). Physics, protocol and outputs live
in the layered .par files; this script builds the mesh and electrodes and chains the
checkpointed stages:

  pre       common + s1_pace -> s2_block -> s3_reentry            (baseline reentry)
  ablate    common + scar + s4_ablate_terminate, from the s3 checkpoint  (scenario A)
  reinduce  common + scar + s1 -> s2 -> s3, new output folders      (scenario B)

Only per-run values are passed on the command line: output/mesh/electrode paths,
restart checkpoints, T_LIFT, and (optionally) tuned conductivities.
"""

import glob
import json
import os
import shutil

import numpy as np

from carputils import settings, tools
from carputils.carpio import txt, igb

IMP_MODEL  = 'Koivumaki'           # TODO confirm name (bench --list-imps); must match common.par

# Slab mesh, hard-coded: mesher -size 3.0 3.0 0.4 (cm) -resolution 250 250 250 (um)
SLAB_MM     = (30.0, 30.0, 4.0)   # x, y, thickness (mm); centred on the origin
DX_MM       = 0.25                # mesh resolution (mm), all three axes

# Electrode lattice written by make_electrodes.py; point order gap, x, y, sample
EGM_GAPS_MM = (0.25, 1.0)
EGM_XY_MM   = np.arange(-11, 12)  # 23 x 23 positions, 1 mm pitch
EGM_SAMPLES = 8                   # surface points per cylindrical electrode, averaged

# Shared-pipeline export (Shared_Pipeline/egm.py, ground_truth.py): HD Grid 4 x 4 placements
# cut from the lattice at every centre where the whole grid fits, at both gaps
GRID_OFFSETS_MM = (-6, -2, 2, 6)  # 4 mm pitch, electrode e = row*4 + col (col along x)
SCAR_R_MM       = 2.5             # lesion radius; d_mm = SCAR_R_MM - r (> 0 inside the lesion)

DT_US = 20         # tissue time step (us) - must match dt in common.par
TS    = 1          # Crank-Nicolson - must match parab_solve in common.par
SPACEDT_MS = 1.0   # output step - must match spacedt in common.par

# Conductivity ratios kept when --tune-cv / --gi / --ge rescale common.par's values
ANISO   = 1.85     # longitudinal : transverse conductivity (paper CV ratio 1.36, CV ~ sqrt(g))
BUNDLE  = 0.6      # myobundle (tags 2, 3) = wall x 0.6 (paper: -40%)

EXAMPLE_DIR = os.path.dirname(os.path.abspath(__file__))
TUNED_FILE  = os.path.join(EXAMPLE_DIR, 'tuned_cv.csv')   # cache of tuned gi/ge

# Stages per scenario: (stage .par file, output folder, folder whose checkpoint it restarts from)
SCENARIOS = {
    'pre':      (False, [('s1_pace',             's1_pace',             None),
                         ('s2_block',            's2_block',            's1_pace'),
                         ('s3_reentry',          's3_reentry',          's2_block')]),
    'ablate':   (True,  [('s4_ablate_terminate', 's4_ablate_terminate', 's3_reentry')]),
    'reinduce': (True,  [('s1_pace',             'post_s1',             None),
                         ('s2_block',            'post_s2',             'post_s1'),
                         ('s3_reentry',          'post_s3',             'post_s2')]),
}


# define parameters exposed to the user on the commandline
def parser():
    parser = tools.standard_parser()
    group  = parser.add_argument_group('experiment specific options')

    group.add_argument('--scenario', default='all', choices=['all'] + list(SCENARIOS),
                       help='all = pre, then ablate and reinduce. ablate needs the pre '
                            'run\'s s3 checkpoint in the same output folder')
    group.add_argument('--t-lift', type=float, default=1950.0,
                       help='Time the exit block is lifted (ms) = s2 tend and checkpoint. '
                            'Set from the dry run in the setup notes (APD90/arrival window)')
    group.add_argument('--tune-cv', type=float, default=None,
                       help='Target longitudinal wall CV (m/s). Runs tuneCV once, caches gi/ge in '
                            'tuned_cv.csv, and rescales every region keeping the ratios. '
                            'Default: the conductivities in common.par')
    group.add_argument('--gi', type=float, default=None,
                       help='Longitudinal wall intracellular conductivity (S/m), instead of --tune-cv')
    group.add_argument('--ge', type=float, default=None,
                       help='Longitudinal wall extracellular conductivity (S/m), instead of --tune-cv')
    return parser

def tuneCV(args, job):
    """
    Return (gi, ge) in S/m giving --tune-cv on this setup.
    Reads tuned_cv.csv first; runs tuneCV only if
    this (cv, dx, dt, ts, model) has not been tuned before.
    TODO: tuneCV runs the model without common.par's cAF im_param remodelling.
    """
    key = '{:g},{:g},{},{},{}'.format(args.tune_cv, DX_MM, DT_US, TS, IMP_MODEL)
    if os.path.exists(TUNED_FILE):
        with open(TUNED_FILE) as f:
            for line in f:
                if line.startswith(key + ','):
                    gi, ge = map(float, line.strip().split(',')[-2:])
                    print('Using cached tuning for CV={}: gi={} ge={}'.format(args.tune_cv, gi, ge))
                    return gi, ge

    # Run tuneCV in its own folder (it writes results.dat, logs and a mesh)
    tdir = os.path.join(EXAMPLE_DIR, 'tunecv_cv{:g}_dx{:g}'.format(args.tune_cv, DX_MM))
    os.makedirs(tdir, exist_ok=True)
    cwd = os.getcwd()
    os.chdir(tdir)
    try:
        if os.path.isfile('results.dat'):
            os.remove('results.dat')
        cmd = [settings.execs.TUNECV,
               '--np',         args.np,
               '--resolution', DX_MM * 1000.0,      # mm -> um
               '--gi',         0.174,                # common.par wall, longitudinal
               '--ge',         0.625,
               '--ts',         TS,
               '--dt',         DT_US,
               '--tol',        0.0001,
               '--length',     0.25,                 # as in the tutorial
               '--lumping',    True,                 # common.par: mass_lumping = 1
               '--model',      IMP_MODEL,
               '--velocity',   args.tune_cv,
               '--converge',   True]
        job.bash(cmd)
        res = txt.read('results.dat')                # [CV, gi, ge, ...]
    finally:
        os.chdir(cwd)

    cv, gi, ge = float(res[0]), float(res[1]), float(res[2])
    print('Tuned CV={:.4f} m/s: gi={:.6f} ge={:.6f}'.format(cv, gi, ge))
    new = not os.path.exists(TUNED_FILE)
    with open(TUNED_FILE, 'a') as f:
        if new:
            f.write('target_cv,dx_mm,dt_us,ts,model,measured_cv,gi,ge\n')
        f.write('{},{:.5f},{:.6f},{:.6f}\n'.format(key, cv, gi, ge))
    return gi, ge


def conductivityOpts(gi, ge, stage):
    """
    Override common.par's conductivities from a longitudinal wall (gi, ge), keeping the
    1.85:1 anisotropy and the bundle = 0.6 x wall ratio. The exit segment (gregion[2])
    is left alone in s2_block, where the .par blocks it.
    """
    opts = []
    for r, scale in ((0, 1.0), (1, BUNDLE), (2, BUNDLE)):
        if r == 2 and stage == 's2_block':
            continue
        for ax, aniso in (('l', 1.0), ('t', ANISO), ('n', ANISO)):
            opts += ['-gregion[{}].g_i{}'.format(r, ax), gi * scale / aniso,
                     '-gregion[{}].g_e{}'.format(r, ax), ge * scale / aniso]
    return opts


def jobID(args):
    """
    Generate name of top level output directory. No date, so a later --scenario ablate
    finds the pre run's checkpoints in the same folder. Under 3D_Code/runs/ wherever the
    script is started from, so the shared pipeline's run glob finds it.
    """
    cv = f'_cv{args.tune_cv:g}' if args.tune_cv else ''
    return os.path.join(EXAMPLE_DIR, 'runs', f'reentry_tlift{args.t_lift:g}{cv}_np{args.np}')


def buildMesh(job):
    """
    mesher block -> mesh/slab_raw, retagged by element centroid (tags in the setup notes):
    mesh/slab_bundle (pre, tags 1-4) and mesh/slab_bundle_scar (post, + tag 5).
    Same .pts/.lon for both, so checkpoints restart across pre and post.
    Returns the two mesh base names. Reused if already built.
    """
    mdir = os.path.join(job.ID, 'mesh')
    pre, post = os.path.join(mdir, 'slab_bundle'), os.path.join(mdir, 'slab_bundle_scar')
    if os.path.exists(post + '.elem'):
        return pre, post

    os.makedirs(mdir, exist_ok=True)
    raw = os.path.join(mdir, 'slab_raw')
    cmd = [settings.execs.MESHER]
    for i in range(3):
        cmd += ['-size[{}]'.format(i), SLAB_MM[i] / 10.0,               # mm -> cm
                '-resolution[{}]'.format(i), DX_MM * 1000.0]            # mm -> um
    job.bash(cmd + ['-mesh', raw])

    pts = np.loadtxt(raw + '.pts', skiprows=1)
    lo, hi = pts.min(axis=0) / 1000.0, pts.max(axis=0) / 1000.0
    if not (np.allclose(lo, -np.array(SLAB_MM) / 2, atol=0.01) and
            np.allclose(hi, np.array(SLAB_MM) / 2, atol=0.01)):
        raise ValueError('mesher block is not centred on the origin: {} to {} mm'.format(lo, hi))

    lines = open(raw + '.elem').read().split('\n')[1:]
    elems = [l.split() for l in lines if l.strip()]
    idx  = np.array([[int(v) for v in e[1:5]] for e in elems])         # tetrahedra (Tt)
    c    = pts[idx].mean(axis=1) / 1000.0                              # centroids in mm
    x, y, z = c[:, 0], c[:, 1], c[:, 2]

    tag = np.ones(len(c), dtype=int)                                   # 1 = wall
    in_x = (x > -7.5) & (x < 7.5)
    sheath = in_x & (z > 0.25) & (((np.abs(y) > 1.25) & (np.abs(y) < 1.5)) |
                                  ((z < 0.5) & (np.abs(y) < 1.5)))
    bundle = in_x & (np.abs(y) < 1.25) & (z > 0.5)
    tag[sheath] = 4
    tag[bundle] = 2
    tag[bundle & (x > 5.5)] = 3                                        # exit segment
    tag_post = tag.copy()
    tag_post[np.hypot(x, y) < SCAR_R_MM] = 5                           # transmural scar cylinder

    for base, t in ((pre, tag), (post, tag_post)):
        with open(base + '.elem', 'w') as f:
            f.write(f'{len(elems)}\n')
            for e, ti in zip(elems, t):
                f.write(' '.join(e[:5]) + f' {ti}\n')
        for ext in ('.pts', '.lon'):
            shutil.copy(raw + ext, base + ext)
    print('Mesh tags (pre): ' + str(dict(zip(*np.unique(tag, return_counts=True)))))
    print('Scar elements: {}'.format(int(np.sum(tag_post == 5))))
    return pre, post


def writeRecoveryPoints(job):
    """
    Run make_electrodes.py in the job folder (writes electrodes/egm_points.pts) and
    return the path without extension, as -phie_rec_ptf expects.
    """
    base = os.path.abspath(os.path.join(job.ID, 'electrodes', 'egm_points'))
    if not os.path.exists(base + '.pts'):
        cwd = os.getcwd()
        os.chdir(job.ID)
        try:
            job.bash(['python3', os.path.join(EXAMPLE_DIR, 'make_electrodes.py')])
        finally:
            os.chdir(cwd)
    return base


def checkpoint(simdir):
    """(checkpoint name for -start_statef, its time in ms) of the stage run in simdir."""
    found = sorted(glob.glob(os.path.join(simdir, 'state.*')))
    if not found:
        raise FileNotFoundError('No checkpoint (state.*) in ' + simdir)
    name = found[-1][:-len('.roe')] if found[-1].endswith('.roe') else found[-1]
    t = float(os.path.basename(name)[len('state.'):].rsplit('.', 1)[0])  # state.1795.0 -> 1795
    return name, t


def buildEGMs(simdir, t0_ms, egm_points):
    """
    Unipolar electrograms from phie_recovery.igb: average each electrode's surface
    samples. Saves <simdir>/egm.npz with
      lattice_uni (gap, ix, iy, T) in mV, gap_mm, x_mm, y_mm   the full 1 mm lattice
    and, in the shared-pipeline format (P = gap x grid centre placements of a 4 x 4 grid):
      uni (P, 16, T), uni_xy (P, 16, 3) electrode centres in mm, uni_sample_xyz (P, 16, 8, 3),
      d_mm (P, 16), centre_xy_mm (P, 2), p_gap_mm (P,), t_ms (T,) in absolute time
    Bipoles, omnipoles and LATs are made from these in post-processing.
    """
    data, _, _ = igb.read(os.path.join(simdir, 'phie_recovery.igb'))   # (npts, T)
    phi = np.asarray(data, dtype=np.float32)
    n = len(EGM_XY_MM)
    lattice = phi.reshape(len(EGM_GAPS_MM), n, n, EGM_SAMPLES, -1).mean(axis=3)
    samples = (np.loadtxt(egm_points + '.pts', skiprows=1) / 1000.0).reshape(
        len(EGM_GAPS_MM), n, n, EGM_SAMPLES, 3)                       # mm

    off = np.array(GRID_OFFSETS_MM)
    centres = EGM_XY_MM[(EGM_XY_MM + off.min() >= EGM_XY_MM[0]) & (EGM_XY_MM + off.max() <= EGM_XY_MM[-1])]
    sel = [(g, cx, cy) for g in range(len(EGM_GAPS_MM)) for cx in centres for cy in centres]
    ix = lambda v: np.searchsorted(EGM_XY_MM, v)                       # noqa: E731
    grid = [(ix(cx + off[c]), ix(cy + off[r])) for _, cx, cy in sel for r in range(4) for c in range(4)]
    g16 = np.repeat([g for g, _, _ in sel], 16)
    i, j = np.array(grid).T
    uni = lattice[g16, i, j].reshape(len(sel), 16, -1)
    smp = samples[g16, i, j].reshape(len(sel), 16, EGM_SAMPLES, 3)
    xyz = smp.mean(axis=2)
    np.savez_compressed(os.path.join(simdir, 'egm.npz'),
                        lattice_uni=lattice, gap_mm=np.array(EGM_GAPS_MM), x_mm=EGM_XY_MM, y_mm=EGM_XY_MM,
                        uni=uni, uni_xy=xyz, uni_sample_xyz=smp,
                        d_mm=SCAR_R_MM - np.hypot(xyz[..., 0], xyz[..., 1]),
                        centre_xy_mm=np.array([(cx, cy) for _, cx, cy in sel], float),
                        p_gap_mm=np.array([EGM_GAPS_MM[g] for g, _, _ in sel]),
                        t_ms=t0_ms + np.arange(uni.shape[-1]) * SPACEDT_MS)
    print('Electrograms saved: ' + os.path.join(simdir, 'egm.npz'))


@tools.carpexample(parser, jobID)
def run(args, job):

    job.overwrite_behaviour = 'overwrite'

    # --------------- Conductivities: tuned, from the CLI, or common.par's --------
    if args.tune_cv is not None:
        args.gi, args.ge = tuneCV(args, job)
    elif (args.gi is None) != (args.ge is None):
        raise ValueError('Give both --gi and --ge (or --tune-cv, or neither for common.par)')

    # --------------- Mesh and electrodes (shared by every stage) ------------------
    os.makedirs(job.ID, exist_ok=True)
    mesh_pre, mesh_post = buildMesh(job)
    egm_points = writeRecoveryPoints(job)

    scenarios = list(SCENARIOS) if args.scenario == 'all' else [args.scenario]
    for scenario in scenarios:
        scar, stages = SCENARIOS[scenario]
        for stage, out, restart in stages:
            simdir = os.path.join(job.ID, out)

            # Layered settings: later files override earlier ones
            cmd  = tools.carp_cmd(os.path.join(EXAMPLE_DIR, 'common.par'))
            if scar:
                cmd += ['+F', os.path.join(EXAMPLE_DIR, 'scar.par')]
            cmd += ['+F', os.path.join(EXAMPLE_DIR, stage + '.par')]

            # Per-run values (override the .par files)
            cmd += ['-simID',        simdir,
                    '-meshname',     mesh_post if scar else mesh_pre,
                    '-phie_rec_ptf', egm_points]
            t0 = 0.0
            if restart:
                state, t0 = checkpoint(os.path.join(job.ID, restart))
                cmd += ['-start_statef', state]
            if stage == 's2_block':
                cmd += ['-tend', args.t_lift, '-tsav[0]', args.t_lift]
            if args.gi is not None:
                cmd += conductivityOpts(args.gi, args.ge, stage)

            print('=== {}: {} -> {}'.format(scenario, stage, simdir))
            job.carp(cmd)

            # Metadata and mesh for the shared pipeline (egm.py, ground_truth.py)
            with open(os.path.join(simdir, 'meta.json'), 'w') as f:
                json.dump({'model': 'slab3d', 'run': os.path.basename(job.ID) + '/' + out,
                           'state': 'post' if scar else 'pre', 'scenario': scenario,
                           'stage': stage}, f, indent=1)
            with open(os.path.join(simdir, 'meshname.txt'), 'w') as f:
                f.write(os.path.abspath(mesh_post if scar else mesh_pre) + '\n')

            # Unipolar electrograms from the recovered potentials
            if os.path.exists(os.path.join(simdir, 'phie_recovery.igb')):
                buildEGMs(simdir, t0, egm_points)
            else:
                print('No phie_recovery.igb - electrograms not built')

    # Do visualization: Vm of the last stage run
    if args.visualize and not settings.platform.BATCH:
        job.meshalyzer(mesh_pre, os.path.join(simdir, 'vm.igb'))


if __name__ == '__main__':
    run()
