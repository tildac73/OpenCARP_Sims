#!/usr/bin/env python

import os
from datetime import date

import numpy as np

from carputils import settings, tools, mesh
from carputils.carpio import txt

ACTIVE_TAG = 1
SCAR_TAG   = 100   # ablation lesion: necrotic, non-conducting band (post only)
ISO_TAG    = 101   # isolated wall beyond the lesion: viable, decoupled, never activated (post only)
IMP_MODEL  = 'Courtemanche'

CORNER_LEG = 5.0   # mm, leg of the corner stimulus triangle for 45 deg (x + y <= -L + CORNER_LEG)

DT_US = 25         # tissue time step (us) - must match dt in the .par files
TS    = 1          # Crank-Nicolson - must match parab_solve in the .par files

EXAMPLE_DIR = os.path.dirname(os.path.abspath(__file__))
TUNED_FILE  = os.path.join(EXAMPLE_DIR, 'tuned_cv.csv')   # cache of tuned gi/ge

# Region layout (ionic + conductivity) lives in these files
PAR_FILES = {'pre':  'slab2d_pre.par',
             'post': 'slab2d_post.par'}

# define parameters exposed to the user on the commandline
def parser():
    parser = tools.standard_parser()
    group  = parser.add_argument_group('experiment specific options')

    group.add_argument('--dx', type=float, default=0.25,
                       help='Mesh resolution (mm): 0.25 or 0.125')
    group.add_argument('--size', type=float, default=20.0,
                       help='Sheet edge length (mm); scar occupies x >= 0 half')
    group.add_argument('--angle', type=int, default=0, choices=[0, 45, 90],
                       help='Wavefront direction relative to scar-boundary normal (deg): '
                            '0 = into the boundary, 90 = parallel to it, 45 = oblique')
    group.add_argument('--tune-cv', type=float, default=None,
                       help='Target CV (m/s). Runs tuneCV (CRN, this dx, dt, ts) once, caches '
                            'gi/ge in tuned_cv.csv, and uses them. Replaces --gi/--ge')
    group.add_argument('--gi', type=float, default=None,
                       help='Intracellular conductivity (S/m), if not using --tune-cv')
    group.add_argument('--ge', type=float, default=None,
                       help='Extracellular conductivity (S/m), if not using --tune-cv')
    group.add_argument('--bcl', type=float, default=1000.0,
                       help='Single-cell prepacing cycle length (ms)')
    group.add_argument('--npls', type=int, default=20,
                       help='Number of single-cell prepacing beats')
    group.add_argument('--tend', type=float, default=250.0,
                       help='Tissue simulation duration (ms)')
    group.add_argument('--stim-width', type=float, default=0.5,
                       help='Thickness of the planar stimulus band (mm)')
    group.add_argument('--state', default='post', choices=['pre', 'post'],
                       help='pre: whole sheet active (no scar). '
                            'post: lesion band from x = 0, isolated wall beyond it')
    group.add_argument('--lesion-width', type=float, default=6.0,
                       help='Lesion band width (mm), starting at x = 0 (post only). '
                            'Default 6 mm = 2 x the 3 mm half-width in Wang\'s boundary rule')
    group.add_argument('--lesion-g', type=float, default=1e-3,
                       help='Lesion intracellular conductivity as a fraction of --gi (post only)')
    return parser

def tuneCV(args, job):
    """
    Return (gi, ge) in S/m giving --tune-cv on this setup, following the openCARP
    03A_study_prep_tuneCV tutorial. Reads tuned_cv.csv first; runs tuneCV only if
    this (cv, dx, dt, ts, model) has not been tuned before.
    """
    key = '{:g},{:g},{},{},{}'.format(args.tune_cv, args.dx, DT_US, TS, IMP_MODEL)
    if os.path.exists(TUNED_FILE):
        with open(TUNED_FILE) as f:
            for line in f:
                if line.startswith(key + ','):
                    gi, ge = map(float, line.strip().split(',')[-2:])
                    print('Using cached tuning for CV={}: gi={} ge={}'.format(args.tune_cv, gi, ge))
                    return gi, ge

    # Run tuneCV in its own folder (it writes results.dat, logs and a mesh)
    tdir = os.path.join(EXAMPLE_DIR, 'tunecv_cv{:g}_dx{:g}'.format(args.tune_cv, args.dx))
    os.makedirs(tdir, exist_ok=True)
    cwd = os.getcwd()
    os.chdir(tdir)
    try:
        if os.path.isfile('results.dat'):
            os.remove('results.dat')
        cmd = [settings.execs.TUNECV,
               '--np',         args.np,
               '--resolution', args.dx * 1000.0,     # mm -> um
               '--gi',         0.174,                # tutorial starting guess
               '--ge',         0.625,
               '--ts',         TS,
               '--dt',         DT_US,
               '--tol',        0.0001,
               '--length',     0.25,                 # as in the tutorial
               '--lumping',    False,
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


def runSingleCell(args, job):
    # Single-cell state file (reused if it already exists)
    sv = os.path.join(EXAMPLE_DIR, 'CRN_bcl{:g}_{}beats.sv'.format(args.bcl, args.npls))
    if os.path.exists(sv):
        return sv

    # Duration of the simulation
    duration = args.npls*args.bcl

    # Single-cell simulation
    cmd = ['--imp',           IMP_MODEL,
           '--bcl',           args.bcl,
           '--numstim',       args.npls,
           '--duration',      duration,
           '--dt',            DT_US / 1000.0,  # ms, matches tissue dt
           '--save-ini-file', sv,
           '--save-ini-time', duration,
           '--dt-out',        duration,
           '--fout={}'.format(os.path.join(job.ID, 'bench_crn'))]

    # Run bench
    job.bench(cmd)
    return sv

def jobID(args):
    """
    Generate name of top level output directory.
    """
    today = date.today()
    state = 'pre' if args.state == 'pre' else f'post_lw{args.lesion_width:g}_lg{args.lesion_g:g}'
    cv = f'_cv{args.tune_cv:g}' if args.tune_cv else ''
    return (f'{today.isoformat()}_simple_{state}_a{args.angle}_dx{args.dx:g}{cv}'
            f'_{args.tend}_{args.flv}_np{args.np}')


def stimVertices(meshname, args, half, post, job):
    """
    Write a .vtx file of the stimulated nodes for the requested angle. Coordinates in mm.
      0 deg : strip x <= -half + w           (wave travels +x, into the boundary)
     90 deg : strip y <= -half + w           (wave travels +y, parallel to the boundary)
     45 deg : corner triangle x + y <= -2*half + CORNER_LEG
    In post, stimulated nodes are restricted to active tissue (x < 0), so the
    lesion and isolated wall are never stimulated directly.
    """
    pts = np.loadtxt(meshname + '.pts', skiprows=1) / 1000.0   # um -> mm
    x, y = pts[:, 0], pts[:, 1]
    w, eps = args.stim_width, 1e-6

    if args.angle == 0:
        mask = x <= -half + w + eps
    elif args.angle == 90:
        mask = y <= -half + w + eps
    else:
        mask = x + y <= -2*half + CORNER_LEG + eps
    if post:
        mask &= x < -eps

    idx = np.flatnonzero(mask)
    if idx.size == 0:
        raise ValueError('Stimulus electrode is empty - check --stim-width and --dx')

    os.makedirs(job.ID, exist_ok=True)
    vtx = os.path.join(job.ID, 'stim_a{}.vtx'.format(args.angle))
    with open(vtx, 'w') as f:
        f.write('{}\nintra\n'.format(idx.size))
        f.write('\n'.join(map(str, idx)) + '\n')
    return vtx


@tools.carpexample(parser, jobID)
def run(args, job):

    post = args.state == 'post'

    # --------------- Conductivities: tune (or reuse cached) or take from CLI --
    if args.tune_cv is not None:
        args.gi, args.ge = tuneCV(args, job)
    elif args.gi is None or args.ge is None:
        raise ValueError('Give either --tune-cv, or both --gi and --ge')

    # --------------- Setting up the mesh ---------------------

    # Size of the wedge (mm)
    wedge_x = 60
    wedge_y = 60
    wedge_z = 0

    # Generate mesh, units in mm, centered around origin
    # By default, elements are tagged as "1". Can be changed by adding box or sphere regions
    # See carputils doxygen documentation for details
    geom = mesh.Block(size=(wedge_x,wedge_y,wedge_z),resolution=args.dx,etype='hexa')

    # Set fibre angle to 0° for the 2D monolayer (no sheet angle in 2D)
    geom.set_fibres(0, 0)

    # Add regions ------> (0,0,0) is the center of the mesh, not the left lower corner!
    # post: x = 0 is the lesion edge (Wang's scar boundary). Non-overlapping boxes.
    if post:
        lw = args.lesion_width
        if not 0 < lw < wedge_x/2.0:
            raise ValueError('--lesion-width must be between 0 and half the sheet width')
        lesion = mesh.BoxRegion((0, -wedge_y/2.0, -1), (lw, wedge_y/2.0, 1), SCAR_TAG)
        iso    = mesh.BoxRegion((lw, -wedge_y/2.0, -1), (wedge_x/2.0, wedge_y/2.0, 1), ISO_TAG)
        geom.add_region(lesion)
        geom.add_region(iso)
    tags = [ACTIVE_TAG, SCAR_TAG, ISO_TAG] if post else [ACTIVE_TAG]

    # Generate and return base name
    meshname = mesh.generate(geom)


    # ---------------- Prepacing ---------------------------------------------

    sv = runSingleCell(args,job)


    # ---------------- Fixed settings from the state's .par ------------------
    # (ionic/conductivity regions, stimulus timing, numerics, LATs)
    cmd  = tools.carp_cmd(os.path.join(EXAMPLE_DIR, PAR_FILES[args.state]))
    cmd += tools.gen_physics_opts(IntraTags=tags)

    # ---------------- Run-dependent settings (override the .par) -----------
    # Prepaced cell state (active tissue, and isolated wall in post)
    cmd += ['-imp_region[0].im_sv_init', sv]

    # Tuned isotropic conductivities for this dx; lesion g_i scaled down
    for ax in ('l', 't', 'n'):
        cmd += ['-gregion[0].g_i' + ax, args.gi,
                '-gregion[0].g_e' + ax, args.ge]
        if post:
            cmd += ['-gregion[1].g_i' + ax, args.gi * args.lesion_g,
                    '-gregion[1].g_e' + ax, args.ge]

    # Stimulus electrode: node list for the requested angle (see stimVertices)
    vtx  = stimVertices(meshname, args, wedge_x / 2.0, post, job)
    cmd += ['-stim[0].elec.vtx_file', vtx]

    # Record where the mesh lives (needed for CV checks and the electrogram code)
    with open(os.path.join(job.ID, 'meshname.txt'), 'w') as f:
        f.write(os.path.abspath(meshname) + '\n')

    cmd += ['-tend', args.tend, '-simID', job.ID, '-meshname', meshname]

    if args.visualize:
        cmd += ['-gridout_i', 3]  # output both surface & volumetric mesh for visualization

    # Run simulation
    job.carp(cmd)

    # Do visualization
    if args.visualize and not settings.platform.BATCH:
        # Prepare file paths
        geom = os.path.join(job.ID, os.path.basename(meshname) + '_i')
        # display trensmembrane voltage
        data = os.path.join(job.ID, 'vm.igb.gz')
        # Alternatively, you could visualize the activation times instead
        # data = os.path.join(job.ID, 'init_acts_activation-thresh.dat')

        # load predefined visualization settings
        view = tools.simfile_path(os.path.join(EXAMPLE_DIR, 'initial.mshz'))

        # Call meshalyzer
        job.meshalyzer(geom, data, view)


if __name__ == '__main__':
    run()