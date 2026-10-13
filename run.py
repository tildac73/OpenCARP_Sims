#!/usr/bin/env python

import os
from datetime import date

from carputils import settings, tools, mesh

ACTIVE_TAG = 1
SCAR_TAG   = 100
IMP_MODEL  = 'Courtemanche'
PASSIVE_IMP = 'PASSIVE'

EXAMPLE_DIR = os.path.dirname(__file__)

# define parameters exposed to the user on the commandline
def parser():
    parser = tools.standard_parser()
    group  = parser.add_argument_group('experiment specific options')

    group.add_argument('--dx', type=float, default=0.25,
                       help='Mesh resolution (mm): 0.25 or 0.125')
    group.add_argument('--size', type=float, default=20.0,
                       help='Sheet edge length (mm); scar occupies x >= 0 half')
    group.add_argument('--angle', type=int, default=0, choices=[0, 45, 90],
                       help='Wavefront direction relative to scar-boundary normal (deg)')
    group.add_argument('--gi', type=float, required=True,
                       help='Tuned intracellular conductivity from tuneCV (S/m), for this dx')
    group.add_argument('--ge', type=float, required=True,
                       help='Tuned extracellular conductivity from tuneCV (S/m), for this dx')
    group.add_argument('--bcl', type=float, default=1000.0,
                       help='Single-cell prepacing cycle length (ms)')
    group.add_argument('--npls', type=int, default=20,
                       help='Number of single-cell prepacing beats')
    group.add_argument('--tend', type=float, default=250.0,
                       help='Tissue simulation duration (ms)')
    group.add_argument('--stim-width', type=float, default=0.5,
                       help='Thickness of the planar stimulus band (mm)')
    return parser

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
           '--dt',            0.025,       # ms, matches tissue dt
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
    return f'{today.isoformat()}_simple_{args.tend}_{args.flv}_np{args.np}'


@tools.carpexample(parser, jobID)
def run(args, job):

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
    tag  = 100
    scar = mesh.BoxRegion((0, -wedge_y/2.0, -1), (wedge_x/2.0, wedge_y/2.0, 1), SCAR_TAG)
    geom.add_region(scar)

    # Generate and return base name
    meshname = mesh.generate(geom)


    # ---------------- Prepacing ---------------------------------------------

    sv = runSingleCell(args,job)


    # ---------------- Fixed settings from slab2d.par ------------------------
    # (ionic/conductivity regions, stimulus timing, numerics, LATs)
    cmd  = tools.carp_cmd(os.path.join(EXAMPLE_DIR, 'slab2d.par'))
    cmd += tools.gen_physics_opts(IntraTags=[ACTIVE_TAG, SCAR_TAG])

    # ---------------- Run-dependent settings (override the .par) -----------
    # Prepaced cell state
    cmd += ['-imp_region[0].im_sv_init', sv]

    # Tuned isotropic conductivities for this dx
    for ax in ('l', 't', 'n'):
        cmd += ['-gregion[0].g_i' + ax, args.gi,
                '-gregion[0].g_e' + ax, args.ge]

    # Stimulus box (um): stim-width strip along the left edge, full height.
    # Wave travels in +x, straight into the scar boundary at x = 0.
    half = wedge_x / 2.0 * 1000.0              # mm -> um
    w    = args.stim_width * 1000.0
    tol  = 10.0                                # um, so edge nodes are included
    cmd += ['-stim[0].elec.p0[0]', -half - tol,
            '-stim[0].elec.p1[0]', -half + w,
            '-stim[0].elec.p0[1]', -half - tol,
            '-stim[0].elec.p1[1]',  half + tol,
            '-stim[0].elec.p0[2]', -1000.0,
            '-stim[0].elec.p1[2]',  1000.0]

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