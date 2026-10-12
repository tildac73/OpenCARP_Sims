#!/usr/bin/env python

import os

from carputils.model import ConductivityRegion, optionlist
from carputils.model.ionic import CourtemancheIonicModel
from carputils.model.lat import Lat

EXAMPLE_DESCRIPTIVE_NAME = 'Initial 2D Simulation'
EXAMPLE_AUTHOR = 'Andrew Crozier <andrew.crozier@medunigraz.at>, Axel Loewe <axel.loewe@kit.edu>'
EXAMPLE_DIR = os.path.dirname(__file__)

from datetime import date

# import required carputils modules
from carputils import settings
from carputils import tools
from carputils import mesh
from carputils import testing


# define parameters exposed to the user on the commandline
def parser():
    parser = tools.standard_parser()
    parser.add_argument('--tend',
                        type=float, default=20.,
                        help='Duration of simulation (ms). Run for longer to also see repolarization.')
    return parser


def jobID(args):
    """
    Generate name of top level output directory.
    """
    today = date.today()
    return f'{today.isoformat()}_simple_{args.tend}_{args.flv}_np{args.np}'


@tools.carpexample(parser, jobID)
def run(args, job):
    # Generate mesh, units in mm, centered around origin
    # By default, elements are tagged as "1". Can be changed by adding box or sphere regions
    # See carputils doxygen documentation for details
    geom = mesh.Block(size=(2, 0.5, 0))

    # Set fibre angle to 0° for the 2D monolayer (no sheet angle in 2D)
    geom.set_fibres(0, 0)

    # Generate and return base name
    meshname = mesh.generate(geom)

    # Instead of generating a mesh, you could also load one using the 'meshname' openCARP
    # command. This requires .elem and .pts files that can be generated for VTK files using
    # meshtool 

    # Get basic command line, including solver options from external .par file
    cmd = tools.carp_cmd(tools.bundle_include(os.path.join(EXAMPLE_DIR, 'initial.par')))

    # Attach electrophysiology physics (monodomain) to mesh region with tag 1
    cmd += tools.gen_physics_opts(IntraTags=[1])

    cmd += ['-simID', job.ID,
            '-meshname', meshname,
            '-tend', args.tend]

    # Set monodomain conductivities
    region = ConductivityRegion(IDs=[1],  # use these settings for the mesh region with tag 1
                                name="myocardium",
                                g_el=0.625,  # extracellular conductivity in longitudinal direction
                                g_et=0.236,  # extracellular conductivity in transverse direction
                                g_en=0.236,  # extracellular conductivity in sheet direction
                                g_il=0.174,  # intracellular conductivity in longitudinal direction
                                g_it=0.019,  # intracellular conductivity in transverse direction
                                g_in=0.019,  # intracellular conductivity in sheet direction
                                g_mult=0.5)  # scale all conducitivites by a factor (to reduce conduction velocity)

    # Define the ionic model to use
    courtemanche_setup = CourtemancheIonicModel(IDs=[1])

    # compute local activation times in postprocessing
    lat = Lat(ID='activation',
              all=0,  # only detect first activation
              measurand=0,  # determine LAT from transmembrane voltage
              mode=0,  # take maximum slope to detemine LAT
              threshold=-10)  # -10mV threshold

    cmd += optionlist([region, courtemanche_setup, lat])
    # Define the geometry of the first stimulus (index 0) at end of the block
    # (upper bound of x coordinate), units in um
    stim = mesh.block_bc_opencarp(geom, 'stim', 0, 'x', True)
    cmd += stim

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
