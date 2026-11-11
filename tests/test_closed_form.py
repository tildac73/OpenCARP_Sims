"""Closed-form anchors for the P1 operators behind the lead fields, the electrograms and the test oracle.

The oracle in helpers.py assembles its matrices with efov.leadfield, so agreement between the pipeline and
the oracle cannot detect an error they share (tensor construction, units, sign). These tests compare the
same operators with values worked out by hand; both cases are exact for linear finite elements.

  1. Source operator. For Vm = g . x and a lead field z = h . x on tissue of volume V with intracellular
     tensor sigma_i (principal values gi_l along the fibre, gi_t across), the electrogram
     -z^T S_i Vm = -V h^T sigma_i g. With fibres along x in one half of the wall and along y in the other,
     and g = h = (1, 0, 0): -(V/2) (gi_l + gi_t). Units (efov.leadfield): mm, S/m, mV and mV/uA give mV.
  2. Bulk operator and current normalisation. A bar of blood (conductivity g_bath, cross-section A) with
     1 uA entering at one end and the other end grounded carries a uniform current density, so the lead
     field falls by dx / (g_bath A) mV/uA over a length dx (mm, S/m, mm^2).
"""
import numpy as np

from efov import config as cfg, leadfield
from helpers import SMALL, boxMesh, boxWithBath, referenceLeadField


def test_source_operator_matches_the_closed_form_for_linear_fields():
    config = cfg.loadConfig(SMALL, {"wallMm": 1.0})
    positions, elements, tags = boxWithBath([4, 3], 1.0, 0.5, 0.5)
    centres = positions[elements].mean(axis=1)
    fibres = np.where(tags[:, None] == 0, 0.0, [1.0, 0.0, 0.0])
    fibres[(tags != 0) & (centres[:, 2] < 0)] = [0.0, 1.0, 0.0]
    tissue = dict(positions=positions, elements=elements, tags=tags, fibres=fibres, tissueNodes=np.unique(elements[tags != 0]))
    model = leadfield.ContributionModel(config, tissue)
    x = positions[tissue["tissueNodes"]]
    giL, giT = cfg.par(config, "gregion[0].g_il"), cfg.par(config, "gregion[0].g_it")
    volume = 4.0 * 3.0 * 1.0
    for g, h, expected in (((1, 0, 0), (1, 0, 0), -(volume / 2) * (giL + giT)),      # along x: longitudinal in one half, transverse in the other
                           ((0, 1, 0), (0, 1, 0), -(volume / 2) * (giT + giL)),
                           ((0, 0, 1), (0, 0, 1), -volume * giT),                    # through the wall: transverse everywhere
                           ((1, 0, 0), (0, 1, 0), 0.0)):                             # principal axes: no cross term
        vm, z = 7.0 * (x @ np.array(g, float)) - 80.0, 3.0 * (x @ np.array(h, float)) + 0.2
        signal = model.nodalWeights(z) @ vm
        assert abs(signal - 21.0 * expected) < 1e-9 * 21.0 * volume * giL
        assert abs(model.coefficients(z).ravel() @ vm[model.localElements].ravel() - signal) < 1e-9 * 21.0 * volume * giL


def test_bulk_operator_and_unit_current_match_a_uniform_bar():
    config = cfg.loadConfig(SMALL, {"wallMm": 1.0})
    positions, elements = boxMesh([0, 0, 0], [12, 2, 2], 0.5)
    tissue = dict(positions=positions, elements=elements, tags=np.zeros(len(elements), int), fibres=np.zeros((len(elements), 3)),
                  tissueNodes=np.array([], int))
    source = np.flatnonzero(np.isclose(positions[:, 0], 0.0))
    ground = np.flatnonzero(np.isclose(positions[:, 0], 12.0))
    z = referenceLeadField(config, tissue, ground, source)
    bath, area = cfg.par(config, "gregion[2].g_bath"), 2.0 * 2.0
    mean = lambda xMm: z[np.isclose(positions[:, 0], xMm)].mean()
    assert abs((mean(4.0) - mean(9.0)) - 5.0 / (bath * area)) < 1e-6 * 5.0 / (bath * area)
    assert abs(mean(9.0) - 3.0 / (bath * area)) < 1e-6 * 3.0 / (bath * area)        # absolute level: 3 mm from the ground
    assert z.min() >= -1e-12                                                         # a current source raises the potential (sign)
