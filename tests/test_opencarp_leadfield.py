"""openCARP's lead-field module (openCARP >= 20; FM-35) against an independent direct solve of the same discrete problem.

Small Kuhn box with bath, a lesion strip and bilayer fibres (sigma_i and sigma_e anisotropies differ), the corner
ground and two electrode-sized node sets in the bath, and a synthetic planar-wave Vm. Checks:
  * openCARP's lead fields (LF_<name>.igb, tissue nodes) equal the oracle's (1 uA spread by the mass matrix over
    the electrode, normalised by its volume; physics/leadfield.cc), relative L2 < 1e-6;
  * openCARP's electrograms (ECG_<name>.dat) equal -Z^T S_i Vm with the sigma_i tensors, relative L2 < 1e-4.
Skipped when openCARP is missing or has no lead-field module.
"""
import shutil

import numpy as np
import pytest

from efov import config as cfg, leadfield, meshbuild
from efov.geometry import electrodeNodeSets
from efov.io import writeIgb, writeMesh
from helpers import SMALL, referenceLeadField
from test_groundtruth import tissueModel

pytestmark = pytest.mark.skipif(shutil.which("openCARP") is None or not leadfield.supportsLeadfield(shutil.which("openCARP") or "openCARP"),
                                reason="openCARP >= 20 (lead-field module) not installed")


def test_opencarp_leadfield_module_matches_the_direct_solve(tmp_path):
    config = cfg.loadConfig(SMALL, {"wallMm": 1.0})
    tissue = tissueModel()
    corners = tissue["positions"][tissue["elements"]]
    negative = np.einsum("ei,ei->e", np.cross(corners[:, 1] - corners[:, 0], corners[:, 2] - corners[:, 0]), corners[:, 3] - corners[:, 0]) < 0
    tissue["elements"][negative] = tissue["elements"][negative][:, [1, 0, 2, 3]]   # openCARP needs positively oriented tetrahedra
    meshBase = writeMesh(tmp_path / "mesh" / "box", tissue["positions"], tissue["elements"], tissue["tags"], tissue["fibres"])
    ground = meshbuild.groundNodes(tissue["positions"], tissue["elements"], tissue["tags"], 0.5)
    bath = meshbuild.catheterSideNodes(tissue["positions"], tissue["elements"], tissue["tags"], 1.0)
    sets = electrodeNodeSets(tissue["positions"], bath, np.array([[0.0, 0.0, 1.0], [0.0, 1.0, 1.0]]), 0.3, 0.6)
    p = tissue["positions"][tissue["tissueNodes"]]
    t = np.arange(0, 30.01, 0.2)
    vm = -80 + 100 / (1 + np.exp(-(t[:, None] - 10 - (p[None, :, 0] + 3) / 0.9) / 0.5))
    writeIgb(tmp_path / "vm.igb", vm, 0.0, 0.2)
    leads = [(leadfield.leadName(s), s) for s in sets]
    fields, traces = leadfield.runOpenCarpLeadfield(config, meshBase, ground, leads, tmp_path / "vm.igb", tmp_path / "run", 0.2)
    model = leadfield.ContributionModel(config, tissue)
    for name, nodes in leads:
        z = leadfield.loadLeadField(fields[name])
        reference = referenceLeadField(config, tissue, ground, nodes)[tissue["tissueNodes"]]
        assert np.linalg.norm(z - reference) / np.linalg.norm(reference) < 1e-6
        times, values = leadfield.readEcg(traces[name])
        expected = vm @ model.nodalWeights(z)
        assert len(values) == len(t) and np.linalg.norm(values - expected) / np.linalg.norm(expected) < 1e-4
