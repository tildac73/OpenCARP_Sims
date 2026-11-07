"""openCARP's Laplace solve (the lead fields, FM-34) against an independent direct solve of the same discrete problem.

Small Kuhn box with bath, a lesion strip and bilayer fibres (sigma_i and sigma_e anisotropies differ), the
corner ground and one electrode-sized node set in the bath. openCARP holds the electrode at 1 mV;
the reference solves K u = 0 with the same boundary values, K = sigma_i + sigma_e in tissue and lesion
and g_bath in the bath. Skipped when openCARP is not installed.
"""
import shutil

import numpy as np
import pytest

from efov import config as cfg, leadfield, meshbuild
from efov.geometry import electrodeNodeSets
from efov.io import writeMesh
from helpers import SMALL, referenceLaplacePotential
from test_groundtruth import tissueModel

pytestmark = pytest.mark.skipif(shutil.which("openCARP") is None, reason="openCARP not installed")


def test_opencarp_laplace_solves_the_bidomain_extracellular_operator(tmp_path):
    config = cfg.loadConfig(SMALL, {"wallMm": 1.0})
    tissue = tissueModel()
    corners = tissue["positions"][tissue["elements"]]
    negative = np.einsum("ei,ei->e", np.cross(corners[:, 1] - corners[:, 0], corners[:, 2] - corners[:, 0]), corners[:, 3] - corners[:, 0]) < 0
    tissue["elements"][negative] = tissue["elements"][negative][:, [1, 0, 2, 3]]   # openCARP needs positively oriented tetrahedra
    meshBase = writeMesh(tmp_path / "mesh" / "box", tissue["positions"], tissue["elements"], tissue["tags"], tissue["fibres"])
    ground = meshbuild.groundNodes(tissue["positions"], tissue["elements"], tissue["tags"], 0.5)
    bath = meshbuild.catheterSideNodes(tissue["positions"], tissue["elements"], tissue["tags"], 1.0)
    nodes = electrodeNodeSets(tissue["positions"], bath, np.array([[0.0, 0.0, 1.0]]), 0.3, 0.6)[0]
    u = leadfield.laplacePotential(config, meshBase, ground, nodes, len(tissue["positions"]), tmp_path / "run")
    reference = referenceLaplacePotential(config, tissue, ground, nodes, leadfield.electrodePotentialMv(config))
    assert np.linalg.norm(u - reference) / np.linalg.norm(reference) < 1e-6
    z, current = leadfield.leadFieldFromPotential(u, leadfield.bulkStiffness(config, tissue), nodes, ground,
                                                  leadfield.electrodePotentialMv(config))
    assert current > 0 and np.isfinite(z).all()
