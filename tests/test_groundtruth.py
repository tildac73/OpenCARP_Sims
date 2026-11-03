"""Reciprocity, shares and classes on a small box with the Python lead-field route (no openCARP)."""
import numpy as np
from scipy.sparse.linalg import splu

from efov import config as cfg, groundtruth, leadfield, meshbuild
from efov.geometry import electrodeNodeSets
from helpers import SMALL, boxWithBath


def tissueModel():
    positions, elements, tags = boxWithBath([6, 6], 1.0, 1.0, 0.25)
    centres = positions[elements].mean(axis=1)
    tags = np.where((tags == 1) & (centres[:, 1] < -1.5), 5, tags)                 # a lesion strip
    fibres = np.where(tags[:, None] == 0, 0.0, [1.0, 0.0, 0.0])
    return dict(positions=positions, elements=elements, tags=tags, fibres=fibres, tissueNodes=np.unique(elements[tags != 0]))


def forwardSignal(config, tissue, ground, nodes, vm):
    """Reference: mean phi over the electrode nodes from K phi = -S_m Vm (same discrete problem as openCARP)."""
    tags, inTissue = tissue["tags"], tissue["tags"] != 0
    _, bath = leadfield.principalConductivities(config)
    bulk = np.repeat((bath * np.eye(3))[None], len(tags), axis=0)
    bulk[inTissue] = leadfield.tissueTensors(config, tags[inTissue], tissue["fibres"][inTissue], "bulk")
    source = np.zeros_like(bulk)
    source[inTissue] = leadfield.tissueTensors(config, tags[inTissue], tissue["fibres"][inTissue], "monodomain")
    K = leadfield.assemble(tissue["positions"], tissue["elements"], bulk)
    S = leadfield.assemble(tissue["positions"], tissue["elements"], source)
    free = np.setdiff1d(np.arange(len(tissue["positions"])), ground)
    solve = splu(K[free][:, free].tocsc()).solve
    full = np.zeros((len(tissue["positions"]), vm.shape[1]))
    full[tissue["tissueNodes"]] = vm
    phi = np.zeros_like(full)
    phi[free] = solve(-(S @ full)[free])
    return phi[nodes].mean(axis=0)


def test_contributions_reproduce_the_forward_signal_and_classes_partition():
    config = cfg.loadConfig(SMALL, {"wallMm": 1.0})
    tissue = tissueModel()
    p = tissue["positions"][tissue["tissueNodes"]]
    t = np.linspace(0, 30, 61)
    vm = -80 + 100 / (1 + np.exp(-(t[None, :] - 10 - (p[:, :1] + 3) / 0.9) / 0.5))          # planar wave in +x
    ground = meshbuild.groundNodes(tissue["positions"], tissue["elements"], tissue["tags"], 0.5)
    bath = np.flatnonzero(tissue["positions"][:, 2] > 0.5)
    sets = electrodeNodeSets(tissue["positions"], bath, np.array([[0.0, 0.0, 1.0], [0.0, 1.0, 1.0]]), 0.3, 0.6)
    solver = leadfield.PythonLeadFieldSolver(config, tissue, ground)
    za, zb = solver.leadField(sets[0]), solver.leadField(sets[1])
    np.testing.assert_allclose(za[sets[1]].mean(), zb[sets[0]].mean(), rtol=1e-9)           # reciprocity
    model = leadfield.ContributionModel(config, tissue)
    for weights in ([1.0, 0.0], [1.0, -1.0]):
        coefficients = weights[0] * model.coefficients(za) + weights[1] * model.coefficients(zb)
        shares, mass, phi, total = groundtruth.shares(model, coefficients, vm)
        direct = weights[0] * forwardSignal(config, tissue, ground, sets[0], vm) + weights[1] * forwardSignal(config, tissue, ground, sets[1], vm)
        np.testing.assert_allclose(phi, direct, rtol=1e-7, atol=1e-9 * np.abs(direct).max())
        assert abs(total - 1) < 1e-10 and np.all(mass >= 0)
        lateral, depth, layer = groundtruth.elementGeometry(model, config, [0.0, 0.0])
        classes = groundtruth.classSummary(model, config, shares, mass, lateral, depth)
        assert abs(sum(classes[f"share_{c}"] for c in groundtruth.CLASSES) - 1) < 1e-10
        assert abs(sum(classes[f"mass_{c}"] for c in groundtruth.CLASSES) - 1) < 1e-10
        assert layer.min() == 0 and layer.max() == 3                                         # 1 mm wall, 0.25 mm layers
        shape = groundtruth.shapeDescriptors(model, shares, lateral, depth, [0.0, 0.0], [0.5, 0.9])
        assert shape["p50_volumeMm3"] <= shape["p90_volumeMm3"]
