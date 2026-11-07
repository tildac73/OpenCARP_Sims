"""Reciprocity, electrogram weights, shares and classes on a small box (no openCARP: the lead-field runs are
replaced by the test oracle in helpers; tests/test_opencarp_leadfield.py checks openCARP >= 20 against it)."""
import numpy as np

from efov import config as cfg, groundtruth, leadfield, meshbuild
from efov.geometry import electrodeNodeSets
from helpers import SMALL, boxWithBath, patchLeadfield, referenceLeadField, weightedElectrogram


def tissueModel():
    positions, elements, tags = boxWithBath([6, 6], 1.0, 1.0, 0.25)
    centres = positions[elements].mean(axis=1)
    tags = np.where((tags == 1) & (centres[:, 1] < -1.5) & (centres[:, 1] > -2.5), 5, tags)   # a lesion strip, isolated tissue beyond
    fibres = np.where(tags[:, None] == 0, 0.0, [1.0, 0.0, 0.0])
    fibres[(tags != 0) & (centres[:, 2] < 0)] = [0.0, 1.0, 0.0]                                 # bilayer: sigma_i and sigma_m differ
    return dict(positions=positions, elements=elements, tags=tags, fibres=fibres, tissueNodes=np.unique(elements[tags != 0]))


def test_reciprocity_weights_and_shares_with_the_sigma_i_source():
    config = cfg.loadConfig(SMALL, {"wallMm": 1.0})
    tissue = tissueModel()
    p = tissue["positions"][tissue["tissueNodes"]]
    t = np.linspace(0, 30, 61)
    vm = -80 + 100 / (1 + np.exp(-(t[None, :] - 10 - (p[:, :1] + 3) / 0.9) / 0.5))          # planar wave in +x
    ground = meshbuild.groundNodes(tissue["positions"], tissue["elements"], tissue["tags"], 0.5)
    bath = meshbuild.catheterSideNodes(tissue["positions"], tissue["elements"], tissue["tags"], 1.0)
    sets = electrodeNodeSets(tissue["positions"], bath, np.array([[0.0, 0.0, 1.0], [0.0, 1.0, 1.0]]), 0.3, 0.6)
    za, zb = (referenceLeadField(config, tissue, ground, nodes)[tissue["tissueNodes"]] for nodes in sets)   # what the cache holds
    model = leadfield.ContributionModel(config, tissue)
    reference = {k: weightedElectrogram(config, tissue, ground, sets[k], vm) for k in (0, 1)}
    np.testing.assert_allclose(model.nodalWeights(za) @ vm, reference[0], rtol=1e-7, atol=1e-9 * np.abs(reference[0]).max())
    # the sigma_m source (openCARP pseudo-bidomain) gives a different signal on this bilayer slab
    sigmaM = weightedElectrogram(config, tissue, ground, sets[0], vm, "monodomain")
    assert np.linalg.norm(sigmaM - reference[0]) > 0.05 * np.linalg.norm(reference[0])
    coefficients = model.coefficients(za)
    np.testing.assert_allclose(coefficients.sum(axis=1), 0, atol=1e-12 * np.abs(coefficients).max())   # uniform Vm contributes nothing
    for weights in ([1.0, 0.0], [1.0, -1.0]):
        combined = weights[0] * model.coefficients(za) + weights[1] * model.coefficients(zb)
        shares, mass, phi, total = groundtruth.shares(model, combined, vm)
        direct = weights[0] * reference[0] + weights[1] * reference[1]
        np.testing.assert_allclose(phi, direct, rtol=1e-7, atol=1e-9 * np.abs(direct).max())
        assert abs(total - 1) < 1e-10 and np.all(mass >= 0)
        lateral, depth, layer = groundtruth.elementGeometry(model, config, [0.0, 0.0])
        classes = groundtruth.classSummary(model, config, shares, mass, lateral, depth)
        assert abs(sum(classes[f"share_{c}"] for c in groundtruth.CLASSES) - 1) < 1e-10
        assert abs(sum(classes[f"mass_{c}"] for c in groundtruth.CLASSES) - 1) < 1e-10
        assert layer.min() == 0 and layer.max() == 3                                         # 1 mm wall, 0.25 mm layers
        shape = groundtruth.shapeDescriptors(model, shares, lateral, depth, [0.0, 0.0], [0.5, 0.9])
        assert shape["p50_volumeMm3"] <= shape["p90_volumeMm3"]


def test_lead_field_cache_reuses_and_refuses(tmp_path, monkeypatch):
    import pytest
    from efov.io import writeIgb
    config = cfg.loadConfig(SMALL, {"wallMm": 1.0})
    tissue = dict(tissueModel(), meshBase="box")
    patchLeadfield(monkeypatch, {"box": tissue})
    ground = meshbuild.groundNodes(tissue["positions"], tissue["elements"], tissue["tags"], 0.5)
    bath = meshbuild.catheterSideNodes(tissue["positions"], tissue["elements"], tissue["tags"], 1.0)
    sets = electrodeNodeSets(tissue["positions"], bath, np.array([[0.0, 0.0, 1.0], [0.0, 1.0, 1.0]]), 0.3, 0.6)
    vm = np.random.default_rng(0).normal(-80, 5, (5, len(tissue["tissueNodes"])))
    writeIgb(tmp_path / "vm.igb", vm, 0.0, 0.2)
    args = (config, tissue, ground, sets, tmp_path / "vm.igb", tmp_path / "cache")
    paths, constructed, times, traces = leadfield.leadFieldsAndTraces(*args, tmp_path / "run1", log=lambda *_: None)
    assert constructed == 2 and all(p.exists() for p in paths) and traces.shape == (2, 5) and np.allclose(times, np.arange(5) * 0.2)
    assert leadfield.loadLeadField(paths[0]).shape == (len(tissue["tissueNodes"]),)
    again = leadfield.leadFieldsAndTraces(*args, tmp_path / "run2", log=lambda *_: None)
    assert again[1] == 0 and np.abs(again[3] - traces).max() < 1e-4 * np.abs(traces).max()   # from the cache (float32 LF files)
    with pytest.raises(ValueError, match="different mesh"):
        leadfield.leadFieldsAndTraces(config, tissue, ground[:-1], sets, tmp_path / "vm.igb", tmp_path / "cache", tmp_path / "run3")
