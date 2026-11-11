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
        assert -1e-10 <= classes["mass_retained"] <= 1 + 1e-10                         # FM-37: a part of the total
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


def test_boundary_crossings_known_answers():
    """FM-37: crossing distances by linear interpolation; NaN when the curve never crosses or starts below the level."""
    import pandas as pd
    d = np.array([-2.0, 0.0, 2.0, 4.0, 6.0])
    falling = np.array([0.9, 0.5, 0.3, 0.1, 0.0])
    assert groundtruth.firstCrossingMm(d, falling, 0.5) == 0.0
    assert np.isclose(groundtruth.firstCrossingMm(d, falling, 0.2), 3.0)
    assert np.isclose(groundtruth.firstCrossingMm(d[::-1], falling[::-1], 0.2), 3.0)      # order of placements does not matter
    assert np.isnan(groundtruth.firstCrossingMm(d, np.full(5, 0.8), 0.5))
    assert np.isnan(groundtruth.firstCrossingMm(d, np.full(5, 0.05), 0.1))
    summary = pd.DataFrame(dict(condition=["intact"] * 5 + ["lesion"], kind="bi", label="E5-E6|E9", gapMm=1.0,
                                distanceInsideMm=list(d) + [0.0], share_retained=list(falling) + [1.0],
                                mass_retained=list(falling) + [1.0], shareAbsSum=[1.1, 1.1, 1.2, 1.1, 1.1, 1.0]))
    table = groundtruth.boundaryCrossings(summary, [0.5, 0.2], maximumShareAbsSum=2.0)
    assert len(table) == 4 and set(table.measure) == {"signedShare", "mass"} and (table.placements == 5).all()
    assert np.isclose(table[(table.measure == "mass") & (table.level == 0.2)].distanceMm.iloc[0], 3.0)
    assert (table.illConditionedPlacements == 0).all() and table.distanceMm.notna().all()
    # FM-43: one nearly null placement (sum |s_e| = 70) withholds the signed-share crossings, not the |c|-mass ones
    summary.loc[2, "shareAbsSum"] = 70.0
    table = groundtruth.boundaryCrossings(summary, [0.5, 0.2], maximumShareAbsSum=2.0)
    assert (table.illConditionedPlacements == 1).all()
    assert table[table.measure == "signedShare"].distanceMm.isna().all() and table[table.measure == "mass"].distanceMm.notna().all()


def test_signed_share_is_ill_conditioned_for_a_bipole_parallel_to_the_wavefront():
    """FM-43: planar wave along x on a box that is symmetric in y; recordings centred on the boundary y = 0.

    A bipole along the wave (x) records a large signal: little cancellation (sum |s_e| near 1), and by mirror
    symmetry half of its |c|-mass lies on either side of the boundary, whatever the tissue parameters.
    A bipole across the wave (y) records almost nothing: its signed shares are large terms of opposite sign
    (sum |s_e| far above the reporting limit), while its |c|-mass split stays at one half.
    """
    config = cfg.loadConfig(SMALL, {"wallMm": 1.0})
    positions, elements, tags = boxWithBath([8, 10], 1.0, 1.0, 0.5)
    fibres = np.where(tags[:, None] == 0, 0.0, [1.0, 0.0, 0.0])
    tissue = dict(positions=positions, elements=elements, tags=tags, fibres=fibres, tissueNodes=np.unique(elements[tags != 0]))
    p = positions[tissue["tissueNodes"]]
    t = np.arange(0, 40, 0.5)
    vm = -80 + 100 / (1 + np.exp(-(t[None, :] - 12 - (p[:, :1] + 4) / 0.9) / 0.5))
    ground = meshbuild.groundNodes(positions, elements, tags, 0.5)
    bath = meshbuild.catheterSideNodes(positions, elements, tags, 1.0)
    centres = np.array([[-1.0, 0.0, 1.0], [1.0, 0.0, 1.0], [0.0, -2.0, 1.0], [0.0, 2.0, 1.0]])
    model = leadfield.ContributionModel(config, tissue)
    c = [model.coefficients(referenceLeadField(config, tissue, ground, nodes)[tissue["tissueNodes"]])
         for nodes in electrodeNodeSets(positions, bath, centres, 0.5, 1.0)]
    retained = model.centres[:, 1] >= 0.0
    limit = config["fixed"]["groundTruth"]["maximumShareAbsSum"]
    along, alongMass, _, _ = groundtruth.shares(model, c[1] - c[0], vm)
    across, acrossMass, _, _ = groundtruth.shares(model, c[3] - c[2], vm)
    assert np.abs(along).sum() < limit < np.abs(across).sum()
    assert abs(alongMass[retained].sum() / alongMass.sum() - 0.5) < 0.03 and abs(along[retained].sum() - 0.5) < 0.03
    assert abs(acrossMass[retained].sum() / acrossMass.sum() - 0.5) < 0.03
