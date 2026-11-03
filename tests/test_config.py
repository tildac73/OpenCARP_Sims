import copy
import json

import numpy as np
import pytest

from efov import config as cfg
from helpers import finalConfig, smallConfig


def test_final_config_is_valid_and_matches_the_decided_model():
    c = cfg.loadConfig()
    assert cfg.slabSizeMm(c).tolist() == [40.0, 60.0, 1.5]
    assert c["tissue"]["conductivitiesSPerM"] == dict(g_il=0.344, g_it=0.06, g_in=0.06, g_el=0.117, g_et=0.08, g_en=0.08)
    assert c["bath"] == {**c["bath"], "thicknessMm": 5.0, "allSides": True, "groundCubeMm": 1.0, "bloodSPerM": 0.74}
    assert c["lesion"]["scar"] == "insulating" and c["numerics"]["forwardModel"] == "pseudoBidomain"
    assert cfg.meshNodeCount(c) == 341_784                  # README estimate for the 0.5 mm model
    assert cfg.restartTimeMs(c) == 1050.0                   # DL-2026-11-01-06
    assert len(cfg.centreYmm(c)) == 65
    np.testing.assert_allclose(c["tissue"]["cellSurfVolRatioPerUm"], 0.14 / 6.42, rtol=3e-3)


def test_small_config_is_valid():
    cfg.validateConfig(smallConfig())


@pytest.mark.parametrize("path, value, message", [
    ("tissue.spacingMm", 0.3, "multiple"),
    ("numerics.phieOutputHeightMm", 1.0, "phieOutputHeightMm"),
    ("analysis.differentialBandHz", [30.0, 2600.0], "Nyquist"),
    ("lesion.scar", "conducting", "insulating"),
    ("pacing.beatCount", 2, "beatCount"),
])
def test_invalid_settings_are_rejected(path, value, message):
    c = finalConfig()
    *parents, last = path.split(".")
    target = c
    for key in parents:
        target = target[key]
    target[last] = value
    with pytest.raises(ValueError, match=message):
        cfg.validateConfig(c)


def test_fingerprint_follows_inputs_and_parents():
    c = finalConfig()
    base = cfg.fingerprint(c, "simulation", ["p"])
    assert cfg.fingerprint(c, "simulation", ["p"]) == base
    assert cfg.fingerprint(c, "simulation", ["q"]) != base
    changed = copy.deepcopy(c); changed["bath"]["bloodSPerM"] = 0.7
    assert cfg.fingerprint(changed, "simulation", ["p"]) != base
    documented = copy.deepcopy(c); documented["bath"]["_about"] = "edited text"
    assert cfg.fingerprint(documented, "simulation", ["p"]) == base      # documentation does not invalidate
    analysis = copy.deepcopy(c); analysis["changePoints"]["stepMm"] = 0.1
    assert cfg.fingerprint(analysis, "simulation", ["p"]) == base        # analysis changes keep the simulation


def test_import_production_model(tmp_path):
    production = dict(createdUtc="2026-11-03T00:00:00+00:00", domainMm=[60.0, 90.0, 1.5], spacingMm=0.25,
                      cellSurfVolRatioPerUm=0.0257, conductivitiesSPerM=finalConfig()["tissue"]["conductivitiesSPerM"],
                      beatCount=8, stimulusStrength=100.0, timeStepUs=10.0, sampleStepMs=0.1)
    (tmp_path / "productionModel.json").write_text(json.dumps(production))
    c = cfg.importProductionModel(tmp_path / "productionModel.json", outputPath=tmp_path / "config.json")
    assert cfg.slabSizeMm(c).tolist() == [60.0, 90.0, 1.5] and c["tissue"]["spacingMm"] == 0.25
    assert c["provenance"]["tissue.spacingMm"]["status"] == "confirmed"
    assert set(cfg.unconfirmed(c)) == {"analysis.calibration.targetBipolarMv"}
    cfg.loadConfig(tmp_path / "config.json")
