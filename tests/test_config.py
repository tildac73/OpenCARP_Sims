import json

import numpy as np
import pytest

from efov import config as cfg
from helpers import MODEL, SMALL


def test_model_config_is_valid_and_matches_the_plan():
    c = cfg.loadConfig(MODEL)
    assert c["values"]["wallMm"] == 1.9 and c["values"]["fibres"]["model"] == "bilayer"
    assert c["values"]["scar"] == "ablation" and c["parOverrides"] == {}
    np.testing.assert_allclose(cfg.spacingMm(c), [0.25, 0.25, 0.1])
    assert cfg.beatCount(c) == 4 and cfg.cycleLengthMs(c) == 500 and cfg.sampleStepMs(c) == 0.2
    assert cfg.par(c, "bidomain", int) == 0 and c["fixed"]["tissue"]["lesionWidthMm"] == 5.6
    assert "calibration" not in c["fixed"]["analysis"]


def test_small_config_is_valid():
    cfg.loadConfig(SMALL)


def test_overrides_and_scar_options():
    c = cfg.loadConfig(MODEL, {"scar": "ablationPorated", "wallMm": 2.0})
    assert c["parOverrides"]["gregion[1].g_el"] == 0.461 and cfg.par(c, "gregion[1].g_el") == 0.461
    assert cfg.slabSizeMm(c)[2] == 2.0
    with pytest.raises(KeyError):
        cfg.loadConfig(MODEL, {"notAVariable": 1})


@pytest.mark.parametrize("overrides, message", [
    ({"wallMm": 1.05}, "multiple of its mesh spacing"),
    ({"waveAngleDeg": 120.0}, "waveAngleDeg"),
    ({"epicardialMedium": "lung"}, "epicardialMedium"),
    ({"gridSpacingMm": 0.9}, "overlap"),
])
def test_invalid_settings_are_rejected(overrides, message):
    with pytest.raises(ValueError, match=message):
        cfg.loadConfig(MODEL, overrides)


def test_fingerprints_separate_simulation_from_analysis():
    base = cfg.loadConfig(MODEL)
    def chain(c):
        return cfg.fingerprint(c, "simulation", [cfg.fingerprint(c, "tune", [cfg.fingerprint(c, "conditioning")])])
    assert chain(cfg.loadConfig(MODEL, {"gapsMm": [2.0]})) == chain(base)          # electrode variant reuses the simulation
    assert chain(cfg.loadConfig(MODEL, {"noiseSdMv": 0.01})) == chain(base)
    assert chain(cfg.loadConfig(MODEL, {"wallMm": 2.0})) != chain(base)
    assert chain(cfg.loadConfig(MODEL, {"scar": "insulating"})) != chain(base)
    assert cfg.fingerprint(cfg.loadConfig(MODEL, {"gapsMm": [2.0]}), "leadFields") != cfg.fingerprint(base, "leadFields")
    tuned = cfg.withParOverrides(base, {"imp_region[0].cellSurfVolRatio": 0.03})
    assert chain(tuned) == chain(base)                                           # beta is the tune stage's output


def test_lead_fields_are_shared_by_variants_that_keep_the_conductor():
    base = cfg.fingerprint(cfg.loadConfig(MODEL), "conductor")
    for same in ({"waveAngleDeg": 60.0}, {"cvTargetMPerS": 0.7}, {"gapsMm": [2.0]}, {"noiseSdMv": 0.01}):
        assert cfg.fingerprint(cfg.loadConfig(MODEL, same), "conductor") == base, same
    for different in ({"wallMm": 2.0}, {"scar": "insulating"}, {"fibres": {"model": "uniform", "angleDeg": 0.0}},
                      {"epicardialMedium": "blood"}):
        assert cfg.fingerprint(cfg.loadConfig(MODEL, different), "conductor") != base, different


def test_tuning_record_round_trip(tmp_path):
    c = cfg.loadConfig(SMALL)
    record = tmp_path / "tuning.json"
    assert cfg.tunedBetaFor(c, record) is None
    cfg.recordTuning(c, dict(cellSurfVolRatioPerUm=0.025), record)
    assert cfg.tunedBetaFor(c, record) == 0.025
    assert cfg.tunedBetaFor(cfg.loadConfig(SMALL, {"cvTargetMPerS": 0.7}), record) is None
