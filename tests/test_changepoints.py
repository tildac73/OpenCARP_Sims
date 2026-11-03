import numpy as np
import pandas as pd

from efov import changepoints
from helpers import finalConfig

SETTINGS = finalConfig()["changePoints"]


def test_exponential_decay_to_plateau_is_recovered():
    d = np.arange(0.5, 14.6, 0.5)
    pp = np.exp(np.log(1.5) - 0.4 * np.minimum(d, 6.0))
    _, fits, _ = changepoints.fitProfile(d, pp, SETTINGS)
    # Model selection uses an unbroken line (as the clinical Rmd), so either family may be selected;
    # the exponential fit must still find the generating change point.
    exponential = next(f for f in fits if f["model"] == "Exponential")
    assert exponential["fitStatus"] == "supportedDescriptiveChangePoint"
    assert abs(exponential["changePointMm"] - 6.0) <= 0.05
    assert next(f for f in fits if f["selected"])["descriptiveSupport"]


def test_flat_and_short_profiles_are_not_supported():
    d = np.arange(0.5, 14.6, 0.5)
    flat = 0.5 * np.exp(0.01 * np.random.default_rng(2).normal(size=len(d)))
    _, fits, _ = changepoints.fitProfile(d, flat, SETTINGS)
    assert not any(f.get("descriptiveSupport") for f in fits)
    _, fits, _ = changepoints.fitProfile(d[:5], flat[:5], SETTINGS)
    assert fits == [dict(fitStatus="insufficientDetectedDistances", count=5, selected=True)]


def test_eligibility_needs_prerequisites_and_a_quiet_intact_control(tmp_path):
    d = np.arange(-8, 14.6, 0.5)
    rows = []
    for condition, profile in [("lesion", np.exp(-0.4 * np.minimum(np.clip(d, 0, None), 6.0))), ("intact", np.ones_like(d))]:
        rows += [dict(condition=condition, gapMm=0.25, noiseRepeat=0, kind="uni", label="E5", ppMv=float(v),
                      distanceInsideMm=float(x), cliqueDistanceInsideMm=float(x)) for x, v in zip(d, profile)]
    points = pd.DataFrame(rows)
    fits = changepoints.primaryFits(changepoints.runChangePoints(finalConfig(), points, tmp_path, True))
    assert fits[fits.condition == "lesion"].eligibleForInterpretation.all()
    fits = changepoints.primaryFits(changepoints.runChangePoints(finalConfig(), points, tmp_path, False))
    assert not fits.eligibleForInterpretation.any()
