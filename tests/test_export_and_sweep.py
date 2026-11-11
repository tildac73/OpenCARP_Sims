import pandas as pd

from efov import export, sweep
from helpers import ROOT


def test_wang_rows_follow_her_column_conventions():
    points = pd.DataFrame(dict(kind=["uni", "bi", "omni"], distanceInsideMm=[2.0, -1.0, 5.0], ppMv=[1.0, 0.1, 0.05],
                               pfHz=[float("nan"), 300.0, float("nan")], label=["E5", "E5-E6|E9", "E5-E6-E9"], placement=[0, 0, 0],
                               gapMm=[0.25] * 3, noiseRepeat=[0] * 3, ppWindow=["window", "peakFrequency", "window"],
                               pfGlobalHz=[float("nan"), 290.0, float("nan")]))
    rows = export.wangRows(points, "roof", 3.0)
    assert list(rows.uni1_bi2_omni3) == [1, 2, 3] and list(rows.DistAcrossPath) == [-1.0, -4.0, 2.0]
    assert rows.inf0_roof1.unique().tolist() == [1] and rows.patientID.unique().tolist() == [1]
    assert pd.isna(rows.PFreq[0]) and rows.PFreq[1] == 300.0 and rows.PFreq[2] == 0.0   # undetectable PF = 0
    assert export.wangRows(points, "roof", 3.0, "pfGlobalHz").PFreq[1] == 290.0
    assert export.parseDatasetName(export.datasetName(0.25, 2, "global")) == (0.25, 2, "global")


def test_sweep_variants_from_the_plan():
    plan = sweep.variants({"oneAtATime": ["wallMm", "scar"], "named": {"roof": {"region": "roof"}}}, ROOT / "config" / "model.json")
    assert "baseline" in plan and plan["roof"] == {"region": "roof"}
    assert plan["wallMm=1.0"] == {"wallMm": 1.0} and plan["scar=insulating"] == {"scar": "insulating"}


def test_predicted_noise_shift_and_comparison_table(tmp_path):
    """FM-38: shift predicted if the plateau is noise; the comparison table joins Wang's tables and the ground truth."""
    import math
    from types import SimpleNamespace
    from efov import report
    assert math.isclose(report.predictedShiftMm("Exponential", -0.25, 9.96), -math.log(2) / 0.25)
    assert math.isclose(report.predictedShiftMm("Power-law", -0.5, 12.0), 12.0 * (2 ** -2 - 1))
    assert math.isnan(report.predictedShiftMm("Exponential", 0.1, 5.0))
    out = tmp_path; (out / "changePoints").mkdir(); (out / "groundTruth").mkdir()
    # one gap: a noise-free dataset (repeat 0) and three noisy realisations with change points 9, 10 and 12 mm
    datasets = [export.datasetName(1.0, repeat, "ensite") for repeat in range(4)]
    cp = [20.0, 9.0, 10.0, 12.0]
    pd.DataFrame(dict(dataset=datasets, Location="Inferior", Metric="Voltage", Electrode="Bipolar", cp_mm_mean=cp,
                      cp_mm_lower=[c - 1 for c in cp], cp_mm_upper=[c + 1 for c in cp], value_at_cp_mean=0.04)
                 ).to_csv(out / "changePoints" / "SuppResults5_change_points_all.csv", index=False)
    pd.DataFrame(dict(dataset=datasets * 2, Location="Inferior", Metric="Voltage", Electrode="Bipolar", model="Exponential",
                      term=["beta1"] * 4 + ["cp"] * 4, mean=[-0.25] * 4 + cp)
                 ).to_csv(out / "changePoints" / "Table5_bayesian_posteriors_all.csv", index=False)
    pd.DataFrame(dict(kind="bi", label=["a", "b", "a", "b"], gapMm=1.0, measure=["mass", "mass", "signedShare", "signedShare"], level=0.1,
                      distanceMm=[3.0, 5.0, float("nan"), float("nan")], placements=9, illConditionedPlacements=[0, 0, 2, 3])
                 ).to_csv(out / "groundTruth" / "boundaryCrossings.csv", index=False)
    table = report.efovComparison(SimpleNamespace(out=out))
    assert len(table) == 1 and (out / "efovComparison.csv").exists()
    row = table.iloc[0]
    # FM-45: the row compared with Wang is the median over the noisy realisations; the noise-free fit is a reference column
    assert row.cp_mm_noisy_median == 10.0 and row.cp_mm_noisy_min == 9.0 and row.cp_mm_noisy_max == 12.0 and row.noisyRealisations == 3
    assert row.cp_mm_clean == 20.0 and math.isclose(row.logDropAtCp, 2.5)
    assert math.isclose(row.predictedShiftPerNoiseDoublingMm, -math.log(2) / 0.25)
    # FM-43: the |c|-mass crossing is reported; an ill-conditioned signed-share crossing stays missing
    assert math.isclose(row.gt_mass_10pct_mm, 4.0) and math.isnan(row.gt_signedShare_10pct_mm)
    shown = report.comparisonLines(table)[0]
    assert "cp_mm_noisy_median" in shown and "gt_mass_10pct_mm" in shown and "50pct" not in shown and "signedShare" not in shown


def test_relative_noise_floor_in_the_noise_doubling_table(tmp_path):
    """FM-44: noiseDoubling.csv carries noise-only PP / healthy bipolar median for every noise level."""
    from efov.io import saveJson
    model = ROOT / "config" / "model.json"
    for name, sd, medianMv in (("baseline", 0.005, 5.0), (sweep.variantName("noiseSdMv", 0.01), 0.01, 5.0)):
        folder = tmp_path / "variants" / name
        (folder / "measurements").mkdir(parents=True)
        saveJson(folder / "measurements" / "amplitudes.json", dict(
            noiseOnlyBipolarPpMv=5 * sd, relativeNoiseFloor=[dict(gapMm=1.0, relativeNoiseFloor=5 * sd / medianMv)], clinicalReferenceRatio=0.01 / 1.44))
    floors = sweep.relativeNoiseFloors(tmp_path, {"baseline": 0.005, sweep.variantName("noiseSdMv", 0.01): 0.01}).set_index("noiseSdMv")
    assert abs(floors.relativeNoiseFloor[0.005] - 0.005) < 1e-12 and abs(floors.relativeNoiseFloor[0.01] / floors.relativeNoiseFloor[0.005] - 2) < 1e-12
    assert 0.04 in sweep.loadJson(model)["sweep"]["noiseSdMv"]["planned"]
