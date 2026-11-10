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
    dataset = export.datasetName(1.0, 0, "ensite")
    pd.DataFrame(dict(dataset=[dataset], Location=["Inferior"], Metric=["Voltage"], Electrode=["Bipolar"], cp_mm_mean=[10.0],
                      cp_mm_lower=[9.0], cp_mm_upper=[11.0], value_at_cp_mean=[0.04])).to_csv(out / "changePoints" / "SuppResults5_change_points_all.csv", index=False)
    pd.DataFrame(dict(dataset=[dataset] * 2, Location=["Inferior"] * 2, Metric=["Voltage"] * 2, Electrode=["Bipolar"] * 2,
                      model=["Exponential"] * 2, term=["beta1", "cp"], mean=[-0.25, 10.0])).to_csv(out / "changePoints" / "Table5_bayesian_posteriors_all.csv", index=False)
    pd.DataFrame(dict(kind=["bi", "bi"], label=["a", "b"], gapMm=[1.0, 1.0], measure=["mass", "mass"], level=[0.1, 0.1],
                      distanceMm=[3.0, 5.0], placements=[9, 9])).to_csv(out / "groundTruth" / "boundaryCrossings.csv", index=False)
    table = report.efovComparison(SimpleNamespace(out=out))
    row = table.iloc[0]
    assert math.isclose(row.logDropAtCp, 2.5) and math.isclose(row.gt_mass_10pct_mm, 4.0) and (out / "efovComparison.csv").exists()
