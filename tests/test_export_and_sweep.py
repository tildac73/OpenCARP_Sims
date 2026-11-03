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
