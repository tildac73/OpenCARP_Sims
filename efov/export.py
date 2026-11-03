"""Exports in Wang's EnSite column format, for her analysis code (FM-12).

One analysis dataset = one gap and one noise realisation. Its folder holds the two files her Rmd
reads by name:
  merged_rectangle_summary_euclidean.xlsx       lesion run   (her post-ablation map)
  merged_rectangle_summary_euclidean_pre.xlsx   intact run   (her pre-ablation map)
Columns her read_map uses: patientID (1: one simulated slab), inf0_roof1 (region), uni1_bi2_omni3,
DistAcrossPath (distance from the lesion-centre line = distance inside the lesion - boundaryOffset),
PP (mV), PFreq (Hz; empty for unipoles, 0 when undetectable). Extra columns identify each row.
combineExports pools the inferior and roof runs into one dataset, as her Location factor expects.
"""
from pathlib import Path

import numpy as np
import pandas as pd

from .io import saveJson

KIND_CODE = {"uni": 1, "bi": 2, "omni": 3}
FILES = {"lesion": "merged_rectangle_summary_euclidean.xlsx", "intact": "merged_rectangle_summary_euclidean_pre.xlsx"}


def wangRows(points, region, boundaryOffsetMm):
    """Points table of one condition -> Wang's columns (plus identifiers)."""
    pf = np.where(points.kind == "uni", np.nan, points.pfHz.fillna(0.0))
    return pd.DataFrame(dict(
        patientID=1, inf0_roof1=0 if region == "inferior" else 1, uni1_bi2_omni3=points.kind.map(KIND_CODE).to_numpy(),
        DistAcrossPath=points.distanceInsideMm.to_numpy() - boundaryOffsetMm, PP=points.ppMv.to_numpy(), PFreq=pf,
        label=points.label.to_numpy(), placement=points.placement.to_numpy(), gapMm=points.gapMm.to_numpy(),
        noiseRepeat=points.noiseRepeat.to_numpy(), ppWindow=points.ppWindow.to_numpy(), pfGlobalHz=points.pfGlobalHz.to_numpy()))


def datasetName(gapMm, noiseRepeat):
    return f"gap{gapMm:g}mm_noise{noiseRepeat}"


def runExport(config, measurementsFolder, folder):
    """Write one Wang-format dataset per (gap, noise realisation)."""
    folder, measurementsFolder = Path(folder), Path(measurementsFolder)
    region, offset = config["values"]["region"], config["fixed"]["export"]["boundaryOffsetMm"]
    tables = {c: pd.read_csv(measurementsFolder / f"{c}Points.csv") for c in FILES}
    written = []
    for gap in sorted(tables["lesion"].gapMm.unique()):
        for repeat in sorted(tables["lesion"].noiseRepeat.unique()):
            target = folder / datasetName(gap, repeat)
            target.mkdir(parents=True, exist_ok=True)
            for condition, name in FILES.items():
                t = tables[condition]
                wangRows(t[(t.gapMm == gap) & (t.noiseRepeat == repeat)], region, offset).to_excel(target / name, index=False)
            written.append(str(target))
    saveJson(folder / "datasets.json", dict(datasets=written, region=region, boundaryOffsetMm=offset,
                                            note="patientID = 1: one simulated slab; points are not independent replicates"))
    return written


def combineExports(runFolders, target):
    """Pool the same datasets of several runs (e.g. inferior + roof) into target/<dataset>/."""
    target = Path(target)
    names = sorted({p.name for run in runFolders for p in (Path(run) / "export").iterdir() if p.is_dir()})
    for name in names:
        (target / name).mkdir(parents=True, exist_ok=True)
        for file in FILES.values():
            parts = [pd.read_excel(Path(run) / "export" / name / file) for run in runFolders if (Path(run) / "export" / name / file).exists()]
            pd.concat(parts, ignore_index=True).to_excel(target / name / file, index=False)
    saveJson(target / "datasets.json", dict(datasets=[str(target / n) for n in names], runs=[str(r) for r in runFolders]))
    return [target / n for n in names]
