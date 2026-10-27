#!/usr/bin/env python3
"""Reprocess boundaryPilotV2 with a declared EGM measurement window.

Refinement of 27 Oct 2026 (see the Part 1 findings log). No simulation is rerun.
The original pipeline functions are imported unchanged from 3D_Code and
Shared_Pipeline; this driver only

  * sets analysis.beatWindowMs (used for PP, bipole/omni selection and
    contribution reconstruction; physiology checks are NOT rerun),
  * optionally sets analysis.noiseSdMv (noise-level sweep),
  * holds the gain at the original calibration value,
  * splits the work into short stages so each fits the analysis VM's time limit.

Stages (run in order, each writes into <out>/results):
  measure            processCondition for both conditions
  contrib-intact     computeContributions, intact
  contrib-lesion     computeContributions, lesion
  finalise           saveExperimentResults (+ clinical change points)
  changepoints-only  analyseChangePoints without contributions (noise sweep)
"""
import argparse
import json
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent  # Masters_V2
sys.path.insert(0, str(HERE / "shim"))  # numpy stand-in, only if scipy is missing
try:
    import scipy  # noqa: F401
    if getattr(scipy, "__file__", "").startswith(str(HERE / "shim")):
        print("Using numpy scipy shim (validated in test_shim.py)", flush=True)
except ImportError:
    pass
sys.path.insert(1, str(ROOT / "3D_Code"))
sys.path.insert(2, str(ROOT / "Shared_Pipeline"))

import pandas as pd  # noqa: E402
import boundary_analysis as analysis  # noqa: E402
from boundary_clinical import analyseChangePoints  # noqa: E402

SOURCE_RUN = ROOT / "boundaryPilotV2 2"


def prepareFolder(out, window, noiseSd):
    out.mkdir(parents=True, exist_ok=True)
    (out / "results").mkdir(exist_ok=True)
    for name in ["mesh", "intact", "lesion"]:
        link = out / name
        if not link.exists():
            os.symlink(os.path.relpath(SOURCE_RUN / name, out), link)
    config = json.loads((SOURCE_RUN / "config.json").read_text())
    config["analysis"]["beatWindowMs"] = list(window)
    if noiseSd is not None:
        config["analysis"]["noiseSdMv"] = noiseSd
    original = json.loads((SOURCE_RUN / "results/calibration.json").read_text())
    (out / "refinement.json").write_text(json.dumps(dict(
        sourceRun=str(SOURCE_RUN.name),
        egmWindowMsAfterBeat16Stimulus=list(window),
        noiseSdMvPerRawSample=config["analysis"]["noiseSdMv"],
        gainHeldAtOriginal=original["gain"],
        physiologyRerun=False,
        physiologyPassedOriginal=json.loads((SOURCE_RUN / "results/pilotStatus.json").read_text())["physiologyPassed"],
    ), indent=2))
    (out / "config.json").write_text(json.dumps(config, indent=2))
    return config, original["gain"]


def fixedGain(gain, out):
    def calibrate(runFolder, config):
        (Path(runFolder) / "results/calibration.json").write_text(json.dumps(dict(
            gain=gain, heldFromOriginalRun=True,
            interpretation="Original imposed calibration retained; not recalculated in the new window",
        ), indent=2))
        return gain
    return calibrate


def readMeasurements(out, condition):
    return pd.read_csv(out / "results" / f"{condition}Points.csv")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=["measure", "contrib-intact", "contrib-lesion",
                                          "finalise", "changepoints-only"])
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--window", type=float, nargs=2, default=[25.0, 100.0])
    parser.add_argument("--noise-sd", type=float, default=None)
    parser.add_argument("--conditions", nargs="+", default=["intact", "lesion"])
    arguments = parser.parse_args()
    out = arguments.out.resolve()
    config, gain = prepareFolder(out, arguments.window, arguments.noise_sd)
    physiologyPassed = json.loads((out / "refinement.json").read_text())["physiologyPassedOriginal"]
    analysis.calibrateGain = fixedGain(gain, out)

    if arguments.stage == "measure":
        analysis.calibrateGain(out, config)
        for condition in arguments.conditions:
            analysis.processCondition(out, condition, config, gain)
            print("measured", condition, flush=True)
    elif arguments.stage.startswith("contrib-"):
        condition = arguments.stage.split("-", 1)[1]
        table = analysis.computeContributions(
            out, condition, config, readMeasurements(out, condition), gain, physiologyPassed)
        table.to_pickle(out / "results" / f"_contributions_{condition}.pkl")
        print(table[["recordingId", "rawFov80ReconstructionMm", "signedCorrelation", "shapeError"]].to_string())
    elif arguments.stage == "finalise":
        measurements = [readMeasurements(out, c) for c in ["intact", "lesion"]]
        contributions = [pd.read_pickle(out / "results" / f"_contributions_{c}.pkl")
                         for c in ["intact", "lesion"]]
        _, passed = analysis.saveExperimentResults(
            out, config, physiologyPassed, measurements, contributions)
        print("pilot acceptance passed:", passed)
    elif arguments.stage == "changepoints-only":
        measurements = pd.concat([readMeasurements(out, c) for c in ["intact", "lesion"]],
                                 ignore_index=True)
        try:
            analyseChangePoints(out, config, measurements, pilotPassed=physiologyPassed)
        except FileNotFoundError as error:
            # The final contribution comparison needs points.csv from a full run.
            # All change-point tables are written before that step.
            print("change-point tables written; contribution comparison skipped:", error.filename)
        else:
            print("change points written")


if __name__ == "__main__":
    main()
