"""Real openCARP, start to finish, on config/test_small.json (opt-in: EFOV_RUN_OPENCARP=1).

Checks that every stage runs on real output (monodomain runs, openCARP lead-field module (>= 20) per
electrode, FM-23) and that the implementation checks hold: CV tuning, shares summing to 1, the lesion
strip and the isolated tissue staying quiet. The small slab is not expected to
pass the profile prerequisites (too narrow for a stable far region). The changePoints stage needs R
with Wang's packages; it is skipped (with the reason recorded) if Rscript is missing.
"""
import os
import shutil

import pytest

from efov.config import loadConfig
from efov.pipeline import Run, StaleOutputError
from helpers import SMALL

pytestmark = pytest.mark.skipif(os.environ.get("EFOV_RUN_OPENCARP") != "1" or not shutil.which("openCARP"),
                                reason="set EFOV_RUN_OPENCARP=1 with openCARP on PATH")


def test_small_model_end_to_end(tmp_path):
    config = loadConfig(SMALL)
    run = Run(config, tmp_path / "run", processCount=int(os.environ.get("EFOV_NP", "2")))
    until = "report" if shutil.which("Rscript") else "export"
    if until == "export":
        run.all(until="export")
        run.runStage("groundTruth")
    else:
        run.all(until="report")
    assert run.result("tune")["cellSurfVolRatioPerUm"] > 0
    assert run.result("groundTruth")["passed"]
    physiology = run.result("physiology")
    for condition in ("intact", "lesion"):
        assert physiology["conditions"][condition]["criteria"]["oneActivationPerBeat"]
        assert physiology["conditions"][condition]["criteria"]["quietLesionCore"]
    assert (run.jobs["lesion"] / "run.par").exists()                            # the resolved openCARP file of the run
    # A rerun reuses every stage; a changed analysis setting is refused for the stages that use it.
    Run(config, run.out).runStage("measurements")
    changed = loadConfig(SMALL, {"noiseSdMv": 0.01})
    with pytest.raises(StaleOutputError):
        Run(changed, run.out).runStage("measurements")
    Run(changed, run.out).runStage("electrograms")                              # upstream of the change: still reused
