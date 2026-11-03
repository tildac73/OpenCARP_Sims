"""Real openCARP, start to finish, on config/test_small.json (opt-in: EFOV_RUN_OPENCARP=1; ~20 min on 2 cores).

Checks that every stage runs and that the implementation checks hold on real output. The small slab
is not expected to pass the profile prerequisites (too narrow for a 10 mm far region).
"""
import os
import shutil

import pytest

from efov.config import loadConfig
from efov.pipeline import Run
from helpers import ROOT

pytestmark = pytest.mark.skipif(os.environ.get("EFOV_RUN_OPENCARP") != "1" or not shutil.which("openCARP"),
                                reason="set EFOV_RUN_OPENCARP=1 with openCARP on PATH")


def test_small_model_end_to_end(tmp_path):
    config = loadConfig(ROOT / "config" / "test_small.json")
    run = Run(config, tmp_path / "run", processCount=int(os.environ.get("EFOV_NP", "2")))
    run.all()
    accepted = (run.out / "acceptance.json").read_text()
    assert '"implementation.forwardModel": true' in accepted and '"implementation.groundTruth": true' in accepted
    physiology = run.result("physiology")
    for condition in ("intact", "lesion"):
        assert physiology["conditions"][condition]["criteria"]["oneActivationPerBeat"]
        assert physiology["conditions"][condition]["criteria"]["quietLesionCore"]
    assert all(status == "complete" for status in run.status().values())
    # A rerun reuses every stage; a changed analysis setting is refused for the stages that use it.
    Run(config, run.out).all()
    changed = loadConfig(ROOT / "config" / "test_small.json")
    changed["changePoints"]["stepMm"] = 0.1
    from efov.pipeline import StaleOutputError
    with pytest.raises(StaleOutputError):
        Run(changed, run.out).runStage("changePoints")
    Run(changed, run.out).runStage("measurements")              # upstream of the change: still reused
