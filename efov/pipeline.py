"""The final model from start to finish, as named stages over one output folder.

    preflight     software versions, repository commit, solver flavour check
    conditioning  bench single-cell limit cycle
    simulation    openCARP intact and lesion runs (carputils, pseudo-bidomain, bath, prepacing)
    physiology    activation, stability and conduction-velocity checks
    window        measurement window from intact activation (before any electrogram is read)
    electrograms  unipolar phi_e at every electrode; forward-model check (openCARP = sigma_m solve)
    measurements  gain, noise, recordings, peak-to-peak, filter-history and profile checks
    changePoints  clinical-style decay-to-plateau fits
    groundTruth   exact contributing-tissue fields and their checks
    report        acceptance.json, report.md, figures

Each stage writes <output>/stages/<stage>.json with its status, result and a fingerprint of the
configuration it used (and of its parents). A rerun skips stages whose fingerprint matches and
refuses to reuse outputs made with different settings. Checks that indicate a broken model setup
(forward-model mismatch, ground-truth reconstruction) stop the run; scientific checks (physiology,
profile prerequisites) are recorded and decide whether results are eligible for interpretation.

Use from Python (notebooks):  run = Run(config, "runs/v3_reference", processCount=24); run.all()
or the command line:          python -m efov run --out runs/v3_reference --np 24
"""
from datetime import datetime, timezone
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import traceback

from . import changepoints, electrograms, groundtruth, measurements, physiology, report, simulation
from .config import REPOSITORY, fingerprint, unconfirmed
from .forward import BathModel
from .io import loadJson, readVertices, saveJson

STAGES = ["preflight", "conditioning", "simulation", "physiology", "window", "electrograms",
          "measurements", "changePoints", "groundTruth", "report"]
PARENTS = {"preflight": [], "conditioning": [], "simulation": ["conditioning"], "physiology": ["simulation"],
           "window": ["simulation"], "electrograms": ["window"], "measurements": ["electrograms"],
           "changePoints": ["measurements", "physiology"], "groundTruth": ["measurements"],
           "report": ["physiology", "electrograms", "measurements", "changePoints", "groundTruth"]}


class StaleOutputError(RuntimeError):
    pass


def now():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


class Run:
    def __init__(self, config, outputFolder, processCount=1, durationMs=None, log=print):
        self.config, self.out, self.np = config, Path(outputFolder).resolve(), int(processCount)
        self.durationMs = durationMs                                  # short verification runs only
        self.log = log
        self._tissues, self._models = {}, {}
        self.out.mkdir(parents=True, exist_ok=True)
        modelPath = self.out / "model.json"
        if not modelPath.exists():
            saveJson(modelPath, config)

    # ------------------------------------------------------------- bookkeeping
    def folder(self, name):
        path = self.out / name
        path.mkdir(parents=True, exist_ok=True)
        return path

    def record(self, name):
        path = self.out / "stages" / f"{name}.json"
        return loadJson(path) if path.exists() else {}

    def result(self, name):
        record = self.record(name)
        if record.get("status") != "complete":
            raise RuntimeError(f"Stage '{name}' has not completed; run it first")
        return record["result"]

    def fingerprint(self, name):
        if name == "preflight":
            return "environment"
        return fingerprint(self.config, name, [self.fingerprint(p) for p in PARENTS[name]])

    def runStage(self, name):
        expected, previous = self.fingerprint(name), self.record(name)
        if previous.get("status") == "complete":
            if previous.get("fingerprint") != expected:
                raise StaleOutputError(
                    f"Stage '{name}' in {self.out} was produced with different settings. Use a new output folder "
                    f"(or delete {self.out}/stages/{name}.json and that stage's outputs if you are sure).")
            self.log(f"[{name}] complete, reusing")
            return previous["result"]
        self.log(f"[{name}] started {now()}")
        started = now()
        try:
            result = getattr(self, "stage_" + name)()
        except Exception as error:
            saveJson(self.out / "stages" / f"{name}.json", dict(status="failed", started=started, finished=now(),
                     fingerprint=expected, error=f"{type(error).__name__}: {error}", traceback=traceback.format_exc()))
            self.log(f"[{name}] FAILED: {error}")
            raise
        saveJson(self.out / "stages" / f"{name}.json", dict(status="complete", started=started, finished=now(),
                                                            fingerprint=expected, result=result))
        self.log(f"[{name}] complete {now()}")
        return result

    def all(self, until=None):
        for name in STAGES:
            self.runStage(name)
            if name == until:
                break
        return self.record("report").get("result")

    def status(self):
        return {name: self.record(name).get("status", "not started") for name in STAGES}

    # ------------------------------------------------------------- shared objects
    @property
    def jobs(self):
        return {c: self.out / "simulations" / c for c in simulation.CONDITIONS}

    def tissue(self, condition):
        if condition not in self._tissues:
            self._tissues[condition] = simulation.tissueModel(self.jobs[condition])
        return self._tissues[condition]

    def bathModel(self, condition):
        """Independent sigma_m extracellular model of a run (same mesh, conductivities, ground)."""
        if condition not in self._models:
            t, lesion = self.config["tissue"], self.config["lesion"]
            self._models[condition] = BathModel(self.tissue(condition), t["conductivitiesSPerM"], lesion["conductivitySPerM"],
                                                self.config["bath"]["bloodSPerM"],
                                                readVertices(self.jobs[condition] / "ground.vtx"))
        return self._models[condition]

    def window(self):
        return self.result("window")["windowMs"]

    # ------------------------------------------------------------- stages
    def stage_preflight(self):
        record = dict(timeUtc=now(), python=sys.version.split()[0], platform=platform.platform(), cpuCount=os.cpu_count(),
                      processCount=self.np, mpiexecBin=os.environ.get("MPIEXEC_BIN"),
                      unconfirmedValues=unconfirmed(self.config))
        for name in ["openCARP", "mesher", "bench"]:
            record[name] = simulation.findExecutable(name, f"{name.upper()}_BIN")
        if self.np > 1 and not os.environ.get("MPIEXEC_BIN"):
            raise RuntimeError("Set MPIEXEC_BIN to openCARP's own MPI launcher before running with --np > 1")
        import numpy, pandas, scipy
        record["packages"] = dict(numpy=numpy.__version__, scipy=scipy.__version__, pandas=pandas.__version__)
        try:
            import pyamg
            record["packages"]["pyamg"] = pyamg.__version__
        except ImportError:
            record["packages"]["pyamg"] = "not installed: large extracellular solves fall back to ILU-GMRES (pip install pyamg)"
        build = subprocess.run([record["openCARP"], "-buildinfo"], capture_output=True, text=True)
        record["openCarpBuild"] = (build.stdout + build.stderr)[-1500:]
        git = lambda *a: subprocess.run(["git", "-C", str(REPOSITORY), *a], capture_output=True, text=True).stdout.strip()
        record["repositoryCommit"], record["uncommittedChanges"] = git("rev-parse", "HEAD"), bool(git("status", "--porcelain"))
        record["diskFreeGb"] = round(shutil.disk_usage(self.out).free / 1e9, 1)
        preferred, fallback = self.config["numerics"]["solverFlavor"], self.config["numerics"]["fallbackSolverFlavor"]
        record["solverFlavor"] = preferred if simulation.solverFlavorWorks(self.config, self.out / "preflight", preferred, self.np) else fallback
        return record

    def stage_conditioning(self):
        return dict(state=str(simulation.conditionCell(self.config, self.folder("conditioning"))))

    def stage_simulation(self):
        state = self.result("conditioning")["state"]
        flavor = self.record("preflight").get("result", {}).get("solverFlavor")
        timing = {c: simulation.runCondition(self.config, c, state, self.out, self.np, flavor, self.durationMs)
                  for c in simulation.CONDITIONS}
        return dict(timing=timing, nodes={c: simulation.runRecord(j)["nodeSets"] for c, j in self.jobs.items()})

    def stage_physiology(self):
        tissues = {c: self.tissue(c) for c in simulation.CONDITIONS}
        return physiology.runPhysiology(self.config, self.jobs, tissues, self.folder("physiology"))

    def stage_window(self):
        window = physiology.deriveWindow(self.config, self.jobs["intact"], self.tissue("intact"))
        saveJson(self.folder("electrograms") / "window.json", window)
        return window

    def stage_electrograms(self):
        conditions = simulation.CONDITIONS
        checks = electrograms.runElectrograms(self.config, self.jobs, {c: self.tissue(c) for c in conditions},
                                              {c: self.bathModel(c) for c in conditions}, self.window(), self.folder("electrograms"))
        if not all(c["passed"] for c in checks.values()):
            raise RuntimeError(f"openCARP phi_e does not match the independent sigma_m solve: {checks}")
        return dict(forwardModelCheck=checks)

    def stage_measurements(self):
        points, calibration, status = measurements.runMeasurements(self.config, self.out / "electrograms", self.window(),
                                                                   self.folder("measurements"))
        return dict(calibration=calibration, checks=status, recordings=int(len(points)))

    def prerequisites(self):
        checks = self.result("measurements")["checks"]
        return dict(physiology=bool(self.result("physiology")["passed"]),
                    forwardModel=all(c["passed"] for c in self.result("electrograms")["forwardModelCheck"].values()),
                    **checks)

    def stage_changePoints(self):
        import pandas as pd
        points = pd.concat([pd.read_csv(self.out / "measurements" / f"{c}Points.csv") for c in simulation.CONDITIONS])
        prerequisites = self.prerequisites()
        fits = changepoints.runChangePoints(self.config, points, self.folder("changePoints"), all(prerequisites.values()))
        primary = changepoints.primaryFits(fits)
        return dict(prerequisites=prerequisites, profiles=int(len(fits)),
                    primary=primary.reindex(columns=["condition", "gapMm", "kind", "label", "model", "changePointMm",
                                                     "fitStatus", "eligibleForInterpretation"]).to_dict("records"))

    def stage_groundTruth(self):
        checks = groundtruth.runGroundTruth(self.config, self.jobs, {c: self.bathModel(c) for c in simulation.CONDITIONS},
                                            self.out / "measurements", self.result("measurements")["calibration"]["gain"],
                                            self.window(), self.folder("groundTruth"))
        if not checks["passed"]:
            raise RuntimeError(f"Ground-truth fields do not reproduce the electrograms: {checks}")
        return checks

    def stage_report(self):
        return report.writeReport(self)
