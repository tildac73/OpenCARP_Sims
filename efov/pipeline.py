"""The final model from start to finish, as named stages.

    preflight     software versions, repository commit, solver flavour check, R availability
    conditioning  bench single-cell limit cycle
    tune          beta for the CV target at the production mesh (FM-18); the CV gate
    simulation    openCARP intact and lesion runs (monodomain, FM-23; lesion strip, FM-24)
    physiology    activation, planarity, direction and stability checks
    window        measurement window of every placement from intact activation (FM-28), before any electrogram
    leadFields    openCARP Laplace lead field of every distinct electrode (cached per conductor, FM-23)
    electrograms  unipolar phi at every electrode by reciprocity (sigma_i source)
    measurements  acquisition chain, noise, recordings, PP, peak frequency (two definitions), profile checks;
                  physical units, no calibration (FM-25)
    export        Wang-format datasets (one per gap, noise realisation and PF definition)
    changePoints  Wang's Rmd on each dataset (R/wang_replication.R)
    groundTruth   contribution fields from the same lead fields: classes, profiles, shape
    report        acceptance.json, report.md, figures
    compact       optional: keep Vm only where the analysis can read it (FM-20)

Each stage writes a record <folder>/stages/<stage>.json with its status, result and a fingerprint of
the configuration it used (and of its parents). A rerun skips a stage whose fingerprint matches and
refuses outputs made with different settings. conditioning, tune, simulation and physiology depend
only on the simulation settings; with a simulation cache (sweeps, FM-17) their records and outputs
live in <cache>/<simulation fingerprint>/ and are shared by every variant that does not change them.
Lead fields live in <lead-field cache>/<conductor fingerprint>/<condition>/ and are shared by every
variant with the same mesh, fibres and conductivities.
Checks of a broken setup (share sum, CV tuning) stop the run; scientific checks
(physiology, profile prerequisites) are recorded and decide eligibility for interpretation.

Python (notebooks):   run = Run(config, "runs/v4_reference", processCount=24); run.all()
Command line:         python -m efov run --out runs/v4_reference --np 24
"""
from datetime import datetime, timezone
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import traceback

import pandas as pd

from . import electrograms, export, groundtruth, measurements, physiology, report, simulation, tune
from .config import REPOSITORY, fingerprint, par, tunedBetaFor, withParOverrides
from .io import loadJson, saveJson

STAGES = ["preflight", "conditioning", "tune", "simulation", "physiology", "window", "leadFields", "electrograms", "measurements",
          "export", "changePoints", "groundTruth", "report", "compact"]
PARENTS = {"preflight": [], "conditioning": [], "tune": ["conditioning"], "simulation": ["tune"], "physiology": ["simulation"],
           "window": ["simulation"], "leadFields": ["simulation"], "electrograms": ["window", "leadFields"],
           "measurements": ["electrograms"], "export": ["measurements"],
           "changePoints": ["export", "physiology"], "groundTruth": ["measurements", "leadFields"],
           "report": ["physiology", "measurements", "changePoints", "groundTruth"], "compact": ["report"]}
SHARED = ("conditioning", "tune", "simulation", "physiology")


class StaleOutputError(RuntimeError):
    pass


def sharedFolder(config, outputFolder, simulationCache=None):
    """Where the simulation-only stages live: the run folder, or <cache>/<simulation fingerprint> for sweeps."""
    if not simulationCache:
        return Path(outputFolder).resolve()
    chain = fingerprint(config, "simulation", [fingerprint(config, "tune", [fingerprint(config, "conditioning")])])
    return Path(simulationCache).resolve() / chain


def now():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


class Run:
    def __init__(self, config, outputFolder, processCount=1, simulationCache=None, durationMs=None, log=print, leadFieldCache=None):
        self.config, self.out, self.np = config, Path(outputFolder).resolve(), int(processCount)
        self.durationMs, self.log = durationMs, log
        self.leadFieldCache = Path(leadFieldCache or (Path(simulationCache).parent / "leadFieldCache" if simulationCache
                                                       else self.out / "leadFieldCache")).resolve()
        self._tissues = {}
        self.out.mkdir(parents=True, exist_ok=True)
        self.shared = sharedFolder(config, self.out, simulationCache)
        self.shared.mkdir(parents=True, exist_ok=True)
        if not (self.out / "model.json").exists():
            saveJson(self.out / "model.json", dict(values=config["values"], fixed=config["fixed"], parOverrides=config["parOverrides"],
                                                   modelPar=config["modelPar"], sharedSimulationFolder=str(self.shared)))
        if (self.shared / "stages" / "tune.json").exists() and self.record("tune").get("status") == "complete":
            self._applyBeta(self.record("tune")["result"]["cellSurfVolRatioPerUm"])

    # ------------------------------------------------------------- bookkeeping
    def folder(self, name, shared=False):
        path = (self.shared if shared else self.out) / name
        path.mkdir(parents=True, exist_ok=True)
        return path

    def recordPath(self, name):
        return (self.shared if name in SHARED else self.out) / "stages" / f"{name}.json"

    def record(self, name):
        path = self.recordPath(name)
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
                raise StaleOutputError(f"Stage '{name}' in {self.recordPath(name).parent.parent} was produced with different settings. "
                                       "Use a new output folder (or delete that stage's record and outputs if you are sure).")
            self.log(f"[{name}] complete, reusing")
            return previous["result"]
        self.log(f"[{name}] started {now()}")
        started = now()
        try:
            result = getattr(self, "stage_" + name)()
        except Exception as error:
            saveJson(self.recordPath(name), dict(status="failed", started=started, finished=now(), fingerprint=expected,
                                                 error=f"{type(error).__name__}: {error}", traceback=traceback.format_exc()))
            self.log(f"[{name}] FAILED: {error}")
            raise
        saveJson(self.recordPath(name), dict(status="complete", started=started, finished=now(), fingerprint=expected, result=result))
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
        return {c: self.shared / "simulations" / c for c in simulation.CONDITIONS}

    def tissue(self, condition):
        if condition not in self._tissues:
            self._tissues[condition] = simulation.tissueModel(self.jobs[condition])
        return self._tissues[condition]

    def window(self):
        """Measurement windows of every placement (P, 2), ms after the last stimulus."""
        return self.result("window")["windowsMs"]

    def _applyBeta(self, beta):
        if abs(par(self.config, tune.BETA_KEY) - beta) > 0:
            self.config = withParOverrides(self.config, {tune.BETA_KEY: beta})

    # ------------------------------------------------------------- stages
    def stage_preflight(self):
        record = dict(timeUtc=now(), python=sys.version.split()[0], platform=platform.platform(), cpuCount=os.cpu_count(),
                      processCount=self.np, mpiexecBin=os.environ.get("MPIEXEC_BIN"))
        for name in ["openCARP", "mesher", "bench"]:
            record[name] = simulation.findExecutable(name, f"{name.upper()}_BIN")
        record["Rscript"] = shutil.which("Rscript") or "not found: the changePoints stage needs R with Wang's packages"
        if self.np > 1 and not os.environ.get("MPIEXEC_BIN"):
            raise RuntimeError("Set MPIEXEC_BIN to openCARP's own MPI launcher before running with --np > 1")
        import numpy, pandas, scipy
        record["packages"] = dict(numpy=numpy.__version__, scipy=scipy.__version__, pandas=pandas.__version__)
        build = subprocess.run([record["openCARP"], "-buildinfo"], capture_output=True, text=True)
        record["openCarpBuild"] = (build.stdout + build.stderr)[-1500:]
        git = lambda *a: subprocess.run(["git", "-C", str(REPOSITORY), *a], capture_output=True, text=True).stdout.strip()
        record["repositoryCommit"], record["uncommittedChanges"] = git("rev-parse", "HEAD"), bool(git("status", "--porcelain"))
        record["diskFreeGb"] = round(shutil.disk_usage(self.out).free / 1e9, 1)
        preferred, fallback = self.config["fixed"]["numerics"]["solverFlavor"], self.config["fixed"]["numerics"]["fallbackSolverFlavor"]
        record["solverFlavor"] = preferred if simulation.solverFlavorWorks(self.config, self.out / "preflight", preferred, self.np) else fallback
        return record

    def flavor(self):
        return self.record("preflight").get("result", {}).get("solverFlavor")

    def stage_conditioning(self):
        return dict(state=str(simulation.conditionCell(self.config, self.folder("conditioning", shared=True))))

    def stage_tune(self):
        recorded = tunedBetaFor(self.config)
        if recorded is not None:
            self._applyBeta(recorded)
            return dict(cellSurfVolRatioPerUm=recorded, source="config/tuning.json (efov tune --write-par)")
        result = tune.tuneBeta(self.config, self.folder("tune", shared=True), self.result("conditioning")["state"], self.np, self.flavor())
        self._applyBeta(result["cellSurfVolRatioPerUm"])
        return dict(result, source="tuned in this run's simulation folder")

    def stage_simulation(self):
        state = self.result("conditioning")["state"]
        timing = {c: simulation.runCondition(self.config, c, state, self.folder("simulations", shared=True), self.np, self.flavor(),
                                             self.durationMs) for c in simulation.CONDITIONS}
        return dict(timing=timing, nodes={c: simulation.runRecord(j)["nodeSets"] for c, j in self.jobs.items()},
                    cellSurfVolRatioPerUm=par(self.config, tune.BETA_KEY))

    def stage_physiology(self):
        return physiology.runPhysiology(self.config, self.jobs, {c: self.tissue(c) for c in simulation.CONDITIONS},
                                        self.folder("physiology", shared=True))

    def stage_window(self):
        window = physiology.deriveWindow(self.config, self.jobs["intact"], self.tissue("intact"))
        saveJson(self.folder("electrograms") / "window.json", window)
        return window

    def stage_leadFields(self):
        return electrograms.runLeadFields(self.config, self.jobs, {c: self.tissue(c) for c in simulation.CONDITIONS},
                                          self.leadFieldCache, self.folder("leadFields"), self.np, self.log)

    def stage_electrograms(self):
        return electrograms.runElectrograms(self.config, self.jobs, {c: self.tissue(c) for c in simulation.CONDITIONS},
                                            self.window(), self.out / "leadFields", self.folder("electrograms"))

    def stage_measurements(self):
        points, amplitudes, status = measurements.runMeasurements(self.config, self.out / "electrograms", self.window(),
                                                                  self.folder("measurements"))
        return dict(amplitudes=amplitudes, checks=status, recordings=int(len(points)))

    def stage_export(self):
        return dict(datasets=export.runExport(self.config, self.out / "measurements", self.folder("export")))

    def prerequisites(self):
        return dict(physiology=bool(self.result("physiology")["passed"]), **self.result("measurements")["checks"])

    def stage_changePoints(self):
        rscript = shutil.which("Rscript")
        if not rscript:
            raise RuntimeError("Rscript not found: install R with Wang's packages (efov-runner image)")
        folder = self.folder("changePoints")
        tables = {}
        for dataset in self.result("export")["datasets"]:
            name = Path(dataset).name
            target = folder / name
            code = simulation.runLogged([rscript, str(REPOSITORY / "R" / "wang_replication.R"), dataset, str(target)], folder / f"{name}.log")
            if code != 0:
                raise RuntimeError(f"Wang's analysis failed on {name}; see {folder / (name + '.log')}")
            for table in ("Table5_bayesian_posteriors", "SuppResults5_change_points", "ModelSelection_AIC", "Table3_mixed_models_REML",
                          "Table4_decay_distances", "SuppResults7_ROC", "Table2_voltage_PF_by_tissue"):
                path = target / "tables" / f"{table}.csv"
                if path.exists():
                    tables.setdefault(table, []).append(pd.read_csv(path).assign(dataset=name))
        for table, parts in tables.items():
            pd.concat(parts, ignore_index=True).to_csv(folder / f"{table}_all.csv", index=False)
        prerequisites = self.prerequisites()
        return dict(prerequisites=prerequisites, eligibleForInterpretation=all(prerequisites.values()),
                    datasets=len(self.result("export")["datasets"]), tables=sorted(tables))

    def stage_groundTruth(self):
        checks = groundtruth.runGroundTruth(self.config, self.jobs, {c: self.tissue(c) for c in simulation.CONDITIONS},
                                            self.out / "electrograms", self.out / "measurements", self.window(),
                                            self.folder("groundTruth"))
        if not checks["passed"]:
            raise RuntimeError(f"Ground-truth shares do not sum to 1: {checks}")
        return checks

    def stage_report(self):
        return report.writeReport(self)

    def stage_compact(self):
        """Delete raw output of a simulation no other variant reuses (the sweep runner decides; FM-20)."""
        if not self.config["fixed"]["compactAfterReport"]:
            return dict(compacted=False, reason="fixed.compactAfterReport is false")
        start, end = electrograms.analysisTimes(self.config, self.window())
        done = {}
        for condition, job in self.jobs.items():
            if (job / "vm.igb").exists():
                simulation.compactRun(job, start, end)
            done[condition] = loadJson(job / "compact.json")
        return dict(compacted=True, conditions=done)
