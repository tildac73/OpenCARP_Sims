#!/usr/bin/env python3
"""Readiness kit: run Steps 2, 3 and the Step 6 feasibility test in one go, with gates.

    python3 3D_Code/readiness/kit.py plan  --np 16            # no simulator needed: jobs, sizes, commands
    python3 3D_Code/readiness/kit.py run   --np 16            # everything, in order (resumable)
    python3 3D_Code/readiness/kit.py status                   # stage states and openCARP ETAs
    python3 3D_Code/readiness/kit.py report                   # collect results for copying back

Stages, in order. A blocking stage that fails stops the kit; re-running resumes after the last
completed stage and reuses every completed simulation.

  preflight  versions, executables, MPI launcher, memory, disk            blocking
  tests      unit tests of this repository                                 blocking
  mpi        serial vs parallel equivalence (Step 2 gate)                  blocking (skipped if --np 1)
  smoke      5 ms openCARP run of every kind of parameter file used below  blocking
  niederer   N-version benchmark (Niederer et al. 2011)                    reported
  cv         CV tuning to 0.90 m/s and CV resolution study                 blocking (tuning must converge)
  fibres     openCARP rotated / bilayer fibres vs electrode synthesis      reported
  ladder     3D convergence ladder on the tuned model (incl. 1.5 mm wall)  reported, per run
  bath       pseudo-bidomain blood bath vs independent FEM solve           reported
  bathAnalysis  bath electrograms + ground truth from ladder Vm: bath size,
             bath mesh, scar type, bath vs unbounded                        reported
  report     copy summaries and logs (credentials removed) to <out>/report

All settings: 3D_Code/configs/readiness_kit.json. Outputs: <out> (default 3D_Code/runs/readiness).
"""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import traceback

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import common  # noqa: E402
import boundary_setup  # noqa: E402
from boundary_geometry import makeElectrodeLayout  # noqa: E402

STAGES = ["preflight", "tests", "mpi", "smoke", "niederer", "cv", "fibres", "ladder", "bath", "bathAnalysis", "report"]
BLOCKING = {"preflight", "tests", "mpi", "smoke", "cv"}


class Tee:
    def __init__(self, *streams):
        self.streams = streams

    def write(self, text):
        for stream in self.streams:
            stream.write(common.redact(text)); stream.flush()

    def flush(self):
        for stream in self.streams:
            stream.flush()


# ----------------------------------------------------------------- plan
def nodeCount(sizeMm, spacingMm):
    count = 1
    for length in sizeMm:
        count *= round(length / spacingMm) + 1
    return count


def planJobs(settings, tiers):
    jobs = []
    n = settings["niederer"]
    for dx in n["spacingsMm"]:
        for dt in n["timeStepsUs"]:
            nodes = nodeCount(n["slabSizeMm"], dx)
            jobs.append(dict(stage="niederer", name=f"dx{dx:g}_dt{dt:g}", nodes=nodes, steps=n["durationMs"] * 1000 / dt,
                             diskGb=nodes * (n["durationMs"] / n["outputStepMs"] + 1) * 4 / 1e9))
    cv = settings["cvTuning"]
    beatsMs = (cv["beatCount"] - 1) * cv["cycleLengthMs"] + cv["lastBeatDurationMs"]
    strips = [("tune 0.90 m/s, per iteration", cv["referenceSpacingMm"]),
              *[(f"tune {t:g} m/s, per iteration", cv["referenceSpacingMm"]) for t in cv["sensitivityTargetsMPerS"]],
              *[("evaluate tuned", dx) for dx in cv["evaluationSpacingsMm"]],
              ("transverse", cv["referenceSpacingMm"]),
              *[("pilot conductivities", dx) for dx in cv["pilotConductivitySpacingsMm"]]]
    for purpose, dx in strips:
        nodes = nodeCount([cv["stripLengthMm"], cv["stripCrossSectionMm"], cv["stripCrossSectionMm"]], dx)
        jobs.append(dict(stage="cv", name=f"strip dx{dx:g}: {purpose}", nodes=nodes,
                         steps=beatsMs * 1000 / cv["timeStepUs"], diskGb=nodes * beatsMs / cv["outputStepMs"] * 4 / 1e9))
    ladder = settings["ladder"]
    base = json.loads((common.REPOSITORY / ladder["baseConfig"]).read_text())
    provisional = dict(conductivitiesSPerM={k: v * cv["initialScale"] for k, v in cv["baseConductivitiesSPerM"].items()},
                       stimulusStrength=cv["stimulus"]["strength"])
    import ladder as ladderModule
    for run in ladder["runs"]:
        if run["tier"] not in tiers:
            continue
        config = ladderModule.makeConfig(base, run, ladder, provisional)
        nodes = nodeCount(config["slabSizeMm"], config["meshSpacingMm"])
        recovery = len(makeElectrodeLayout(config)["recoveryPoints"])
        durationMs = config["pacing"]["beatCount"] * config["pacing"]["cycleLengthMs"]
        frames = durationMs / config["sampleStepMs"] + 1
        jobs.append(dict(stage="ladder", name=run["name"], nodes=nodes, recoveryPoints=recovery,
                         steps=2 * durationMs * 1000 / config["timeStepUs"],
                         diskGb=2 * (nodes + recovery) * frames * 4 / 1e9 * (1.5 if run.get("decimateTo") else 1.0)))
    b = settings["bath"]
    nodes = nodeCount([b["tissueSizeMm"][0], b["tissueSizeMm"][1], b["tissueSizeMm"][2] + b["bathThicknessMm"]], b["spacingMm"])
    jobs.append(dict(stage="bath", name="bath", nodes=nodes, steps=b["durationMs"] * 1000 / b["timeStepUs"],
                     diskGb=2 * nodes * (b["durationMs"] / b["outputStepMs"] + 1) * 4 / 1e9))
    mpiConfig = json.loads((common.REPOSITORY / settings["mpi"]["config"]).read_text())
    fibreNodes = nodeCount(mpiConfig["slabSizeMm"], mpiConfig["meshSpacingMm"])
    fibreSteps = 2 * mpiConfig["pacing"]["beatCount"] * mpiConfig["pacing"]["cycleLengthMs"] * 1000 / mpiConfig["timeStepUs"]
    for case in ("uniform30", "bilayer0_90"):
        jobs.append(dict(stage="fibres", name=case, nodes=fibreNodes, steps=fibreSteps,
                         diskGb=2 * fibreNodes * (fibreSteps / 2 * mpiConfig["timeStepUs"] / 1000 / mpiConfig["sampleStepMs"]) * 4 / 1e9))
    runsByName = {run["name"]: run for run in ladder["runs"]}
    for model in settings.get("bathAnalysis", {}).get("models", []):
        source = runsByName.get(model["sourceRun"].split("/")[0])
        if source is None or source["tier"] not in tiers:
            continue
        size = list(ladder["reducedSlabSizeMm"]) if source["domain"] == "reduced" else None
        if size is None:
            continue
        size[2] = source.get("wallMm", size[2]) + model["bathThicknessMm"]
        jobs.append(dict(stage="bathAnalysis", name=f"{model['name']} (FEM, no openCARP; ~1300 solves x 2 conditions)",
                         nodes=nodeCount(size, source["spacingMm"]), steps=2600, diskGb=0.5))
    for job in jobs:
        job["nodeSteps"] = job["nodes"] * job["steps"]
    return jobs


def commandPlan(arguments):
    settings = common.loadSettings()
    tiers = ("A", "B") if arguments.tier == "B" else ("A",)
    jobs = planJobs(settings, tiers)
    throughput = readThroughput(arguments.out)
    print(f"{'stage':9} {'job':48} {'nodes':>10} {'node-steps':>11} {'disk GB':>8} {'est. h':>7}")
    for job in jobs:
        hours = job["nodeSteps"] / throughput if throughput else None
        print(f"{job['stage']:9} {job['name'][:48]:48} {job['nodes']:>10,} {job['nodeSteps']:>11.2e} {job['diskGb']:>8.1f} "
              f"{hours if hours is None else round(hours, 1)!s:>7}")
    totalDisk = sum(job["diskGb"] for job in jobs)
    totalSteps = sum(job["nodeSteps"] for job in jobs)
    print(f"\nTotal node-steps {totalSteps:.2e}; disk about {totalDisk:.0f} GB (tier {arguments.tier}).")
    if throughput:
        print(f"Estimated compute about {totalSteps / throughput:.0f} h at the measured {throughput:.2e} node-steps/h "
              f"(--np {throughput_np(arguments.out)}). Ladder runs with many recovery points run slower than this estimate.")
    else:
        print("Hours are unknown until the mpi stage has measured throughput on this machine.")
    print("\nStage order:", " -> ".join(STAGES))
    return jobs


def readThroughput(out):
    path = Path(out) / "mpi" / "throughput.json"
    return json.loads(path.read_text()).get("nodeStepsPerHour") if path.exists() else None


def throughput_np(out):
    path = Path(out) / "mpi" / "throughput.json"
    return json.loads(path.read_text()).get("processCount") if path.exists() else None


# ----------------------------------------------------------------- stages
def stagePreflight(folder, arguments, settings):
    record = dict(timeUtc=common.nowUtc(), python=sys.version, platform=platform.platform(), cpuCount=os.cpu_count(),
                  settingsFile=str(common.KIT_SETTINGS), settings=settings)
    try:
        record["cpuAffinity"] = len(os.sched_getaffinity(0))
    except AttributeError:
        pass
    executables = {}
    for name, variable in [("openCARP", "OPENCARP_BIN"), ("mesher", "MESHER_BIN"), ("bench", "BENCH_BIN")]:
        executables[name] = boundary_setup.findExecutable(name, variable)       # raises if missing
    record["executables"] = executables
    if arguments.np > 1:
        launcher = os.environ.get("MPIEXEC_BIN")
        if not launcher:
            raise RuntimeError("--np > 1 needs MPIEXEC_BIN (the mpirun matching openCARP; see 3D_Code/numerics/README.md)")
        record["mpiLauncher"] = launcher
        record["mpiVersion"] = capture([launcher, "--version"])
        record["openCarpMpiLibrary"] = capture(["sh", "-c", f"ldd {executables['openCARP']} | grep -i mpi"])
    modules = {}
    for module in ["numpy", "scipy", "pandas", "matplotlib"]:
        try:
            modules[module] = __import__(module).__version__
        except ImportError as error:
            raise RuntimeError(f"Python package {module} is required: {error}")
    record["python packages"] = modules
    try:
        import pyamg
        record["pyamg"] = pyamg.__version__
    except ImportError:
        record["pyamg"] = ("not installed: bathAnalysis falls back to ILU-GMRES for large meshes (slower). "
                           "Recommended: pip install pyamg")
    try:
        import carputils
        record["carputilsLocation"] = str(Path(carputils.__file__).parent)
        record["carputilsCommit"] = capture(["git", "-C", str(Path(carputils.__file__).parent), "rev-parse", "HEAD"])
    except ImportError:
        record["carputilsLocation"] = "not importable (not needed by this kit)"
    record["benchModels"] = capture([executables["bench"], "--list-imps"], timeout=120)
    listing = record["benchModels"] or ""
    record["courtemancheAvailable"] = "Courtemanche" in listing
    # If the listing failed (Courtemanche absent too), availability is unknown: let the smoke test decide.
    record["tenTusscherAvailable"] = ("tenTusscherPanfilov" in listing) if record["courtemancheAvailable"] else None
    record["repositoryCommit"] = capture(["git", "-C", str(common.REPOSITORY), "rev-parse", "HEAD"])
    record["repositoryUncommittedChanges"] = capture(["git", "-C", str(common.REPOSITORY), "status", "--porcelain"])
    memory = Path("/proc/meminfo")
    if memory.exists():
        record["memTotal"] = memory.read_text().splitlines()[0]
    usage = shutil.disk_usage(Path(arguments.out).resolve().parent if not Path(arguments.out).exists() else arguments.out)
    tiers = ("A", "B") if arguments.tier == "B" else ("A",)
    needed = sum(job["diskGb"] for job in planJobs(settings, tiers))
    record.update(diskFreeGb=usage.free / 1e9, diskNeededGbEstimate=needed)
    common.saveJson(folder / "preflight.json", record)
    if usage.free / 1e9 < 1.1 * needed and not arguments.skipDiskCheck:
        raise RuntimeError(f"Free disk {usage.free / 1e9:.0f} GB < 1.1 x estimated need {needed:.0f} GB "
                           "(use --skip-disk-check to override, or --tier A)")
    return dict(status="passed", detail="see preflight.json")


def capture(command, timeout=60):
    try:
        result = subprocess.run([str(c) for c in command], capture_output=True, text=True, timeout=timeout)
        return common.redact((result.stdout + result.stderr).strip())[:20000]
    except (OSError, subprocess.TimeoutExpired) as error:
        return f"unavailable: {error}"


def stageTests(folder, arguments, settings):
    code = common.runLogged([sys.executable, "-m", "unittest", "discover", "-s", common.REPOSITORY / "tests"],
                            folder / "tests.log", cwd=common.REPOSITORY, check=False)
    if code != 0:
        raise RuntimeError("Unit tests failed; see tests.log")
    return dict(status="passed")


def stageMpi(folder, arguments, settings):
    if arguments.np <= 1:
        return dict(status="passed", detail="serial run requested; MPI equivalence not needed")
    config = common.REPOSITORY / settings["mpi"]["config"]
    runner = common.REPOSITORY / "3D_Code" / "run_boundary.py"
    folders = {}
    for processes in (1, arguments.np):
        target = folder / f"mpiCheck_np{processes}"
        folders[processes] = target
        if (target / "lesion" / "meta.json").exists():
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        common.runLogged([sys.executable, runner, "--config", config, "--simulate-only", "--np", processes, "--output", target],
                         folder / f"mpiCheck_np{processes}.log")
    code = common.runLogged([sys.executable, common.REPOSITORY / "3D_Code/numerics/compare_runs.py", folders[1], folders[arguments.np]],
                            folder / "compare_runs.log", check=False)
    summary = json.loads((folders[arguments.np] / "mpiEquivalence.json").read_text())
    shutil.copy2(folders[arguments.np] / "mpiEquivalence.json", folder / "mpiEquivalence.json")
    configValues = json.loads(config.read_text())
    nodes = nodeCount(configValues["slabSizeMm"], configValues["meshSpacingMm"])
    steps = configValues["pacing"]["beatCount"] * configValues["pacing"]["cycleLengthMs"] * 1000 / configValues["timeStepUs"]
    seconds = {p: sum(filter(None, [common.lastProgressSeconds(folders[p] / f"{c}.log") for c in ["intact", "lesion"]])) for p in folders}
    throughput = dict(processCount=arguments.np, serialSeconds=seconds[1], parallelSeconds=seconds[arguments.np],
                      speedUp=seconds[1] / seconds[arguments.np] if seconds[arguments.np] else None,
                      nodeStepsPerHour=2 * nodes * steps / (seconds[arguments.np] / 3600) if seconds[arguments.np] else None,
                      openCARP=common.openCarpBanner(folders[1] / "intact.log"))
    common.saveJson(folder / "throughput.json", throughput)
    if code != 0 or not summary.get("passed"):
        raise RuntimeError("MPI run does not reproduce the serial run (mpiEquivalence.json). Keep both folders; "
                           "do not use --np > 1 until resolved.")
    return dict(status="passed", speedUp=throughput["speedUp"])


def provisionalTuned(settings):
    cv = settings["cvTuning"]
    return dict(conductivitiesSPerM={k: v * cv["initialScale"] for k, v in cv["baseConductivitiesSPerM"].items()},
                stimulusStrength=cv["stimulus"]["strength"])


def stageSmoke(folder, arguments, settings):
    import strip_cv, ladder, bath_check, niederer
    preflight = json.loads((Path(arguments.out) / "preflight" / "preflight.json").read_text())
    done = {}
    if preflight.get("tenTusscherAvailable") is not False:
        niederer.smoke(folder / "niederer", settings["niederer"], arguments.np); done["niederer"] = "passed"
    else:
        done["niederer"] = "skipped: tenTusscherPanfilov not listed by bench --list-imps"
    strip_cv.smoke(folder / "cv", settings["cvTuning"], arguments.np); done["cv"] = "passed"
    import fibre_check
    fibre_check.smoke(folder / "fibres", provisionalTuned(settings), arguments.np); done["fibres"] = "passed"
    ladder.smoke(folder / "ladder", settings["ladder"], arguments.np, provisionalTuned(settings)); done["ladder"] = "passed"
    bath_check.runStage(folder, settings["bath"], provisionalTuned(settings), settings, arguments.np, smoke=True); done["bath"] = "passed"
    return dict(status="passed", detail=done)


def stageNiederer(folder, arguments, settings):
    import niederer
    preflight = json.loads((Path(arguments.out) / "preflight" / "preflight.json").read_text())
    if preflight.get("tenTusscherAvailable") is False:
        return dict(status="skipped", detail="tenTusscherPanfilov not available in this build")
    summary = niederer.runStage(folder, settings["niederer"], arguments.np)
    return dict(status="reported", allActivated=summary["allActivated"])


def stageCv(folder, arguments, settings):
    import strip_cv
    summary = strip_cv.runStage(folder, settings["cvTuning"], arguments.np)
    if not summary["converged"]:
        raise RuntimeError("CV tuning did not converge (cvSummary.json); the ladder and bath need tuned conductivities.")
    return dict(status="passed", scale=summary["scale"], strength=summary["strength"],
                transverseSpeedMPerS=summary.get("transverseSpeedMPerS"))


def readTuned(arguments):
    path = Path(arguments.out) / "cv" / "tuned.json"
    if not path.exists():
        raise RuntimeError("cv/tuned.json is missing: the cv stage must pass first")
    return json.loads(path.read_text())


def stageFibres(folder, arguments, settings):
    import fibre_check
    summary = fibre_check.runStage(folder, readTuned(arguments), arguments.np)
    return dict(status="reported", passed=summary["passed"])


def stageBathAnalysis(folder, arguments, settings):
    import bath_analysis
    summary = bath_analysis.runStage(folder, Path(arguments.out) / "ladder", settings["bathAnalysis"], arguments.np)
    return dict(status="reported", failures=summary["failures"],
                comparisons={c["label"]: c.get("passedClarified", c.get("skipped") or c.get("error")) for c in summary["comparisons"]})


def stageLadder(folder, arguments, settings):
    import ladder
    tiers = ("A", "B") if arguments.tier == "B" else ("A",)
    summary = ladder.runStage(folder, settings["ladder"], readTuned(arguments), arguments.np, tiers)
    return dict(status="reported", failures=summary["failures"],
                comparisons={c["label"]: c.get("passedClarified", c.get("skipped") or c.get("error")) for c in summary["comparisons"]})


def stageBath(folder, arguments, settings):
    import bath_check
    summary = bath_check.runStage(folder, settings["bath"], readTuned(arguments), settings, arguments.np)
    return dict(status="reported", passed=summary.get("passed"))


def stageReport(folder, arguments, settings):
    out = Path(arguments.out)
    patterns = ["*/stage.json", "preflight/preflight.json", "mpi/mpiEquivalence.json", "mpi/throughput.json",
                "niederer/benchmark*.json", "niederer/benchmark*.csv", "cv/cvSummary.json", "cv/tuned.json",
                "ladder/ladderSummary.json", "ladder/convergence.csv", "ladder/*.config.json",
                "ladder/*/windowDerivation.json", "ladder/*/physiologyChecks.csv", "ladder/*/results/pilotStatus.json",
                "ladder/*/results/calibration.json", "ladder/*/decimated_out*/results/pilotStatus.json",
                "bath/bath/bathSummary.json", "fibres/fibreCheck.json", "fibres/*.config.json",
                "bathAnalysis/bathAnalysisSummary.json", "bathAnalysis/bathComparisons.csv",
                "bathAnalysis/*/bathGroundTruth.csv", "bathAnalysis/*/bathModel.json",
                "bathAnalysis/*/run/results/pilotStatus.json", "bathAnalysis/*/run/results/calibration.json",
                "bathAnalysis/*/run/results/clinical/changePointFits.csv",
                "kit.log", "tests/tests.log"]
    copied = []
    for pattern in patterns:
        for path in out.glob(pattern):
            if folder in path.parents:
                continue
            target = folder / path.relative_to(out)
            target.parent.mkdir(parents=True, exist_ok=True)
            if path.suffix in {".json", ".csv", ".log", ".txt"}:
                target.write_text(common.redact(path.read_text(errors="replace")))
            else:
                shutil.copy2(path, target)
            copied.append(str(path.relative_to(out)))
    for log in out.glob("**/*.log"):          # first and last 40 lines of every simulator log, for review
        if folder in log.parents:
            continue
        lines = log.read_text(errors="replace").splitlines()
        excerpt = lines[:40] + (["..."] + lines[-40:] if len(lines) > 80 else lines[40:])
        target = folder / "logExcerpts" / (str(log.relative_to(out)).replace("/", "__") + ".txt")
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(common.redact("\n".join(excerpt)))
    common.saveJson(folder / "index.json", dict(createdUtc=common.nowUtc(), files=copied))
    return dict(status="passed", files=len(copied))


HANDLERS = dict(fibres=stageFibres, bathAnalysis=stageBathAnalysis,
                preflight=stagePreflight, tests=stageTests, mpi=stageMpi, smoke=stageSmoke, niederer=stageNiederer,
                cv=stageCv, ladder=stageLadder, bath=stageBath, report=stageReport)


def commandRun(arguments):
    settings = common.loadSettings()
    out = Path(arguments.out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    log = open(out / "kit.log", "a")
    sys.stdout = Tee(sys.__stdout__, log)
    print(f"\n=== readiness kit run {common.nowUtc()} --np {arguments.np} --tier {arguments.tier} ===")
    stages = STAGES[STAGES.index(arguments.start):] if arguments.start else STAGES
    if arguments.only:
        stages = [arguments.only]
    for stage in stages:
        folder = out / stage
        statusPath = folder / "stage.json"
        if statusPath.exists() and stage != "report":
            previous = json.loads(statusPath.read_text())
            if previous.get("status") in {"passed", "reported", "skipped"} and stage not in (arguments.rerun or []):
                print(f"[{stage}] already {previous['status']} - skipping")
                continue
        folder.mkdir(parents=True, exist_ok=True)
        started = common.nowUtc()
        print(f"[{stage}] started {started}", flush=True)
        try:
            result = HANDLERS[stage](folder, arguments, settings)
        except Exception as error:  # noqa: BLE001 - record any failure, then stop if blocking
            result = dict(status="failed", error=f"{type(error).__name__}: {error}", traceback=traceback.format_exc())
        result.update(started=started, finished=common.nowUtc())
        common.saveJson(statusPath, result)
        print(f"[{stage}] {result['status']}" + (f": {result.get('error')}" if result["status"] == "failed" else ""), flush=True)
        if result["status"] == "failed" and stage in BLOCKING:
            print(f"Stopped at blocking stage '{stage}'. Fix the cause, then rerun the same command to resume.")
            return 1
    if "report" in stages:
        print("Kit finished. Copy", out / "report", "back to the Mac (Masters_V2/_verification/readiness/).")
    return 0


def commandStatus(arguments):
    out = Path(arguments.out)
    for stage in STAGES:
        path = out / stage / "stage.json"
        state = json.loads(path.read_text()) if path.exists() else {}
        print(f"{stage:10} {state.get('status', 'not started'):10} {state.get('finished', '')}  {state.get('error', '')}")
    running = sorted(out.glob("**/*.log"), key=lambda p: p.stat().st_mtime)[-3:] if out.exists() else []
    for log in running:
        lines = [l for l in log.read_text(errors="replace").splitlines() if l.strip()]
        print(f"latest: {log.relative_to(out)}: {lines[-1] if lines else ''}")
    return 0


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("command", choices=["plan", "run", "status", "report"])
    parser.add_argument("--out", default=str(common.REPOSITORY / "3D_Code" / "runs" / "readiness"))
    parser.add_argument("--np", type=int, default=1, help="MPI processes (needs MPIEXEC_BIN when > 1)")
    parser.add_argument("--tier", choices=["A", "B"], default="A", help="B adds the 0.125 mm ladder run")
    parser.add_argument("--from", dest="start", choices=STAGES, help="start at this stage (earlier stages must have passed)")
    parser.add_argument("--only", choices=STAGES)
    parser.add_argument("--rerun", nargs="*", choices=STAGES, help="re-run these stages even if they passed")
    parser.add_argument("--skip-disk-check", dest="skipDiskCheck", action="store_true")
    arguments = parser.parse_args()
    if arguments.command == "plan":
        commandPlan(arguments); return 0
    if arguments.command == "status":
        return commandStatus(arguments)
    if arguments.command == "report":
        arguments.only = "report"
    return commandRun(arguments)


if __name__ == "__main__":
    sys.exit(main())
