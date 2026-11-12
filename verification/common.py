"""Shared helpers of the methodology verification harness.

The harness produces the numbers that the thesis Methodology cites for each modelling and numerical
choice. It never edits config/ or efov/: every check reads the production configuration
(config/model.json, config/model.par) and, where a check needs a different mesh spacing, time step
or protocol, writes a *variant configuration* into its own output folder and runs that. The variant
files are kept beside the results, so each number can be traced to the exact settings that made it.

Every check returns a dict with
    supports   the Methodology statement the number backs
    method     how it was obtained (one or two sentences)
    kind       'derivation' (arithmetic on configured values), 'numerical' (computed here),
               'simulation' (openCARP run) or 'estimator' (statistical recovery test)
    values     the numbers
and is written to <out>/<check>.json with a provenance block (software versions, git revision, time).
A check that fails is recorded with its error and traceback; it never stops the other checks and is
never replaced by a placeholder value.
"""
import copy
import datetime
import platform
import shutil
import subprocess
import sys
import traceback
from pathlib import Path

REPOSITORY = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY))

from efov import config as cfg  # noqa: E402
from efov.io import loadJson, saveJson  # noqa: E402

PRODUCTION_CONFIG = REPOSITORY / "config" / "model.json"


def gitRevision():
    try:
        revision = subprocess.run(["git", "-C", str(REPOSITORY), "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
        dirty = subprocess.run(["git", "-C", str(REPOSITORY), "status", "--porcelain", "--untracked-files=no"],
                               capture_output=True, text=True).stdout.strip()
        return revision + ("+uncommitted" if dirty else "")
    except OSError:
        return "unknown"


def softwareVersions():
    versions = dict(python=platform.python_version(), platform=platform.platform())
    for name in ("numpy", "scipy", "pandas"):
        try:
            versions[name] = __import__(name).__version__
        except ImportError:
            versions[name] = "not installed"
    for executable in ("openCARP", "bench"):
        versions[executable + "Path"] = shutil.which(executable) or "not found"
    try:
        from importlib.metadata import version
        versions["carputils"] = version("carputils")
    except Exception:                                   # noqa: BLE001 - absent carputils is a normal state on the analysis machine
        versions["carputils"] = "not installed"
    return versions


def openCarpBuildInfo():
    """Text openCARP prints for +Help topics is long; the build line is enough to identify the binary."""
    path = shutil.which("openCARP")
    if not path:
        return "openCARP not found"
    try:
        out = subprocess.run([path, "-buildinfo"], capture_output=True, text=True, timeout=60)
        return (out.stdout + out.stderr).strip()[:2000]
    except (OSError, subprocess.TimeoutExpired) as error:
        return f"could not query: {error}"


def provenance():
    return dict(gitRevision=gitRevision(), versions=softwareVersions(),
                writtenAt=datetime.datetime.now().astimezone().isoformat(timespec="seconds"),
                command=" ".join(sys.argv))


def runChecks(checks, out, only=None):
    """Run each (name, function) pair, write <out>/<name>.json, and return {name: status}."""
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    status = {}
    for name, function in checks:
        if only and name not in only:
            continue
        print(f"[{name}] running ...", flush=True)
        try:
            result = function()
            result["status"] = "completed"
        except Exception as error:                      # noqa: BLE001 - recorded with its traceback, never hidden
            result = dict(status="failed", error=f"{type(error).__name__}: {error}", traceback=traceback.format_exc())
        result["provenance"] = provenance()
        saveJson(out / f"{name}.json", result)
        status[name] = result["status"]
        print(f"[{name}] {result['status']}" + (f": {result.get('error')}" if result["status"] == "failed" else ""), flush=True)
    return status


def variantConfig(folder, fixedChanges=None, sweepOverrides=None, parOverrides=None, source=PRODUCTION_CONFIG):
    """A configuration that differs from the production one in stated ways, written to <folder>.

    fixedChanges: {"dotted.path.in.fixed": value}, e.g. {"tissue.spacingInPlaneMm": 0.125}.
    sweepOverrides: {sweep variable: value} (the same mechanism the parameter sweeps use).
    parOverrides: {openCARP key: value} applied on top of model.par.
    The folder receives model.json (changed), copies of model.par and leadfield.par, and
    changes.json listing exactly what differs from production. Returns the loaded configuration.
    """
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    model = loadJson(source)
    for dotted, value in (fixedChanges or {}).items():
        target = model["fixed"]
        keys = dotted.split(".")
        for key in keys[:-1]:
            target = target[key]
        if keys[-1] not in target:
            raise KeyError(f"fixed.{dotted} is not a setting of {source}")
        target[keys[-1]] = copy.deepcopy(value)
    saveJson(folder / "model.json", model)
    for name in (model["parFile"], model["leadFieldParFile"]):
        shutil.copyfile(Path(source).parent / name, folder / name)
    saveJson(folder / "changes.json", dict(source=str(source), fixedChanges=fixedChanges or {},
                                           sweepOverrides=sweepOverrides or {}, parOverrides=parOverrides or {}))
    return cfg.loadConfig(folder / "model.json", sweepOverrides, parOverrides)


def productionConfig():
    return cfg.loadConfig(PRODUCTION_CONFIG)


def relativeDifference(value, reference):
    return float((value - reference) / reference)


def richardson(coarse, medium, fine, ratio=2.0):
    """Observed order p and extrapolated value from three levels with a constant refinement ratio.

    Returns (order, extrapolated) or (nan, nan) when the three values are not monotonic (the usual
    sign that the coarsest level lies outside the asymptotic range), in which case nothing is extrapolated.
    """
    import math
    first, second = medium - coarse, fine - medium
    if first == 0 or second == 0 or first * second < 0:
        return float("nan"), float("nan")
    order = math.log(abs(first / second)) / math.log(ratio)
    return float(order), float(fine + second / (ratio ** order - 1))
