"""Shared helpers for the readiness kit (3D_Code/readiness).

Nothing here makes a scientific choice; settings come from 3D_Code/configs/readiness_kit.json.
Units follow the pilot: mm and ms in Python, micrometres in openCARP mesh files.
"""
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys

import numpy as np

REPOSITORY = Path(__file__).resolve().parents[2]
for folder in ("3D_Code", "Shared_Pipeline"):
    if str(REPOSITORY / folder) not in sys.path:
        sys.path.insert(0, str(REPOSITORY / folder))

import boundary_setup  # noqa: E402

# READINESS_KIT_SETTINGS lets tests (or a deliberate variant) point at another settings file;
# the file actually used is recorded by the preflight stage.
KIT_SETTINGS = Path(os.environ.get("READINESS_KIT_SETTINGS", REPOSITORY / "3D_Code" / "configs" / "readiness_kit.json"))
TEMPLATE = REPOSITORY / "3D_Code" / "3d_pilot.par"

# Credentials embedded in a repository URL (e.g. the openCARP build banner) must never be
# copied into results that leave the workstation.
_CREDENTIAL = re.compile(r"(://)[^/\s:@]+:[^/\s@]+@")


def loadSettings(path=KIT_SETTINGS):
    return json.loads(Path(path).read_text())


def nowUtc():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def saveJson(path, values):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(values, indent=2, default=_jsonDefault) + "\n")


def _jsonDefault(value):
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Cannot store {type(value)} in JSON")


def redact(text):
    return _CREDENTIAL.sub(r"\1<credentials removed>@", text)


def redactFile(path):
    path = Path(path)
    if path.is_file():
        text = path.read_text(errors="replace")
        cleaned = redact(text)
        if cleaned != text:
            path.write_text(cleaned)


def runLogged(command, logPath, cwd=None, check=True):
    """Run a command with stdout+stderr in logPath; credentials are removed from the log afterwards."""
    logPath = Path(logPath)
    logPath.parent.mkdir(parents=True, exist_ok=True)
    command = [str(value) for value in command]
    print("Running:", " ".join(command), flush=True)
    with open(logPath, "w") as logFile:
        logFile.write("Command: " + " ".join(command) + "\n")
        logFile.flush()
        result = subprocess.run(command, stdout=logFile, stderr=subprocess.STDOUT, cwd=cwd)
    redactFile(logPath)
    if check and result.returncode != 0:
        raise subprocess.CalledProcessError(result.returncode, command)
    return result.returncode


def simulatorPrefix(processCount, nodeCount=None):
    """openCARP launch prefix. Small meshes get fewer ranks (>= 2000 nodes per rank)."""
    simulator = boundary_setup.findExecutable("openCARP", "OPENCARP_BIN")
    if nodeCount is not None:
        processCount = max(1, min(processCount, nodeCount // 2000))
    if processCount <= 1:
        return [simulator], 1
    if not os.environ.get("MPIEXEC_BIN"):
        raise RuntimeError("Set MPIEXEC_BIN to the MPI launcher matching openCARP (see 3D_Code/numerics/README.md).")
    return [boundary_setup.findExecutable("mpiexec", "MPIEXEC_BIN"), "-n", processCount, simulator], processCount


def checkSingleLaunch(logPath):
    boundary_setup.checkSimulationLog(logPath)


# ----------------------------------------------------------------- meshes
def generateBox(folder, name, sizeMm, spacingMm):
    """Regular tetrahedral box from openCARP's mesher, centred on the origin (as the pilot)."""
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    base = folder / name
    command = [boundary_setup.findExecutable("mesher", "MESHER_BIN")]
    for axis in range(3):
        command += [f"-size[{axis}]", sizeMm[axis] / 10, f"-resolution[{axis}]", spacingMm * 1000]
    command += ["-mesh", base]
    runLogged(command, folder / f"{name}.mesher.log")
    positions = boundary_setup.readNodePositions(base)
    elements = boundary_setup.readElements(base)
    half = np.asarray(sizeMm, float) / 2
    if not (np.allclose(positions.min(axis=0), -half) and np.allclose(positions.max(axis=0), half)):
        raise ValueError(f"{name}: mesher output is not the requested centred box")
    return positions, elements


def writeMesh(base, positionsMm, elements, tags, fibres=None):
    """Write .pts (um), .elem (tagged tetrahedra) and .lon (one fibre per element)."""
    base = Path(base)
    with open(str(base) + ".pts", "w") as stream:
        stream.write(f"{len(positionsMm)}\n")
        np.savetxt(stream, np.asarray(positionsMm) * 1000, fmt="%.3f")
    with open(str(base) + ".elem", "w") as stream:
        stream.write(f"{len(elements)}\n")
        rows = np.column_stack([elements, tags]).astype(int)
        np.savetxt(stream, rows, fmt="Tt %d %d %d %d %d")
    fibres = np.tile([1.0, 0.0, 0.0], (len(elements), 1)) if fibres is None else fibres
    with open(str(base) + ".lon", "w") as stream:
        stream.write("1\n")
        np.savetxt(stream, fibres, fmt="%.6f")


def writeVertices(path, nodeIds):
    nodeIds = np.asarray(nodeIds, dtype=int)
    if not len(nodeIds):
        raise ValueError(f"{path}: empty vertex set")
    with open(path, "w") as stream:
        stream.write(f"{len(nodeIds)}\nextra\n")
        np.savetxt(stream, nodeIds, fmt="%d")


# ----------------------------------------------------------------- parameter files
def templateForTissue(conductivities):
    """Pilot template, intact variant (one conductivity region), with the given conductivities."""
    text = TEMPLATE.read_text()
    text = "\n".join(line for line in text.splitlines() if not line.startswith("gregion[1]"))
    text = text.replace("num_gregions = 2", "num_gregions = 1")
    return boundary_setup.applyTissueConductivities(text, conductivities)


def writeParameterFile(path, baseText, values):
    """baseText + one line per resolved value (strings quoted), as boundary_setup.writeParameters."""
    lines = []
    for name, value in values.items():
        if isinstance(value, (Path, str)) and not str(value).isidentifier():
            value = json.dumps(str(value))
        lines.append(f"{name} = {value}\n")
    Path(path).write_text(baseText.rstrip() + "\n\n# Values resolved by the readiness kit\n" + "".join(lines))
    return Path(path)


def smokeCopy(parameterPath, smokeFolder, durationMs=5.0):
    """Short copy of a parameter file for the smoke stage: same names, files and solver settings.

    openCARP rejects a pacing protocol that does not fit inside the run: it requires
    ptcl.duration <= tend and ptcl.duration <= ptcl.bcl <= tend (seen on the workstation build).
    So besides tend and simID, every stimulus's bcl is set to tend and npls to 1. Pulse duration
    and strength are left unchanged; durationMs must exceed every pulse duration in the file.
    This affects only the smoke copy, never the real run.
    """
    text = Path(parameterPath).read_text()
    for name, value in [("tend", durationMs), ("simID", json.dumps(str(smokeFolder)))]:
        pattern = re.compile(rf"^{name}\s*=.*$", re.MULTILINE)
        text, count = pattern.subn(f"{name} = {value}", text)
        if count != 1:
            raise ValueError(f"{parameterPath}: expected exactly one '{name}' line, found {count}")
    for match in re.finditer(r"^stim\[(\d+)\]\.ptcl\.duration\s*=\s*([^\s#]+)", text, re.MULTILINE):
        if float(match.group(2)) >= durationMs:
            raise ValueError(f"{parameterPath}: stim[{match.group(1)}] pulse ({match.group(2)} ms) does not fit "
                             f"in the {durationMs} ms smoke run")
    text = re.sub(r"^(stim\[\d+\]\.ptcl\.bcl)\s*=.*$", rf"\1 = {durationMs}", text, flags=re.MULTILINE)
    text = re.sub(r"^(stim\[\d+\]\.ptcl\.npls)\s*=.*$", r"\1 = 1", text, flags=re.MULTILINE)
    target = Path(smokeFolder).parent / (Path(parameterPath).stem + ".smoke.par")
    target.write_text(text)
    return target


def parameterValue(parameterPath, name):
    match = re.search(rf"^{re.escape(name)}\s*=\s*(.*)$", Path(parameterPath).read_text(), re.MULTILINE)
    return None if match is None else match.group(1).strip().strip('"')


# ----------------------------------------------------------------- outputs
def readActivationFile(path, nodeCount=None):
    """openCARP LAT output -> (nodeIndex, timeMs) arrays.

    Two formats occur: one row per event 'node<TAB>time' (lats[].all = 1; the pilot's
    act-thresh.dat), or one value per node in node order with -1 for no activation.
    """
    text = Path(path).read_text().strip()
    if not text:                                   # no activation at all (e.g. no capture)
        return np.zeros(0, int), np.zeros(0)
    rows = np.loadtxt(path, ndmin=2)
    if rows.shape[1] >= 2:
        return rows[:, 0].astype(int), rows[:, 1].astype(float)
    values = rows[:, 0]
    if nodeCount is not None and len(values) != nodeCount:
        raise ValueError(f"{path}: {len(values)} values for {nodeCount} nodes")
    active = values >= 0
    return np.flatnonzero(active), values[active]


def findActivationFile(simulationFolder):
    """The pilot's LAT file is act-thresh.dat (lats[0].ID = act); accept the other openCARP spelling."""
    folder = Path(simulationFolder)
    preferred = folder / "act-thresh.dat"
    if preferred.is_file():
        return preferred
    matches = sorted(folder.glob("*act*thresh*.dat"))
    if len(matches) != 1:
        raise FileNotFoundError(f"{folder}: expected one LAT file, found {[m.name for m in matches]}")
    return matches[0]


def firstActivationInWindow(nodes, times, nodeCount, startMs, endMs):
    """Earliest activation of each node in [startMs, endMs); NaN if none."""
    result = np.full(nodeCount, np.inf)
    inside = (times >= startMs) & (times < endMs)
    np.minimum.at(result, nodes[inside], times[inside])
    result[~np.isfinite(result)] = np.nan
    return result


def fitPlanarSpeed(positionsMm, timesMs, axis):
    """Least-squares plane t = g.x + c; speed = 1/|g| (mm/ms = m/s); angle of g to the axis."""
    valid = np.isfinite(timesMs)
    if valid.sum() < 4:
        raise ValueError("Too few activated nodes for a plane fit")
    design = np.column_stack([positionsMm[valid], np.ones(valid.sum())])
    coefficients, *_ = np.linalg.lstsq(design, timesMs[valid], rcond=None)
    gradient = coefficients[:3]
    magnitude = float(np.linalg.norm(gradient))
    residual = timesMs[valid] - design @ coefficients
    angle = float(np.degrees(np.arccos(np.clip(gradient[axis] / magnitude, -1, 1)))) if magnitude > 0 else np.nan
    return dict(speedMPerS=1.0 / magnitude if magnitude > 0 else np.nan, directionErrorDeg=angle,
                planeResidual95Ms=float(np.percentile(np.abs(residual), 95)), nodeCount=int(valid.sum()))


def openCarpBanner(logPath):
    """Build hash and tag from an openCARP log (credentials already removed)."""
    text = Path(logPath).read_text(errors="replace") if Path(logPath).is_file() else ""
    found = {}
    for key in ("GIT tag", "GIT hash", "GIT repo", "dependency commits"):
        match = re.search(rf"\*\*\* {key}:\s*(.*)", text)
        if match:
            found[key] = redact(match.group(1).strip())
    return found


def lastProgressSeconds(logPath):
    """Wall-clock seconds from openCARP's last progress row ('time %comp time ctime ETA')."""
    if not Path(logPath).is_file():
        return None
    lastClock = None
    for line in Path(logPath).read_text(errors="replace").splitlines():
        parts = line.split()
        if len(parts) >= 5 and re.fullmatch(r"\d+:\d\d:\d\d", parts[3]):
            hours, minutes, seconds = (int(v) for v in parts[3].split(":"))
            lastClock = hours * 3600 + minutes * 60 + seconds
    return lastClock


def copyIfExists(source, target):
    source = Path(source)
    if source.is_file():
        Path(target).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
