"""Reading and writing openCARP files and our own JSON records.

Units: openCARP meshes are in micrometres; every function here returns millimetres.
"""
import json
from pathlib import Path
import re
import shutil

import numpy as np


# ----------------------------------------------------------------- JSON
def _jsonDefault(value):
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Cannot store {type(value).__name__} in JSON")


def saveJson(path, values):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(values, indent=2, default=_jsonDefault, allow_nan=True) + "\n")


def loadJson(path):
    return json.loads(Path(path).read_text())


# ----------------------------------------------------------------- meshes
def readMesh(meshBase):
    """openCARP tetrahedral mesh -> positions (mm), elements (E, 4), tags (E,), fibre vectors (E, 3)."""
    meshBase = str(meshBase)
    with open(meshBase + ".pts") as stream:
        count = int(stream.readline())
        positions = np.loadtxt(stream, ndmin=2)[:count] / 1000.0
    with open(meshBase + ".elem") as stream:
        count = int(stream.readline())
        rows = np.loadtxt(stream, dtype=str, ndmin=2)
    if len(rows) != count:
        raise ValueError(f"{meshBase}.elem: header says {count} elements, file has {len(rows)}")
    if not np.all(rows[:, 0] == "Tt"):
        raise ValueError(f"{meshBase}: expected tetrahedra")
    with open(meshBase + ".lon") as stream:
        stream.readline()
        fibres = np.loadtxt(stream, ndmin=2)[:count, :3]
    return positions, rows[:, 1:5].astype(int), rows[:, 5].astype(int), fibres


def writeVertices(path, nodes, kind="extra"):
    nodes = np.asarray(nodes, int)
    if not len(nodes):
        raise ValueError(f"{path}: no vertices selected")
    with open(path, "w") as stream:
        stream.write(f"{len(nodes)}\n{kind}\n")
        np.savetxt(stream, nodes, fmt="%d")


def readVertices(path):
    with open(path) as stream:
        count = int(stream.readline())
        stream.readline()
        nodes = np.loadtxt(stream, dtype=np.int64, ndmin=1)
    if len(nodes) != count:
        raise ValueError(f"{path}: header says {count} vertices, file has {len(nodes)}")
    return nodes


# ----------------------------------------------------------------- IGB time series
def igbHeader(path):
    """(fields, node count, frame count, dtype) of a scalar IGB file; checks the file is complete."""
    with open(path, "rb") as stream:
        raw = stream.read(1024)
    fields = dict(token.split(":", 1) for token in raw.decode("latin-1").split() if ":" in token)
    nodes = int(np.prod([int(fields.get(axis, 1)) for axis in ("x", "y", "z")]))
    frames = int(fields["t"])
    types = {"float": "f4", "double": "f8"}
    if fields.get("type") not in types:
        raise ValueError(f"{path}: expected scalar float or double IGB data")
    byteOrder = ">" if "big" in fields.get("systeme", "little") else "<"
    dtype = np.dtype(byteOrder + types[fields["type"]])
    if Path(path).stat().st_size < 1024 + nodes * frames * dtype.itemsize:
        raise ValueError(f"{path}: incomplete IGB file")
    return fields, nodes, frames, dtype


def igbTimes(path):
    fields, _, frames, _ = igbHeader(path)
    step = float(fields["inc_t"]) if "inc_t" in fields else float(fields["dim_t"]) / (frames - 1)
    if not np.isfinite(step) or step <= 0:
        raise ValueError(f"{path}: invalid time increment")
    return float(fields.get("org_t", 0)) + np.arange(frames) * step


def readIgb(path, frames=slice(None)):
    """Selected frames as float32 (time, nodes); the file is memory-mapped, not loaded whole."""
    fields, nodes, count, dtype = igbHeader(path)
    values = np.memmap(path, dtype=dtype, mode="r", offset=1024, shape=(count, nodes))
    selected = np.array(values[frames], dtype=np.float32)
    selected *= float(fields.get("facteur", 1))
    selected += float(fields.get("zero", 0))
    return selected


def mergeIgb(first, second, target):
    """target = all frames of first, then second without its first frame (it repeats first's last time)."""
    rawA = open(first, "rb").read(1024).decode("latin-1")
    fieldsA, nodes, framesA, dtype = igbHeader(first)
    fieldsB, nodesB, framesB, dtypeB = igbHeader(second)
    if (nodes, dtype, fieldsA.get("inc_t")) != (nodesB, dtypeB, fieldsB.get("inc_t")):
        raise ValueError(f"Cannot merge {first} and {second}: different nodes, type or time step")
    total = framesA + framesB - 1
    text = re.sub(r"(?<![\w])t:\d+", f"t:{total}", rawA, count=1)
    text = re.sub(r"dim_t:\S+", f"dim_t:{(total - 1) * float(fieldsA['inc_t']):g}", text, count=1)
    body = text.rstrip(" \r\n\x0c\x00")
    if len(body) > 1022:
        raise ValueError("IGB header too long")
    frameBytes = nodes * dtype.itemsize
    with open(target, "wb") as out:
        out.write((body + " " * (1023 - len(body)) + "\x0c").encode("latin-1"))
        with open(first, "rb") as stream:
            stream.seek(1024)
            shutil.copyfileobj(stream, out, 64 << 20)
        with open(second, "rb") as stream:
            stream.seek(1024 + frameBytes)
            shutil.copyfileobj(stream, out, 64 << 20)


# ----------------------------------------------------------------- activation times
def readActivations(path):
    """openCARP LAT file with lats[].all = 1: one 'node<TAB>time' row per event -> (nodes, times)."""
    text = Path(path).read_text().strip()
    if not text:
        return np.zeros(0, int), np.zeros(0)
    rows = np.loadtxt(path, ndmin=2)
    return rows[:, 0].astype(int), rows[:, 1].astype(float)


def firstActivation(nodes, times, nodeCount, startMs, endMs):
    """Earliest activation of each node in [startMs, endMs); NaN where none."""
    result = np.full(nodeCount, np.inf)
    inside = (times >= startMs) & (times < endMs)
    np.minimum.at(result, nodes[inside], times[inside])
    result[~np.isfinite(result)] = np.nan
    return result


def mergeActivations(first, second, target, restartMs):
    """Activation events before restartMs from the prepacing run, from restartMs on from the restarted run."""
    rows = []
    for path, keep in [(first, lambda t: t < restartMs), (second, lambda t: t >= restartMs)]:
        nodes, times = readActivations(path)
        rows += [(n, t) for n, t in zip(nodes, times) if keep(t)]
    np.savetxt(target, np.array(rows) if rows else np.zeros((0, 2)), fmt=["%d", "%.6f"], delimiter="\t")
