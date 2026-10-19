"""Read scalar openCARP IGB files without loading the complete Vm recording."""

from pathlib import Path

import numpy as np


def header(path):
    with open(path, "rb") as stream:
        raw = stream.read(1024)
    fields = dict(token.split(":", 1) for token in raw.decode("latin-1").split() if ":" in token)
    nodes = np.prod([int(fields.get(axis, 1)) for axis in ("x", "y", "z")])
    frames = int(fields["t"])
    byteOrder = ">" if "big" in fields.get("systeme", "little") else "<"
    types = {"float": "f4", "double": "f8"}
    if fields.get("type") not in types:
        raise ValueError(f"{path}: expected scalar float or double IGB data")
    dtype = np.dtype(byteOrder + types[fields["type"]])
    expected = 1024 + int(nodes) * frames * dtype.itemsize
    if nodes <= 0 or frames <= 0 or Path(path).stat().st_size < expected:
        raise ValueError(f"{path}: incomplete IGB file (expected at least {expected} bytes)")
    return fields, int(nodes), frames, dtype


def timeAxis(path):
    fields, _, frames, _ = header(path)
    if "inc_t" in fields:
        step = float(fields["inc_t"])
    elif "dim_t" in fields and frames > 1:
        step = float(fields["dim_t"]) / (frames - 1)
    else:
        raise ValueError(f"{path}: missing IGB time increment")
    if not np.isfinite(step) or step <= 0:
        raise ValueError(f"{path}: invalid time increment {step}")
    return float(fields.get("org_t", 0)) + np.arange(frames) * step


def read(path, frames=slice(None)):
    """Return selected frames as (time, nodes), honouring the fixed 1024-byte header."""
    fields, nodes, count, dtype = header(path)
    values = np.memmap(path, dtype=dtype, mode="r", offset=1024, shape=(count, nodes))
    selected = np.array(values[frames], dtype=np.float32)
    selected *= float(fields.get("facteur", 1))
    selected += float(fields.get("zero", 0))
    return selected
