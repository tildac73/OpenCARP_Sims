"""openCARP parameter files: read config/model.par, apply overrides, write the resolved file of a run.

A parameter file is `key = value` lines; `#` starts a comment. The resolved file of a run is the
model file with each overridden value replaced in place (the line keeps its comment and gains a
"[override]" note), followed by the run-specific lines (mesh, vertex files, simID, restart). That
one file reproduces the openCARP run: `openCARP +F resolved.par` (FM-01).
"""
from pathlib import Path
import re

LINE = re.compile(r"^\s*([A-Za-z_][\w\[\]\.]*)\s*=\s*(.*?)\s*(#.*)?$")


def readPar(path):
    """Ordered {key: value text} of a parameter file. Repeated keys are an error."""
    values = {}
    for number, line in enumerate(Path(path).read_text().splitlines(), start=1):
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        match = LINE.match(line)
        if not match:
            raise ValueError(f"{path}:{number}: not a 'key = value' line: {line!r}")
        key, value = match.group(1), match.group(2)
        if key in values:
            raise ValueError(f"{path}:{number}: {key} is set twice")
        values[key] = value
    return values


def parValue(values, key, kind=float):
    """One value of a parsed parameter file, converted (float, int or str)."""
    if key not in values:
        raise KeyError(f"Parameter {key} is not set in the parameter file")
    text = values[key].strip().strip('"')
    return int(float(text)) if kind is int else (text if kind is str else float(text))


def formatValue(value):
    if isinstance(value, bool):
        return str(int(value))
    if isinstance(value, float):
        return repr(value)
    return str(value)


def resolvedText(modelParPath, overrides, runLines, header=""):
    """Text of the resolved parameter file: model.par with overrides in place, then run-specific lines.

    overrides: {key: value} for keys that exist in model.par (sweep values, tuned beta).
    runLines:  {key: value} for keys that must not exist in model.par (paths and run controls).
    """
    lines = Path(modelParPath).read_text().splitlines()
    known = readPar(modelParPath)
    unknown = set(overrides) - set(known)
    if unknown:
        raise KeyError(f"Overrides name parameters that model.par does not set: {sorted(unknown)}")
    clash = set(runLines) & set(known)
    if clash:
        raise KeyError(f"Run-specific parameters are also set in model.par: {sorted(clash)}")
    out = [header] if header else []
    for line in lines:
        match = LINE.match(line) if line.strip() and not line.lstrip().startswith("#") else None
        if match and match.group(1) in overrides:
            key = match.group(1)
            out.append(f"{key} = {formatValue(overrides[key])}  # [override; model.par: {match.group(2)}]")
        else:
            out.append(line)
    out += ["", "# ---- run-specific (written by efov; not in model.par) ----"]
    out += [f"{key} = {formatValue(value)}" for key, value in runLines.items()]
    return "\n".join(out) + "\n"


def writeResolved(modelParPath, overrides, runLines, target, header=""):
    Path(target).write_text(resolvedText(modelParPath, overrides, runLines, header))
    return Path(target)


def setValue(path, key, value, note):
    """Rewrite one existing key of a parameter file in place (used by `efov tune --write-par`)."""
    lines = Path(path).read_text().splitlines()
    for index, line in enumerate(lines):
        match = LINE.match(line) if line.strip() and not line.lstrip().startswith("#") else None
        if match and match.group(1) == key:
            lines[index] = f"{key} = {formatValue(value)}  # {note}"
            Path(path).write_text("\n".join(lines) + "\n")
            return
    raise KeyError(f"{key} is not set in {path}")
