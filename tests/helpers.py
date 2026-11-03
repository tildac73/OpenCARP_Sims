"""Shared test fixtures: a structured tetrahedral box and a synthetic openCARP-like run."""
import copy
import itertools
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def boxMesh(lower, upper, spacing):
    """Regular lattice box, 6 tetrahedra per cube (conforming Kuhn decomposition)."""
    axes = [np.linspace(lo, hi, round((hi - lo) / spacing) + 1) for lo, hi in zip(lower, upper)]
    grid = np.stack(np.meshgrid(*axes, indexing="ij"), axis=-1)
    shape = grid.shape[:3]
    cells = np.array(list(itertools.product(*(range(n - 1) for n in shape))))
    elements = []
    for order in itertools.permutations(range(3)):
        corner = cells.copy()
        ids = [np.ravel_multi_index(corner.T, shape)]
        for axis in order:
            corner = corner.copy(); corner[:, axis] += 1
            ids.append(np.ravel_multi_index(corner.T, shape))
        elements.append(np.stack(ids, axis=1))
    return grid.reshape(-1, 3), np.concatenate(elements)


def finalConfig():
    return json.loads((ROOT / "config" / "final_model.json").read_text())


def smallConfig():
    return json.loads((ROOT / "config" / "test_small.json").read_text())
