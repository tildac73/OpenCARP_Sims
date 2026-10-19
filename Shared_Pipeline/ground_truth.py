"""Tetrahedron-based contribution reconstruction for the x-fibre slab.

Each tetrahedron retains its own conductivity, volume and linear Vm gradient.
We never average conductivity across the viable-lesion interface. The finite
lead field is evaluated at each tetrahedron's centroid, a quadrature approximation
checked against openCARP recovery before reporting FOV. The common prefactor is
omitted and one signed scale is estimated from intact unipoles.

The primary radius is the persistent 80% waveform-reconstruction radius of a
full-thickness cylinder. Version 1's node-energy radius is removed: it is not
interchangeable with an element-energy definition.
"""
from dataclasses import dataclass
from pathlib import Path
import numpy as np
from scipy import sparse
import egm
import igb_io


@dataclass
class ElementGeometry:
    nodePositions: np.ndarray       # node, xyz; mm
    elementNodes: np.ndarray        # element, four node IDs
    centres: np.ndarray             # element, xyz; mm
    gradients: np.ndarray           # element, xyz, four shape functions; 1/mm
    volumes: np.ndarray             # element; mm^3
    conductivity: np.ndarray        # element, xyz; S/m, fibres along x


def readParameters(parameterPath):
    parameters = {}
    for line in Path(parameterPath).read_text().splitlines():
        line = line.split("#", 1)[0]
        if "=" in line:
            name, value = line.split("=", 1)
            parameters[name.strip()] = value.strip().strip('"')
    return parameters


def readElementGeometry(runFolder, condition):
    """Use local mesh paths so copied results can be reanalysed on another host."""
    runFolder = Path(runFolder)
    meshBase = runFolder / "mesh" / condition
    with open(str(meshBase) + ".pts") as stream:
        nodeCount = int(stream.readline())
        positions = np.loadtxt(stream, ndmin=2) / 1000
    with open(str(meshBase) + ".elem") as stream:
        elementCount = int(stream.readline())
        rows = np.loadtxt(stream, dtype=str, ndmin=2)
    if positions.shape != (nodeCount, 3) or rows.shape != (elementCount, 6):
        raise ValueError("Mesh headers do not match their data")
    if not np.all(rows[:, 0] == "Tt"):
        raise ValueError("Contribution reconstruction requires tetrahedra")
    elements = rows[:, 1:5].astype(int)
    tags = rows[:, 5].astype(int)
    if elements.min() < 0 or elements.max() >= nodeCount:
        raise ValueError("Invalid mesh node index")
    with open(str(meshBase) + ".lon") as stream:
        stream.readline()
        fibres = np.loadtxt(stream, ndmin=2)
    if len(fibres) != elementCount or not np.allclose(np.abs(fibres[:, :3]), [1, 0, 0]):
        raise ValueError("This pilot implementation requires x-aligned fibres")
    parameters = readParameters(runFolder / condition / "parameters.par")
    conductivityByTag = {}
    for region in range(int(parameters["num_gregions"])):
        prefix = f"gregion[{region}]"
        values = [float(parameters[f"{prefix}.g_i{axis}"]) for axis in "ltn"]
        for index in range(int(parameters[f"{prefix}.num_IDs"])):
            conductivityByTag[int(parameters[f"{prefix}.ID[{index}]"])] = values
    if set(tags) - conductivityByTag.keys():
        raise ValueError("Missing intracellular conductivity for an element tag")
    conductivity = np.asarray([conductivityByTag[tag] for tag in tags])
    vertices = positions[elements]
    # V = a + bx + cy + dz in each tetrahedron. Rows are its four vertices.
    matrices = np.concatenate([np.ones((elementCount, 4, 1)), vertices], axis=2)
    volumes = np.abs(np.linalg.det(matrices)) / 6
    if np.any(volumes <= 0) or not np.isfinite(volumes).all():
        raise ValueError("Invalid tetrahedron volume")
    gradients = np.linalg.inv(matrices)[:, 1:, :]
    return ElementGeometry(positions, elements, vertices.mean(axis=1), gradients,
                           volumes, conductivity)


def finiteRecordingField(samplePositions, weights, elementCentres):
    """Average cylinder-surface samples and apply the actual recording weights."""
    field = np.zeros_like(elementCentres)
    for electrode in np.flatnonzero(np.abs(weights) > 1e-12):
        samples = samplePositions[electrode]
        for sample in samples:
            displacement = sample - elementCentres
            distance = np.linalg.norm(displacement, axis=1)
            if np.any(distance == 0):
                raise ValueError("A recovery point coincides with an element centroid")
            field += weights[electrode] * displacement / distance[:, None]**3 / len(samples)
    return field


def ringOperator(geometry, samplePositions, weights, anchor, ringStepMm):
    """Map nodal Vm to a trace for each cylindrical tissue ring using sparse weights.

    This sums conductivity * grad(Vm) * lead field * volume over tetrahedra.
    Constructing the map once avoids a huge element-by-time source array.
    Ring membership uses element centroids and includes the full thickness.
    """
    field = finiteRecordingField(samplePositions, weights, geometry.centres)
    coefficients = np.einsum(
        "ec,ec,ecj,e->ej", field, geometry.conductivity,
        geometry.gradients, geometry.volumes,
    )
    distance = np.linalg.norm(geometry.centres[:, :2] - anchor[:2], axis=1)
    rings = np.floor(distance / ringStepMm).astype(int)
    operator = sparse.coo_matrix(
        (coefficients.ravel(), (np.repeat(rings, 4), geometry.elementNodes.ravel())),
        shape=(int(rings.max()) + 1, len(geometry.nodePositions)),
    ).tocsr()
    radii = (np.arange(operator.shape[0]) + 1) * ringStepMm
    return operator, radii


def readFilteredVoltage(simulationFolder, frames, bandHz, sampleRateHz, filterOrder):
    """Filter nodal Vm in small batches; linear filtering commutes with recovery."""
    voltage = igb_io.read(Path(simulationFolder) / "vm.igb", frames).T
    sections = egm.makeFilter(bandHz, sampleRateHz, filterOrder)
    if sections is not None:
        for first in range(0, len(voltage), 1000):
            voltage[first:first + 1000] = egm.applyFilter(sections, voltage[first:first + 1000])
    return voltage


def reconstructRecording(geometry, voltage, samplePositions, weights, anchor, ringStepMm):
    operator, radii = ringOperator(geometry, samplePositions, weights, anchor, ringStepMm)
    ringTraces = operator @ voltage
    fullTrace = ringTraces.sum(axis=0)
    energy = float(fullTrace @ fullTrace)
    if not np.isfinite(energy) or energy <= 1e-24:
        return fullTrace, np.nan, radii, np.full(len(radii), np.nan)
    cumulative = np.cumsum(ringTraces, axis=0)
    residualFraction = np.sum((fullTrace - cumulative)**2, axis=1) / energy
    # The criterion must remain satisfied at every larger radius, not just once.
    failed = np.flatnonzero(residualFraction > 0.2)
    firstPersistent = int(failed[-1] + 1) if len(failed) else 0
    radius = float(radii[firstPersistent]) if firstPersistent < len(radii) else np.nan
    return fullTrace, radius, radii, 1 - residualFraction
