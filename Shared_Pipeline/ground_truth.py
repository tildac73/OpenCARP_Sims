"""Tetrahedron-based contribution reconstruction for the slab.

Fibres: any orientation per element (fibres.py, Step 7). With every fibre along x the
conductivities stay diagonal in mesh axes and the arithmetic is exactly the v2-pilot's.

Each tetrahedron retains its own conductivity, volume and linear Vm gradient.
We never average conductivity across the viable-lesion interface. The finite
lead field is evaluated at each tetrahedron's centroid, a quadrature approximation
checked against openCARP recovery before reporting FOV. The common prefactor is
omitted and one signed scale is estimated from intact unipoles.

Source conductivity (28 Oct 2026, DL-2026-10-28-01): the monodomain conductivity
G_i G_e / (G_i + G_e), which verify_synthesis.py showed reproduces openCARP phie recovery
in amplitude as well as shape. Before this change the intracellular conductivity was used;
that differs only by a constant when G_i and G_e have equal anisotropy ratios (true in
boundaryPilotV2), so pilot FOV80 values are unchanged, but not otherwise.

The primary radius is the persistent 80% waveform-reconstruction radius of a
full-thickness cylinder. Version 1's node-energy radius is removed: it is not
interchangeable with an element-energy definition.
"""
from dataclasses import dataclass
from pathlib import Path
import numpy as np
from scipy import sparse
import egm
import electrode_synthesis
import fibres as fibreTools
import lead_field
import igb_io


@dataclass
class ElementGeometry:
    nodePositions: np.ndarray       # node, xyz; mm
    elementNodes: np.ndarray        # element, four node IDs
    centres: np.ndarray             # element, xyz; mm
    gradients: np.ndarray           # element, xyz, four shape functions; 1/mm
    volumes: np.ndarray             # element; mm^3
    conductivity: np.ndarray        # element, principal (fibre, sheet, normal) values; S/m
    extracellularConductivity: np.ndarray = None  # element, principal values; S/m (monodomain source term)
    fibreFrames: np.ndarray = None  # element, rows (fibre, sheet, normal) unit vectors; None = fibres along x,
                                    # so principal values are the x, y, z diagonal (v2-pilot arithmetic)


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
        vectorsPerElement = int(stream.readline().split()[0])
        fibreRows = np.loadtxt(stream, ndmin=2)
    if vectorsPerElement not in (1, 2) or fibreRows.shape != (elementCount, 3 * vectorsPerElement):
        raise ValueError("Fibre file does not match the mesh (expected 1 or 2 vectors per element)")
    parameters = readParameters(runFolder / condition / "parameters.par")
    conductivityByTag, extracellularByTag = {}, {}
    for region in range(int(parameters["num_gregions"])):
        prefix = f"gregion[{region}]"
        values = [float(parameters[f"{prefix}.g_i{axis}"]) for axis in "ltn"]
        extracellular = [float(parameters[f"{prefix}.g_e{axis}"]) for axis in "ltn"]
        for index in range(int(parameters[f"{prefix}.num_IDs"])):
            conductivityByTag[int(parameters[f"{prefix}.ID[{index}]"])] = values
            extracellularByTag[int(parameters[f"{prefix}.ID[{index}]"])] = extracellular
    if set(tags) - conductivityByTag.keys():
        raise ValueError("Missing intracellular conductivity for an element tag")
    sheets = fibreRows[:, 3:6] if vectorsPerElement == 2 else None
    return buildElementGeometry(positions, elements, tags, fibreRows[:, :3], conductivityByTag,
                                extracellularByTag, sheets)


def buildElementGeometry(positions, elements, tags, fibres, conductivityByTag, extracellularByTag, sheets=None):
    """Geometry from mesh arrays. conductivityByTag: tag -> (fibre, sheet, normal) S/m."""
    elementCount = len(elements)
    conductivity = np.asarray([conductivityByTag[tag] for tag in tags], float)
    extracellularConductivity = np.asarray([extracellularByTag[tag] for tag in tags], float)
    if sheets is None and fibreTools.isXAligned(fibres):
        frames = None
    else:
        if sheets is None and not (np.allclose(conductivity[:, 1], conductivity[:, 2])
                                   and np.allclose(extracellularConductivity[:, 1], extracellularConductivity[:, 2])):
            raise ValueError("Sheet and normal conductivities differ but the fibre file has no sheet vectors")
        frames = fibreTools.localFrames(fibres, sheets)
    vertices = positions[elements]
    # V = a + bx + cy + dz in each tetrahedron. Rows are its four vertices.
    matrices = np.concatenate([np.ones((elementCount, 4, 1)), vertices], axis=2)
    volumes = np.abs(np.linalg.det(matrices)) / 6
    if np.any(volumes <= 0) or not np.isfinite(volumes).all():
        raise ValueError("Invalid tetrahedron volume")
    gradients = np.linalg.inv(matrices)[:, 1:, :]
    return ElementGeometry(positions, elements, vertices.mean(axis=1), gradients,
                           volumes, conductivity, extracellularConductivity, frames)


def finiteRecordingField(samplePositions, weights, elementCentres, vertices=None, quadrature=None):
    """Average cylinder-surface samples and apply the actual recording weights.

    quadrature: optional {"nearRadiusMm": r, "subdivisionLevel": k} (lead_field.py, Step 4).
    None keeps the one-point centroid rule used for the v2-pilot results.
    """
    quadrature = quadrature or {}
    near, level = quadrature.get("nearRadiusMm", 0.0), quadrature.get("subdivisionLevel", 0)
    field = np.zeros_like(elementCentres)
    for electrode in np.flatnonzero(np.abs(weights) > 1e-12):
        samples = samplePositions[electrode]
        sampleFields = lead_field.averageLeadField(samples, vertices, elementCentres, near, level)
        field += weights[electrode] * sampleFields.mean(axis=0)
    return field


def ringOperator(geometry, samplePositions, weights, anchor, ringStepMm, source="monodomain", quadrature=None):
    """Map nodal Vm to a trace for each cylindrical tissue ring using sparse weights.

    This sums conductivity * grad(Vm) * lead field * volume over tetrahedra.
    Constructing the map once avoids a huge element-by-time source array.
    Ring membership uses element centroids and includes the full thickness.
    """
    vertices = geometry.nodePositions[geometry.elementNodes] if quadrature else None
    field = finiteRecordingField(samplePositions, weights, geometry.centres, vertices, quadrature)
    coefficients = electrode_synthesis.elementCoefficients(
        field, electrode_synthesis.sourceConductivity(geometry, source), geometry.gradients, geometry.volumes)
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


def reconstructRecording(geometry, voltage, samplePositions, weights, anchor, ringStepMm, quadrature=None):
    operator, radii = ringOperator(geometry, samplePositions, weights, anchor, ringStepMm, quadrature=quadrature)
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
