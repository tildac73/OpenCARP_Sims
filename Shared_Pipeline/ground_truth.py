"""Approximate tissue contributions for the regular, x-fibre 3D slab.

The calculation is deliberately separate from openCARP's recovered potentials:
1. estimate grad(Vm) on the regular node grid;
2. multiply by intracellular conductivity and nodal integration volume;
3. project onto the recording's finite-electrode lead field;
4. expand a full-thickness cylinder until the requested energy criterion holds.

The common physical prefactor is omitted. boundary_analysis compares waveform,
polarity and a common fitted scale against openCARP recovery before accepting FOV.
Arrays called sourceDensity have shape (xyz component, node, time).
"""

from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse

import egm
import igb_io


def readParameters(parameterPath):
    parameters = {}
    for line in Path(parameterPath).read_text().splitlines():
        line = line.split("#", 1)[0]
        if "=" in line:
            name, value = line.split("=", 1)
            parameters[name.strip()] = value.strip().strip('"')
    return parameters


def nodeConductivities(simulationFolder, meshBase, nodeCount):
    """Average the adjacent tetrahedra's intracellular conductivities at each node."""
    parameters = readParameters(Path(simulationFolder) / "parameters.par")
    conductivityByTag = {}
    for regionIndex in range(int(parameters["num_gregions"])):
        prefix = f"gregion[{regionIndex}]"
        # Require explicit values: do not silently invent a conductivity default.
        conductivity = [float(parameters[f"{prefix}.g_i{axis}"]) for axis in "ltn"]
        for tagIndex in range(int(parameters[f"{prefix}.num_IDs"])):
            tissueTag = int(parameters[f"{prefix}.ID[{tagIndex}]"])
            conductivityByTag[tissueTag] = conductivity
    elements = pd.read_csv(str(meshBase) + ".elem", sep=r"\s+", skiprows=1, header=None)
    elementNodes = elements.iloc[:, 1:5].to_numpy(int)
    tissueTags = elements.iloc[:, 5].to_numpy(int)
    fibres = np.loadtxt(str(meshBase) + ".lon", skiprows=1, ndmin=2)
    if not np.allclose(np.abs(fibres[:, :3]), [1, 0, 0], atol=1e-3):
        raise ValueError("Contribution calculation requires fibres along x")
    missingTags = set(tissueTags) - conductivityByTag.keys()
    if missingTags:
        raise ValueError(f"Missing conductivity settings for tissue tags {missingTags}")
    elementConductivities = np.array([conductivityByTag[tag] for tag in tissueTags])
    flatNodeIds = elementNodes.ravel()
    adjacentCount = np.bincount(flatNodeIds, minlength=nodeCount)
    if np.any(adjacentCount == 0):
        raise ValueError("Mesh contains a node with no adjacent tetrahedron")
    conductivity = np.empty((nodeCount, 3), dtype=np.float32)
    for axis in range(3):
        total = np.bincount(
            flatNodeIds, weights=np.repeat(elementConductivities[:, axis], 4), minlength=nodeCount
        )
        conductivity[:, axis] = total / adjacentCount
    return conductivity


def sourceDensity(simulationFolder, frames=slice(None)):
    """Return volume-weighted sources, ordered node positions, and grid spacing in mm."""
    simulationFolder = Path(simulationFolder)
    meshBase = (simulationFolder / "meshname.txt").read_text().strip()
    positions = np.loadtxt(meshBase + ".pts", skiprows=1) / 1000
    conductivity = nodeConductivities(simulationFolder, meshBase, len(positions))
    voltage = igb_io.read(simulationFolder / "vm.igb", frames)
    if voltage.shape[1] != len(positions):
        raise ValueError("Vm node count does not match mesh points")

    # Reorder the mesh into a regular array: time, z, y, x (x varies fastest).
    spacingMm = min(np.diff(np.unique(positions[:, axis])).min() for axis in range(3))
    gridIndices = []
    for axis in range(3):
        indices = np.rint((positions[:, axis] - positions[:, axis].min()) / spacingMm).astype(int)
        if not np.allclose(positions[:, axis], positions[:, axis].min() + spacingMm * indices):
            raise ValueError("Contribution calculation needs equal Cartesian grid spacing")
        gridIndices.append(indices)
    gridShape = [indices.max() + 1 for indices in gridIndices]
    flatIndices = np.ravel_multi_index(gridIndices[::-1], gridShape[::-1])
    if np.prod(gridShape) != len(positions) or len(np.unique(flatIndices)) != len(positions):
        raise ValueError("Contribution calculation needs a complete regular 3D node grid")
    nodeOrder = np.argsort(flatIndices)
    positions = positions[nodeOrder]
    conductivity = conductivity[nodeOrder]
    frameCount = len(voltage)
    voltageGrid = voltage[:, nodeOrder].reshape([frameCount] + gridShape[::-1])
    del voltage
    # np.gradient returns z, y, x derivatives here; reverse to x, y, z.
    gradients = np.gradient(voltageGrid, spacingMm, axis=(1, 2, 3))[::-1]
    del voltageGrid

    # Trapezoidal volume weights: half weight on each outer face.
    nodeVolumes = np.full(gridShape[::-1], spacingMm**3, dtype=np.float32)
    for axis in range(3):
        face = [slice(None)] * 3
        for endIndex in [0, -1]:
            face[axis] = endIndex
            nodeVolumes[tuple(face)] *= 0.5
    nodeVolumes = nodeVolumes.ravel()
    sources = np.empty((3, len(positions), frameCount), dtype=np.float32)
    for axis in range(3):
        gradient = gradients[axis].reshape(frameCount, -1)
        sources[axis] = (gradient * (conductivity[:, axis] * nodeVolumes)).T
    return sources, positions, spacingMm


def filterSources(sources, bandHz, sampleRateHz, filterOrder):
    """Apply the same causal filter as the electrodes, in small memory batches."""
    filterSections = egm.makeFilter(bandHz, sampleRateHz, filterOrder)
    if filterSections is None:
        return sources
    filteredSources = np.empty_like(sources)
    batchSize = 4000
    for axis in range(3):
        for firstNode in range(0, sources.shape[1], batchSize):
            nodeSlice = slice(firstNode, firstNode + batchSize)
            filteredSources[axis, nodeSlice] = egm.applyFilter(
                filterSections, sources[axis, nodeSlice]
            )
    return filteredSources


def sourceEnergyTerms(sources):
    """Cache sum over time of source[a] * source[b], for each node.

    Cross terms matter because a node's x/y/z contributions add before squaring.
    einsum performs this sum without a large temporary array.
    """
    energyTerms = np.empty((3, 3, sources.shape[1]))
    for firstAxis in range(3):
        for secondAxis in range(firstAxis, 3):
            crossEnergy = np.einsum(
                "nt,nt->n", sources[firstAxis], sources[secondAxis], dtype=np.float64
            )
            energyTerms[firstAxis, secondAxis] = crossEnergy
            energyTerms[secondAxis, firstAxis] = crossEnergy
    return energyTerms


def pointLeadField(electrodePosition, nodePositions):
    """Gradient of 1/r with respect to tissue position; shape (3, nodes)."""
    displacement = electrodePosition - nodePositions
    squaredDistance = np.sum(displacement**2, axis=1)
    if np.any(squaredDistance == 0):
        raise ValueError("An electrode sample coincides with a tissue node")
    return (displacement / squaredDistance[:, None] ** 1.5).T.astype(np.float32)


def finiteElectrodeFields(samplePositions, nodePositions):
    """Average the point lead fields over each electrode's cylinder samples."""
    fields = []
    for electrodeSamples in samplePositions:
        sampleFields = [pointLeadField(position, nodePositions) for position in electrodeSamples]
        fields.append(np.mean(sampleFields, axis=0))
    return fields


def contributionRadii(
    leadField, sources, energyTerms, anchor, nodePositions, spacingMm, fraction=0.8
):
    """Return reconstruction radius, node-energy radius, full trace and both curves.

    Cylinders include the full tissue thickness. The reconstruction criterion is
    persistent: residual energy must stay <= 1-fraction at every larger radius.
    Radius bins retain the original convention [R-spacing, R).
    """
    lateralDistance = np.hypot(nodePositions[:, 0] - anchor[0], nodePositions[:, 1] - anchor[1])
    ringIndex = (lateralDistance / spacingMm).astype(int)
    ringCount = ringIndex.max() + 1
    fullTrace = np.einsum("cn,cnt->t", leadField, sources)
    totalEnergy = float(np.sum(fullTrace**2))
    if totalEnergy == 0:
        return np.nan, np.nan, fullTrace, None

    # A sparse matrix adds the nodes in each ring without storing every node trace.
    nodeIds = np.arange(len(nodePositions))
    ringSignals = np.zeros((ringCount, sources.shape[2]), dtype=np.float32)
    for axis in range(3):
        ringWeights = sparse.csr_matrix(
            (leadField[axis], (ringIndex, nodeIds)), shape=(ringCount, len(nodePositions))
        )
        ringSignals += ringWeights @ sources[axis]
    partialTraces = np.cumsum(ringSignals, axis=0)
    residualFraction = np.sum((fullTrace - partialTraces) ** 2, axis=1) / totalEnergy
    failedRings = np.flatnonzero(residualFraction > 1 - fraction)
    reconstructionRadius = (failedRings[-1] + 2) * spacingMm if len(failedRings) else spacingMm

    field64 = leadField.astype(np.float64)
    nodeEnergy = np.maximum(0, np.einsum("cn,dn,cdn->n", field64, field64, energyTerms))
    cumulativeEnergy = np.cumsum(np.bincount(ringIndex, weights=nodeEnergy, minlength=ringCount))
    nodeRadius = (
        np.searchsorted(cumulativeEnergy, fraction * cumulativeEnergy[-1]) + 1
    ) * spacingMm
    nodeFraction = (
        cumulativeEnergy / cumulativeEnergy[-1]
        if cumulativeEnergy[-1] > 0
        else np.full(ringCount, np.nan)
    )
    radii = (np.arange(ringCount) + 1) * spacingMm
    curves = (radii, 1 - residualFraction, nodeFraction)
    return float(reconstructionRadius), float(nodeRadius), fullTrace, curves
