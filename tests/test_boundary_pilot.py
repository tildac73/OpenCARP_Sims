"""Boundary software tests. Synthetic waves are test fixtures, never CRN evidence."""
import copy
import importlib.util
import itertools
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
import pandas as pd

projectFolder = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(projectFolder / '3D_Code'))
sys.path.insert(0, str(projectFolder / 'Shared_Pipeline'))
import boundary_setup as setup
import boundary_geometry as geometry
import boundary_checks as checks
import boundary_analysis as analysis
import boundary_figures as figures
import ground_truth as groundTruth


def testConfig():
    config = setup.loadConfig(projectFolder / '3D_Code/configs/boundary_pilot.json')
    config.update(slabSizeMm=[12., 16., 2.], meshSpacingMm=1.)
    config['pacing'].update(beatCount=3, cycleLengthMs=100., stripWidthMm=1.)
    config['electrodes'].update(gridOffsetsMm=[-1.5, -.5, .5, 1.5], centreYmm=[-4., 0., 4.], gapsMm=[.5, 1.])
    config['analysis'].update(beatWindowMs=[0., 80.], filterHistoryMs=20., groundTruthCentreYmm=[0.],
                              referenceGapMm=.5, noiseRepeats=1, farRegionMm=3.)
    config['acceptance'].update(regionXmm=[-2., 2.], viableRegionYmm=[2., 5.])
    return config


def writeTestMesh(meshBase, config):
    """Six tetrahedra per Cartesian cube; used only to replace mesher in software tests."""
    axes = [np.linspace(-size/2, size/2, round(size/config['meshSpacingMm']) + 1) for size in config['slabSizeMm']]
    x, y, z = np.meshgrid(*axes, indexing='ij')
    positions = np.column_stack([x.ravel(), y.ravel(), z.ravel()])
    dimensions = x.shape
    elements = []
    for i, j, k in itertools.product(*(range(count-1) for count in dimensions)):
        base = np.array([i, j, k])
        for ordering in itertools.permutations(range(3)):
            corners = [base.copy()]
            for direction in ordering:
                corner = corners[-1].copy()
                corner[direction] += 1
                corners.append(corner)
            ids = [np.ravel_multi_index(tuple(corner), dimensions) for corner in corners]
            elements.append(ids)
    meshBase = Path(meshBase)
    meshBase.parent.mkdir(parents=True, exist_ok=True)
    np.savetxt(str(meshBase)+'.pts', positions*1000, header=str(len(positions)), comments='')
    with open(str(meshBase)+'.elem', 'w') as stream:
        stream.write(f'{len(elements)}\n')
        for ids in elements:
            stream.write('Tt ' + ' '.join(map(str, ids)) + ' 1\n')
    with open(str(meshBase)+'.lon', 'w') as stream:
        stream.write('1\n')
        np.savetxt(stream, np.tile([1., 0., 0.], (len(elements), 1)))


def writeIgb(path, values):
    values = np.asarray(values, dtype='<f4')
    frames, nodes = values.shape
    header = f'x:{nodes} y:1 z:1 t:{frames} type:float systeme:little.endian org_t:0 inc_t:1 '
    with open(path, 'wb') as stream:
        stream.write(header.encode().ljust(1023, b' ') + b'\x0c')
        stream.write(values.tobytes())


def prepareFixture(runFolder, config):
    def fakeMesher(command, logPath):
        writeTestMesh(command[-1], config)
        Path(logPath).write_text('SYNTHETIC TEST FIXTURE, not an openCARP run\n')
    with patch.object(setup, 'findExecutable', return_value='test-mesher'), patch.object(setup, 'runCommand', side_effect=fakeMesher):
        setup.prepareExperiment(runFolder, config)


def addSyntheticRecordings(runFolder, config):
    """Prescribed travelling voltage and matching source projection, with known negative scale."""
    runFolder = Path(runFolder)
    layout = dict(np.load(runFolder / 'electrodes/layout.npz'))
    timeMs = np.arange(config['pacing']['beatCount'] * config['pacing']['cycleLengthMs'] + 1)
    for condition in ['intact', 'lesion']:
        folder = runFolder / condition
        folder.mkdir(exist_ok=True)
        meshBase = runFolder / 'mesh' / condition
        positions = setup.readNodePositions(meshBase)
        elapsed = timeMs[:, None] % config['pacing']['cycleLengthMs'] - (10 + 1.5 * (positions[:, 0] + 6))
        rise = np.clip(elapsed / 2, 0, 1)
        fall = np.clip((elapsed - 22) / 8, 0, 1)
        voltage = -80 + 100 * (rise - fall)
        if condition == 'lesion':
            voltage[:, positions[:, 1] < 0] = -80
        writeIgb(folder / 'vm.igb', voltage)
        (folder / 'meshname.txt').write_text(str(meshBase))
        (folder / 'parameters.par').write_text((runFolder / f'{condition}.par').read_text())
        sourceDensity, nodePositions, _ = groundTruth.sourceDensity(folder)
        projections = []
        for point in layout['recoveryPoints']:
            field = groundTruth.pointLeadField(point, nodePositions)
            projections.append(-.07 * np.einsum('cn,cnt->t', field, sourceDensity))
        writeIgb(folder / 'phie_recovery.igb', np.array(projections).T)
        setup.buildEGMs(folder, runFolder, config)
        setup.saveJson(folder / 'meta.json', dict(complete=True, syntheticTestFixture=True))


class BoundaryGeometryTests(unittest.TestCase):
    def testStraightAndCircleDistance(self):
        points = np.array([[3., 4., 0], [0, -2, 0]])
        np.testing.assert_allclose(geometry.signedDistance(points, {'shape':'straight', 'positionMm':0}), [-4, 2])
        np.testing.assert_allclose(geometry.signedDistance(points, {'shape':'circle', 'centreMm':[0,0], 'radiusMm':2.5}), [-2.5,.5])

    def testSelectedPairUsesItsOwnMidpoint(self):
        config = testConfig()
        layout = geometry.makeElectrodeLayout(config)
        positions = layout['centres'][0]
        weights = np.zeros(16)
        weights[0], weights[1] = -1, 1
        result = geometry.recordingGeometry(positions, layout['samples'][0], 'bi', 'E0-E1|E4', weights, config['boundary'], 'lesion')
        expected = (positions[0] + positions[1])/2
        self.assertAlmostEqual(result['anchorYmm'], expected[1])
        self.assertNotEqual(result['selectedPairDistanceInsideMm'], result['cliqueDistanceInsideMm'])
        self.assertEqual(result['selectedElectrodeIds'], '0,1')

    def testSurfaceAveragingAndGatherOrder(self):
        layout = geometry.makeElectrodeLayout(testConfig())
        np.testing.assert_allclose(layout['recoveryPoints'][layout['recoveryIndex']], layout['samples'])
        np.testing.assert_allclose(layout['samples'].mean(axis=2), layout['centres'])
        self.assertEqual(layout['samples'].shape, (6,16,8,3))

    def testRejectInvalidWindowAndNyquist(self):
        config = testConfig()
        config['analysis']['beatWindowMs'] = [0,101]
        with self.assertRaises(ValueError): setup.validateConfig(config)
        config = testConfig()
        config['analysis']['differentialBandHz'][1] = 500
        with self.assertRaises(ValueError): setup.validateConfig(config)

    def testSeedIndependentOfCallOrder(self):
        seed = analysis.stableSeed(1701, 'lesion', 2)
        analysis.stableSeed(1701, 'intact', 3)
        self.assertEqual(seed, analysis.stableSeed(1701, 'lesion', 2))
        self.assertNotEqual(seed, analysis.stableSeed(1701, 'intact', 2))

    def testPlaneDirectionAndSpeed(self):
        positions = np.array(list(itertools.product([-1.,0.,1.], repeat=3)))
        result = checks.fitActivationPlane(positions, 20 + positions[:,0]/.5)
        self.assertAlmostEqual(result['speedMmPerMs'], .5)
        self.assertAlmostEqual(result['directionErrorDeg'], 0)
        self.assertLess(result['planeResidual95Ms'], 1e-10)

    def testFlatTraceDoesNotValidate(self):
        result = analysis.waveformAgreement(np.ones(20), np.ones(20))
        self.assertTrue(np.isnan(result['signedCorrelation']))
        times, count = checks.activationTimes(np.arange(20), np.full((20,2), -80), -20)
        self.assertTrue(np.isnan(times).all())
        self.assertTrue((count == 0).all())


class BoundaryIntegrationTests(unittest.TestCase):
    def testMatchedInputsAndStaleResume(self):
        with tempfile.TemporaryDirectory() as temporary:
            runFolder = Path(temporary) / 'pilot'
            config = testConfig()
            prepareFixture(runFolder, config)
            np.testing.assert_array_equal(setup.readNodePositions(runFolder/'mesh/intact'), setup.readNodePositions(runFolder/'mesh/lesion'))
            np.testing.assert_array_equal(setup.readElements(runFolder/'mesh/intact'), setup.readElements(runFolder/'mesh/lesion'))
            intactParameters = (runFolder/'intact.par').read_text()
            self.assertIn('num_gregions = 1', intactParameters)
            self.assertNotIn('gregion[1]', intactParameters)
            self.assertIn('stim[0].elec.vtx_file', intactParameters)
            stimulus = np.loadtxt(runFolder/'lesionStimulus.vtx', skiprows=2, dtype=int)
            positions = setup.readNodePositions(runFolder/'mesh/lesion')
            self.assertTrue((positions[stimulus,1] > 0).all())
            setup.prepareExperiment(runFolder, config)
            config['timeStepUs'] = 10
            with self.assertRaisesRegex(ValueError, 'Simulation setting timeStepUs changed'):
                setup.prepareExperiment(runFolder, config)

    def testEndToEndSyntheticPipeline(self):
        with tempfile.TemporaryDirectory() as temporary:
            runFolder = Path(temporary) / 'pilot'
            config = testConfig()
            prepareFixture(runFolder, config)
            addSyntheticRecordings(runFolder, config)
            self.assertTrue(checks.checkExperiment(runFolder, config))
            table, _ = analysis.analyseExperiment(runFolder, config, True)
            self.assertEqual(len(table), 2*2*6*61)
            self.assertEqual(table.recordingId.nunique(), len(table))
            computed = table[table.groundTruthComputed]
            self.assertTrue((computed.noiseRepeat == 0).all())
            self.assertTrue((computed.signedCorrelation.dropna() < -.99).all())
            self.assertGreater(computed.fovEligible.sum(), 0)
            self.assertTrue(np.allclose(computed.fittedScale.dropna(), -.07, rtol=1e-3))
            archive = np.load(runFolder/'results/intactTraces.npz')
            np.testing.assert_allclose(archive['measuredMv'], archive['cleanAtSelectedWeightsMv'] + archive['noiseAtSelectedWeightsMv'], atol=1e-6)
            source = np.load(runFolder/'intact/egm.npz')
            rawSlice, measureSlice = analysis.measurementSlices(source['t_ms'], config)
            gain = json.loads((runFolder/'results/calibration.json').read_text())['gain']
            clean = gain * source['uni'][...,rawSlice]
            differential = analysis.filterElectrodes(clean,1000,config)[...,measureSlice]
            for row in table[(table.condition=='intact') & (table.kind=='bi')].iloc[::17].to_dict('records'):
                weights = np.array([row[f'weightE{i}'] for i in range(16)])
                expected = weights @ differential[row['placement']]
                np.testing.assert_allclose(expected, archive['cleanAtSelectedWeightsMv'][row['traceIndex']], atol=1e-6)
            figures.makeFigures(runFolder, config)
            self.assertEqual(len(list((runFolder/'figures').glob('*.png'))), 5)

    def testNoCaptureCannotPass(self):
        with tempfile.TemporaryDirectory() as temporary:
            runFolder = Path(temporary) / 'pilot'
            config = testConfig()
            prepareFixture(runFolder, config)
            for condition in ['intact','lesion']:
                folder = runFolder/condition
                folder.mkdir()
                nodes = len(setup.readNodePositions(runFolder/'mesh'/condition))
                writeIgb(folder/'vm.igb',np.full((301,nodes),-80.))
            self.assertFalse(checks.checkExperiment(runFolder, config))
            result = json.loads((runFolder/'physiologyStatus.json').read_text())
            self.assertFalse(result['passed'])


if __name__ == '__main__':
    unittest.main()
