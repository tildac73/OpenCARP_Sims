"""Shared_Pipeline unit tests (IGB I/O, recordings, filters, ground truth).

Ported 28 Oct 2026 from the retired tests/test_pipeline.py, which tested the pre-rework
carputils pipeline (3D_Code/run.py, make_electrodes.py) removed in commit 1a71aa5.
Only behaviour that still exists in the current code is kept; checks are unchanged in intent.
Run: python3 -m unittest tests.test_shared_pipeline -v
"""
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "Shared_Pipeline"))
import egm  # noqa: E402
import ground_truth  # noqa: E402
import igb_io  # noqa: E402


def writeIgb(path, values, start=0.0, step=1.0, dtype="<f4", trailer=b""):
    values = np.asarray(values, dtype=dtype)
    frames, nodes = values.shape
    kind = "float" if values.dtype.itemsize == 4 else "double"
    endian = "big" if dtype.startswith(">") else "little"
    header = f"x:{nodes} y:1 z:1 t:{frames} type:{kind} systeme:{endian}_endian org_t:{start} inc_t:{step} "
    with open(path, "wb") as stream:
        stream.write(header.encode().ljust(1023, b" ") + b"\x0c")
        stream.write(values.tobytes())
        stream.write(trailer)


class IgbReading(unittest.TestCase):
    def testOffsetEndianAndTruncation(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "vm.igb"
            values = np.arange(24).reshape(4, 6)
            for dtype in ("<f4", ">f4", ">f8"):
                writeIgb(path, values, start=1950, dtype=dtype, trailer=b"\x0c")
                np.testing.assert_array_equal(igb_io.read(path), values)
                np.testing.assert_array_equal(igb_io.timeAxis(path), [1950, 1951, 1952, 1953])
            path.write_bytes(path.read_bytes()[:-10])
            with self.assertRaisesRegex(ValueError, "incomplete"):
                igb_io.read(path)

    def testFrameSelection(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "vm.igb"
            values = np.arange(50, dtype=float).reshape(10, 5)
            writeIgb(path, values, step=0.2)
            np.testing.assert_array_equal(igb_io.read(path, slice(3, 6)), values[3:6])
            np.testing.assert_allclose(igb_io.timeAxis(path)[:3], [0, 0.2, 0.4])


class Recordings(unittest.TestCase):
    def testSavedWeightsReconstructEveryRecording(self):
        rng = np.random.default_rng(123)
        unipolar = rng.normal(size=(2, 16, 150))
        differential = rng.normal(size=(2, 16, 150))
        recordings = egm.makeRecordings(unipolar, differential, angleStepDeg=2)
        self.assertEqual(len(recordings), 16 + 9 + 36)
        for record in recordings:
            source = unipolar if record.kind == "uni" else differential
            np.testing.assert_allclose(record.traces, np.einsum("pe,pet->pt", record.weights, source), atol=1e-12)

    def testFilterRejectsInvalidBands(self):
        for band in ([30, 500], [0, 100], [200, 100]):
            with self.assertRaises(ValueError):
                egm.makeFilter(band, 1000, 2)
        self.assertIsNotNone(egm.makeFilter([30, 450], 5000, 2))
        self.assertIsNone(egm.makeFilter(None, 5000, 2))


class GroundTruth(unittest.TestCase):
    def testQuiescentTissueHasNoDefinedFieldOfView(self):
        nodes = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]], float)
        matrix = np.concatenate([np.ones((1, 4, 1)), nodes[None]], axis=2)
        geometry = ground_truth.ElementGeometry(
            nodePositions=nodes, elementNodes=np.array([[0, 1, 2, 3]]), centres=nodes.mean(0, keepdims=True),
            gradients=np.linalg.inv(matrix)[:, 1:, :], volumes=np.abs(np.linalg.det(matrix)) / 6,
            conductivity=np.array([[0.3, 0.1, 0.1]]), extracellularConductivity=np.array([[0.6, 0.2, 0.2]]))
        samples = np.array([[[0.2, 0.2, 3.0]], [[0.8, 0.2, 3.0]]])   # two electrodes, one sample each
        weights = np.array([1.0, -1.0])
        voltage = np.full((4, 20), -80.0)                              # no gradient, no source
        _, radius, _, _ = ground_truth.reconstructRecording(geometry, voltage, samples, weights,
                                                            np.array([0.5, 0.2, 3.0]), 0.5)
        self.assertTrue(np.isnan(radius))


if __name__ == "__main__":
    unittest.main()
