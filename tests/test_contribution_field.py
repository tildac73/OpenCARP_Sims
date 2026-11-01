"""contribution_field: shares sum to one and reproduce the forward solve (both source tensors)."""
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np

REPOSITORY = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY / "Shared_Pipeline"))

try:
    import scipy  # noqa: F401
    HAVE_SCIPY = True
except ImportError:
    HAVE_SCIPY = False


def writeBox(base, tissueTop=0.0):
    import fem_extracellular as fem
    positions, elements = fem.structuredBox([-2.0, -2.0, -1.0], [2.0, 2.0, 2.0], 0.5)
    centres = positions[elements].mean(axis=1)
    tags = np.where(centres[:, 2] < tissueTop, 1, 0)
    tags[(tags == 1) & (centres[:, 1] < -1.0)] = 5
    fibres = np.where(tags[:, None] == 0, 0.0, np.array([1.0, 0.0, 0.0]))
    with open(f"{base}.pts", "w") as stream:
        stream.write(f"{len(positions)}\n"); np.savetxt(stream, positions * 1000)
    with open(f"{base}.elem", "w") as stream:
        stream.write(f"{len(elements)}\n")
        for row, tag in zip(elements, tags):
            stream.write("Tt " + " ".join(map(str, row)) + f" {tag}\n")
    with open(f"{base}.lon", "w") as stream:
        stream.write("1\n"); np.savetxt(stream, fibres)


@unittest.skipUnless(HAVE_SCIPY, "needs scipy")
class ContributionField(unittest.TestCase):
    def test_shares_reconstruct_the_forward_solution(self):
        import contribution_field as cf
        g = dict(g_il=1.7, g_it=0.3, g_in=0.3, g_el=0.6, g_et=0.4, g_en=0.4)
        with tempfile.TemporaryDirectory() as folder:
            base = Path(folder) / "box"
            writeBox(base)
            for source in ["sigmaM", "sigmaI"]:
                model = cf.BathModel(base, g, 1e-6, 0.74, source, method="lu")
                rng = np.random.default_rng(1)
                tissueVm = rng.normal(size=(len(model.tissueNodes), 12))
                samples = np.array([[[0.1, 0.2, 0.25 + 0.1 * k] for k in range(4)]] * 16)
                weights = np.zeros(16); weights[0], weights[1] = 1.0, -1.0
                samples[1] += [0.5, 0.0, 0.0]
                row = cf.recordingRow(model, samples, weights)
                shares, phi, total = cf.shares(model, row, tissueVm)
                self.assertAlmostEqual(total, 1.0, places=10)
                direct = row @ model.solver.solve(model.fullVm(tissueVm))
                np.testing.assert_allclose(phi, direct, rtol=1e-8, atol=1e-10)
                grid = cf.voxelGrid([4.0, 4.0, 1.0], 0.5)
                self.assertAlmostEqual(float(cf.voxelise(model, shares, grid).sum()), 1.0, places=10)

    def test_source_tensors(self):
        import contribution_field as cf
        src, rest = cf.sourceTensors("sigmaM", [2.0, 1.0, 1.0], [2.0, 1.0, 1.0])
        np.testing.assert_allclose(src, [1.0, 0.5, 0.5]); np.testing.assert_allclose(src + rest, [4.0, 2.0, 2.0])
        with self.assertRaises(ValueError):
            cf.sourceTensors("monodomain", [1, 1, 1], [1, 1, 1])


if __name__ == "__main__":
    unittest.main()
