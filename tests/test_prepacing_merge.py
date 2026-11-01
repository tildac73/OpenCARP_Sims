"""slab.py prepacing: merged Vm and activation files equal one continuous record (needs carputils)."""
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np

REPOSITORY = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY / "3D_Code" / "production"))
sys.path.insert(0, str(REPOSITORY / "Shared_Pipeline"))
try:
    import slab
    HAVE_CARPUTILS = True
except ImportError:
    HAVE_CARPUTILS = False


def writeIgb(path, data, step):
    header = f"x:{data.shape[1]} y:1 z:1 t:{data.shape[0]} type:float systeme:little_endian dim_t:{(data.shape[0]-1)*step:g} inc_t:{step:g}"
    with open(path, "wb") as stream:
        stream.write((header + " " * (1023 - len(header)) + "\x0c").encode("latin-1"))
        stream.write(np.asarray(data, "<f4").tobytes())


@unittest.skipUnless(HAVE_CARPUTILS, "slab.py needs carputils")
class PrepacingMerge(unittest.TestCase):
    def test_merge_equals_continuous(self):
        import igb_io
        full = np.random.default_rng(0).normal(size=(11, 5)).astype("f4")
        with tempfile.TemporaryDirectory() as folder:
            folder = Path(folder)
            writeIgb(folder / "a.igb", full[:5], 0.2)        # 0 .. 0.8 ms
            writeIgb(folder / "b.igb", full[4:], 0.2)        # restart at 0.8 ms: first frame repeats
            slab.mergeIgb(folder / "a.igb", folder / "b.igb", folder / "m.igb")
            np.testing.assert_array_equal(igb_io.read(folder / "m.igb"), full)
            np.testing.assert_allclose(igb_io.timeAxis(folder / "m.igb"), np.arange(11) * 0.2)
            np.savetxt(folder / "a.dat", [[1, 0.5], [2, 0.9]]); np.savetxt(folder / "b.dat", [[2, 0.9], [3, 1.4]])
            slab.mergeActivations(folder / "a.dat", folder / "b.dat", folder / "m.dat", 0.8)
            np.testing.assert_allclose(np.loadtxt(folder / "m.dat"), [[1, 0.5], [2, 0.9], [3, 1.4]])


if __name__ == "__main__":
    unittest.main()
