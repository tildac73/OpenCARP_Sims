import numpy as np

from efov import io


def writeIgb(path, values, step):
    frames, nodes = values.shape
    header = f"x:{nodes} y:1 z:1 t:{frames} type:float systeme:little_endian org_t:0 dim_t:{(frames - 1) * step:g} inc_t:{step:g}"
    with open(path, "wb") as out:
        out.write((header + " " * (1023 - len(header)) + "\x0c").encode("latin-1"))
        out.write(values.astype("<f4").tobytes())


def test_igb_read_times_and_merge(tmp_path):
    a = np.arange(12, dtype=float).reshape(4, 3)              # t = 0..0.6
    b = np.vstack([a[-1], a[-1] + 100, a[-1] + 200])          # restart repeats the last frame
    writeIgb(tmp_path / "a.igb", a, 0.2); writeIgb(tmp_path / "b.igb", b, 0.2)
    np.testing.assert_allclose(io.igbTimes(tmp_path / "a.igb"), [0, 0.2, 0.4, 0.6])
    np.testing.assert_allclose(io.readIgb(tmp_path / "a.igb", slice(1, 3)), a[1:3])
    io.mergeIgb(tmp_path / "a.igb", tmp_path / "b.igb", tmp_path / "m.igb")
    np.testing.assert_allclose(io.readIgb(tmp_path / "m.igb"), np.vstack([a, b[1:]]))
    np.testing.assert_allclose(io.igbTimes(tmp_path / "m.igb"), np.arange(6) * 0.2)


def test_truncated_igb_is_rejected(tmp_path):
    writeIgb(tmp_path / "a.igb", np.zeros((4, 3)), 0.2)
    data = (tmp_path / "a.igb").read_bytes()
    (tmp_path / "a.igb").write_bytes(data[:-4])
    import pytest
    with pytest.raises(ValueError, match="incomplete"):
        io.igbHeader(tmp_path / "a.igb")


def test_activation_merge_and_first_activation(tmp_path):
    np.savetxt(tmp_path / "first.dat", [[0, 1.0], [1, 2.0], [0, 501.0], [1, 900.0]], fmt=["%d", "%.3f"], delimiter="\t")
    np.savetxt(tmp_path / "second.dat", [[1, 899.0], [1, 1002.0], [0, 1001.0]], fmt=["%d", "%.3f"], delimiter="\t")
    io.mergeActivations(tmp_path / "first.dat", tmp_path / "second.dat", tmp_path / "merged.dat", restartMs=850.0)
    nodes, times = io.readActivations(tmp_path / "merged.dat")
    assert sorted(times.tolist()) == [1.0, 2.0, 501.0, 899.0, 1001.0, 1002.0]   # 900 (first run, after restart) dropped
    np.testing.assert_allclose(io.firstActivation(nodes, times, 3, 1000, 1500), [1001.0, 1002.0, np.nan])


def test_mesh_round_trip_and_vtk(tmp_path):
    from helpers import boxWithBath
    positions, elements, tags = boxWithBath([2, 2], 1.0, 0.5, 0.5)
    fibres = np.where(tags[:, None] == 0, 0.0, [1.0, 0.0, 0.0])
    base = io.writeMesh(tmp_path / "mesh" / "slab", positions, elements, tags, fibres)
    p, e, t, f = io.readMesh(base)
    np.testing.assert_allclose(p, positions, atol=1e-9)
    assert np.array_equal(e, elements) and np.array_equal(t, tags)
    np.testing.assert_allclose(f, fibres)
    path = io.writeVtkCells(tmp_path / "field.vtk", positions, elements, {"share": np.arange(len(elements), dtype=float)})
    head = path.read_bytes()[:200]
    assert b"UNSTRUCTURED_GRID" in head and b"BINARY" in head
