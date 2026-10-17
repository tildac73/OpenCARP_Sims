"""Regression tests; synthetic fixtures verify plumbing, not reentry physiology.

Run: python -m unittest discover -s tests -v
"""

import importlib.util
import itertools
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "Shared_Pipeline"))
import egm
import ground_truth as gt
import igb_io


def write_igb(path, values, start=0, step=1, dtype="<f4", trailer=b""):
    values = np.asarray(values, dtype=dtype)
    count, nodes = values.shape
    kind = "float" if values.dtype.itemsize == 4 else "double"
    endian = "big" if dtype.startswith(">") else "little"
    header = f"x:{nodes} y:1 z:1 t:{count} type:{kind} systeme:{endian}.endian org_t:{start} inc_t:{step} "
    with open(path, "wb") as stream:
        stream.write(header.encode().ljust(1023, b" ") + b"\x0c")
        stream.write(values.tobytes())
        stream.write(trailer)


def import_runner():
    """Stub only the unavailable carputils interface; exercise the real runner code."""
    tools = types.SimpleNamespace(
        carpexample=lambda *a, **k: lambda fn: fn, carp_cmd=lambda path: ["openCARP", "+F", path]
    )
    package = types.ModuleType("carputils")
    package.tools = tools
    package.settings = types.SimpleNamespace(platform=types.SimpleNamespace(BATCH=False))
    carpio = types.ModuleType("carputils.carpio")
    carpio.txt = types.SimpleNamespace()
    with patch.dict(sys.modules, {"carputils": package, "carputils.carpio": carpio}):
        spec = importlib.util.spec_from_file_location("slab_runner", ROOT / "3D_Code/run.py")
        runner = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(runner)
    return runner


class IOAndSignals(unittest.TestCase):
    def test_binary_offset_endian_and_truncation(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "vm.igb"
            values = np.arange(24).reshape(4, 6)
            for dtype in ("<f4", ">f4", ">f8"):
                write_igb(path, values, 1950, dtype=dtype, trailer=b"\x0c")
                np.testing.assert_array_equal(igb_io.read(path), values)
                np.testing.assert_array_equal(igb_io.time_axis(path), [1950, 1951, 1952, 1953])
            path.write_bytes(path.read_bytes()[:-10])
            with self.assertRaisesRegex(ValueError, "incomplete"):
                igb_io.read(path)

    def test_exact_restart_not_last_filename(self):
        runner = import_runner()
        with tempfile.TemporaryDirectory() as tmp:
            for time in ("99.0", "3500.0", "4000.0", "3500.5"):
                (Path(tmp) / f"state.{time}.roe").touch()
            name, time = runner.checkpoint(tmp, 3500)
            self.assertTrue(name.endswith("state.3500.0"))
            self.assertEqual(time, 3500)
            with self.assertRaises(FileNotFoundError):
                runner.checkpoint(tmp, 1950)

    def test_saved_weights_reconstruct_every_recording(self):
        rng = np.random.default_rng(123)
        signal = rng.normal(size=(2, 16, 150))
        xy = rng.normal(size=(2, 16, 3))
        cfg = egm.deep_update(egm.DEFAULTS, {})
        recordings = egm.recordings(signal, signal, xy, cfg, with_weights=True)
        self.assertEqual(len(recordings), 61)
        for _, _, _, trace, weights in recordings:
            np.testing.assert_allclose(trace, np.einsum("pe,pet->pt", weights, signal), atol=1e-12)

    def test_invalid_time_and_filter_windows(self):
        with self.assertRaises(ValueError):
            egm.make_filter([30, 500], 1000, 2)
        with self.assertRaises(ValueError):
            egm.woi_slice(np.arange(10), [30, 50])
        with self.assertRaises(ValueError):
            egm.analysis_slices(np.array([0, 1, 1]), egm.DEFAULTS)
        raw, measure = egm.analysis_slices(
            np.arange(600), dict(egm.DEFAULTS, analysis_window_last_ms=400)
        )
        self.assertEqual((raw.start, raw.stop, measure.start, measure.stop), (100, 600, 100, 500))

    def test_recovery_export_shape_time_and_electrode_order(self):
        runner = import_runner()
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            subprocess.run(
                [sys.executable, str(ROOT / "3D_Code/make_electrodes.py")],
                cwd=tmp,
                check=True,
                capture_output=True,
            )
            sites = directory / "electrodes/egm_points"
            values = np.arange(8464)[None, :] + np.arange(3)[:, None]
            write_igb(directory / "phie_recovery.igb", values, start=1795)
            write_igb(directory / "vm.igb", np.zeros((3, 5)), start=1795)
            runner.buildEGMs(tmp, 1795, str(sites))
            with np.load(directory / "egm.npz") as data:
                self.assertEqual(data["uni"].shape, (242, 16, 3))
                self.assertEqual(data["uni_sample_xyz"].shape, (242, 16, 8, 3))
                np.testing.assert_array_equal(data["t_ms"], [1795, 1796, 1797])
                self.assertEqual(data["uni"][0, 0, 0], 3.5)
                self.assertEqual(data["uni"][0, 1, 0], 4 * 23 * 8 + 3.5)

    def test_four_stage_dispatch_and_failure_stop(self):
        runner = import_runner()
        with tempfile.TemporaryDirectory() as tmp:
            calls = []

            class Job:
                ID = tmp

                def carp(self, command):
                    calls.append(command)
                    folder = Path(command[command.index("-simID") + 1])
                    folder.mkdir()
                    for name in ("vm.igb", "phie_recovery.igb", "parameters.par"):
                        (folder / name).touch()
                    times = {"s1_pace": 1795, "s2_block": 1950, "s3_reentry": 3500}
                    if folder.name in times:
                        (folder / f"state.{times[folder.name]}.0.roe").touch()

            ends = {
                "s1_pace": 1795,
                "s2_block": 1950,
                "s3_reentry": 4000,
                "s4_ablate_terminate": 6000,
            }
            args = types.SimpleNamespace(
                t_lift=1950, tune_cv=None, gi=None, ge=None, scenario="protocol", visualize=False
            )
            with patch.object(runner, "buildMesh", return_value=("pre", "scar")), patch.object(
                runner, "writeRecoveryPoints", return_value="electrodes"
            ), patch.object(runner, "buildEGMs"), patch.object(
                runner.igb_io,
                "time_axis",
                side_effect=lambda p: np.append(
                    np.arange(*runner.stageTimes(Path(p).parent.name, 1950)),
                    ends[Path(p).parent.name],
                ),
            ):
                runner.run(args, Job())
                self.assertEqual([Path(c[c.index("-simID") + 1]).name for c in calls], list(ends))
                self.assertTrue(
                    calls[-1][calls[-1].index("-start_statef") + 1].endswith("state.3500.0")
                )
                self.assertEqual(calls[-1][calls[-1].index("-meshname") + 1], "scar")
                with self.assertRaises(FileExistsError):
                    runner.run(args, Job())

    def test_resume_reuses_complete_stages_and_reruns_the_rest(self):
        runner = import_runner()
        ends = {"s1_pace": 1795, "s2_block": 1950, "s3_reentry": 4000, "s4_ablate_terminate": 6000}
        with tempfile.TemporaryDirectory() as tmp:
            calls = []

            class Job:
                ID = tmp

                def carp(self, command):
                    calls.append(Path(command[command.index("-simID") + 1]).name)
                    folder = Path(command[command.index("-simID") + 1])
                    folder.mkdir()
                    start, end = runner.stageTimes(folder.name, 1950)
                    # restarted stages get a header without org_t: frames from 0
                    write_igb(folder / "vm.igb", np.zeros((int(end - start) + 1, 2)), start=0 if start else 0)
                    for name in ("phie_recovery.igb", "parameters.par"):
                        (folder / name).write_text("x")
                    save = {"s1_pace": 1795, "s2_block": 1950, "s3_reentry": 3500}.get(folder.name)
                    if save:
                        (folder / f"state.{save}.0.roe").touch()

            def fake_egms(simdir, t0, points, t_ms):
                self.assertEqual(t_ms[0], t0)  # restart time recovered from the frame count
                (Path(simdir) / "egm.npz").write_text("x")

            args = types.SimpleNamespace(t_lift=1950, tune_cv=None, gi=None, ge=None,
                                         scenario="protocol", visualize=False, resume=True)
            with patch.object(runner, "buildMesh", return_value=("pre", "scar")), patch.object(
                runner, "writeRecoveryPoints", return_value="electrodes"
            ), patch.object(runner, "buildEGMs", side_effect=fake_egms):
                runner.run(args, Job())
                self.assertEqual(calls, list(ends))
                # A crash during s3 leaves it incomplete: s3 and the s4 branch rerun, s1/s2 are reused.
                (Path(tmp) / "s3_reentry/meta.json").unlink()
                calls.clear()
                runner.run(args, Job())
                self.assertEqual(calls, ["s3_reentry", "s4_ablate_terminate"])
                moved = sorted(p.name.split(".")[1].split("-")[0] for p in Path(tmp).iterdir() if "." in p.name)
                self.assertEqual(moved, ["incomplete", "stale"])
                calls.clear()
                runner.run(args, Job())
                self.assertEqual(calls, [])

    def test_stage_error_stops_the_chain(self):
        runner = import_runner()
        with tempfile.TemporaryDirectory() as tmp:
            calls = []

            class FailedJob:
                ID = tmp

                def carp(self, command):
                    calls.append(command)
                    raise RuntimeError("synthetic simulator failure")

            args = types.SimpleNamespace(
                t_lift=1950, tune_cv=None, gi=None, ge=None, scenario="protocol", visualize=False
            )
            with patch.object(runner, "buildMesh", return_value=("pre", "scar")), patch.object(
                runner, "writeRecoveryPoints", return_value="electrodes"
            ):
                with self.assertRaisesRegex(RuntimeError, "simulator failure"):
                    runner.run(args, FailedJob())
            self.assertEqual(len(calls), 1)

    def test_quiescence_has_no_defined_field_of_view(self):
        source = np.zeros((3, 4, 10))
        fr, fn, signal = gt.fov_metrics(
            np.ones((3, 4)), source, gt.energy_terms(source), np.zeros(4), np.zeros((4, 3)), 1, 0.8
        )
        self.assertTrue(np.isnan(fr) and np.isnan(fn))
        np.testing.assert_array_equal(signal, 0)


def make_fixture(root):
    """Small structured mesh with analytic travelling Vm; no openCARP dependency."""
    shutil.copytree(
        ROOT / "3D_Code", root / "3D_Code", ignore=shutil.ignore_patterns("runs", "__pycache__")
    )
    shutil.copytree(
        ROOT / "Shared_Pipeline",
        root / "Shared_Pipeline",
        ignore=shutil.ignore_patterns("__pycache__"),
    )
    job = root / "3D_Code/runs/reentry_tlift1950_np1"
    mesh = job / "mesh/slab"
    mesh.parent.mkdir(parents=True)
    xs = np.linspace(-3, 3, 5)
    zs = np.linspace(-1.5, 1.5, 3)
    xyz = np.array([(x, y, z) for z in zs for y in xs for x in xs])
    np.savetxt(str(mesh) + ".pts", xyz * 1000, header=str(len(xyz)), comments="")

    def index(x, y, z):
        return (z * 5 + y) * 5 + x

    elements = []
    for z in range(2):
        for y in range(4):
            for x in range(4):
                for permutation in itertools.permutations(range(3)):
                    vertex = [x, y, z]
                    nodes = [index(*vertex)]
                    for axis in permutation:
                        vertex[axis] += 1
                        nodes.append(index(*vertex))
                    elements.append(nodes)
    with open(str(mesh) + ".elem", "w") as stream:
        stream.write(f"{len(elements)}\n")
        stream.writelines("Tt " + " ".join(map(str, nodes)) + " 1\n" for nodes in elements)
    np.savetxt(str(mesh) + ".lon", np.tile([1, 0, 0], (len(elements), 1)), header="1", comments="")
    for name, start, stop, state in [
        ("s1_pace", 0, 1795, "pre"),
        ("s2_block", 1795, 1950, "pre"),
        ("s3_reentry", 1950, 4000, "pre"),
        ("s4_ablate_terminate", 3500, 6000, "post"),
    ]:
        folder = job / name
        folder.mkdir()
        t = np.arange(start, stop + 1)
        vm = np.sin(2 * np.pi * (t[:, None] / 70 - xyz[None, :, 0] / 15)) + 0.3 * np.cos(
            2 * np.pi * (t[:, None] / 49 - xyz[None, :, 1] / 15)
        )
        write_igb(folder / "vm.igb", vm, start=start, trailer=b"\x0c")
        (folder / "meshname.txt").write_text(str(mesh))
        (folder / "parameters.par").write_text(
            "num_gregions = 1\ngregion[0].num_IDs = 1\ngregion[0].ID[0] = 1\ngregion[0].g_il = 0.174\ngregion[0].g_it = 0.094\ngregion[0].g_in = 0.094\n"
        )
        (folder / "meta.json").write_text(
            json.dumps(dict(model="slab3d", run=job.name + "/" + name, state=state, stage=name))
        )
        S, source_xyz, components, _, _ = gt.source_density(str(folder))
        sites = np.array(
            [[(x, y, 2 + gap) for y in [-6, -2, 2, 6] for x in [-6, -2, 2, 6]] for gap in [0.25, 1]]
        )
        uni = np.array(
            [
                [
                    np.einsum("cn,cnt->t", gt.grad_Z(site, source_xyz, components), S)
                    for site in grid
                ]
                for grid in sites
            ]
        )
        np.savez_compressed(
            folder / "egm.npz",
            uni=uni,
            uni_xy=sites,
            t_ms=t,
            d_mm=2.5 - np.hypot(sites[..., 0], sites[..., 1]),
            centre_xy_mm=np.zeros((2, 2)),
            p_gap_mm=np.array([0.25, 1]),
        )
    config = json.loads((root / "Shared_Pipeline/configs/slab3d_default.json").read_text())
    config["groundtruth"].update(
        placements_xy_mm_pre=[[0, 0]],
        placements_xy_mm_post=[[0, 0]],
        map_xy_mm_pre=[[0, 0]],
        map_xy_mm_post=[[0, 0]],
    )
    (root / "Shared_Pipeline/configs/slab3d_default.json").write_text(json.dumps(config))
    return job


class ShellIntegration(unittest.TestCase):
    def test_four_stages_through_egms_groundtruth_and_maps(self):
        with tempfile.TemporaryDirectory(prefix="efov integration ") as tmp:
            root = Path(tmp)
            job = make_fixture(root)
            # Old unrelated outputs must not be picked up by the default four-stage run.
            shutil.copytree(job / "s3_reentry", job / "post_s3")
            env = dict(
                os.environ,
                PYTHON=sys.executable,
                SKIP_SIMS="1",
                MAPS="1",
                MPLCONFIGDIR=str(root / "mpl"),
            )
            result = subprocess.run(
                ["bash", str(root / "3D_Code/efov_3d.sh")],
                cwd="/tmp",
                env=env,
                text=True,
                capture_output=True,
            )
            self.assertEqual(result.returncode, 0, result.stdout + "\n" + result.stderr)
            out = next((root / "results").iterdir())
            points = pd.read_csv(out / "points.csv")
            truth = pd.read_csv(out / "groundtruth.csv")
            merged = pd.read_csv(out / "points_with_gt.csv")
            self.assertEqual(len(points), 4 * 2 * 61)
            self.assertEqual(len(truth), len(points))
            self.assertTrue(merged.groundtruth_computed.all())
            self.assertEqual(truth.selection_source.unique().tolist(), ["measured_egm"])
            self.assertGreater(truth.r_vs_recovery.dropna().abs().min(), 0.999)
            self.assertTrue(np.isfinite(truth.fov80_node_mm).all())
            self.assertEqual(len(list((out / "maps").glob("*_maps.png"))), 4)
            self.assertEqual(len(list((out / "maps").glob("*_maps_depth.png"))), 4)
            self.assertEqual(len(list((out / "maps").glob("*.vtk"))), 16)
            figures = out / "figures"
            self.assertTrue((figures / "FIGURES.md").is_file())
            self.assertGreaterEqual(len(list(figures.glob("*.png"))), 15, result.stdout)
            self.assertEqual(len(list(figures.glob("*.png"))), len(list(figures.glob("*.pdf"))))
            # Every geometry-derived weighted signal matches the selected EGM trace.
            cfg = egm.deep_update(
                egm.DEFAULTS, json.loads(next((job / "configs").iterdir()).read_text())
            )
            cfg["out_dir"] = str(root / "results")
            tissue = gt.load_tissue(str(job / "s3_reentry"), cfg, cfg["groundtruth"])
            sites, samples = gt.electrode_sites(tissue["z"], 0, tissue["xyz"][:, 2].max())
            measured = tissue["measured"][tissue["measured"].placement == 0]
            fields = gt.recording_leadfields(
                sites, cfg, tissue["Sd"], tissue["comps"], tissue["xyz"], samples, measured
            )
            raw_uni = tissue["z"]["uni"][0, :, tissue["egm"]]
            differential = egm.apply_filter(egm.make_filter([30, 450], 1000, 2), raw_uni, False)[
                ..., tissue["w"]
            ]
            for kind, label, site, field in fields:
                if kind == "uni":
                    continue
                row = measured[measured.label == label].iloc[0]
                weights = row[[f"weight_E{e}" for e in range(16)]].to_numpy(float)
                rebuilt = np.einsum("cn,cnt->t", field, tissue["Sd"])
                np.testing.assert_allclose(rebuilt, weights @ differential, atol=2e-6, rtol=2e-4)
            # A standalone ablation analysis obtains gain from existing s3 data.
            post_env = dict(env, SCENARIO="ablate", MAPS="0")
            post = subprocess.run(
                ["bash", str(root / "3D_Code/efov_3d.sh")],
                env=post_env,
                text=True,
                capture_output=True,
            )
            self.assertEqual(post.returncode, 0, post.stdout + post.stderr)
            post_out = next(
                path for path in (root / "results").iterdir() if path.name.endswith("__ablate")
            )
            self.assertEqual(len(pd.read_csv(post_out / "points.csv")), 2 * 61)
            # A ground-truth failure must also make the shell fail.
            (job / "s4_ablate_terminate/meshname.txt").write_text(str(root / "missing_mesh"))
            failed_gt = subprocess.run(
                ["bash", str(root / "3D_Code/efov_3d.sh")],
                env=post_env,
                text=True,
                capture_output=True,
            )
            self.assertNotEqual(failed_gt.returncode, 0)
            self.assertNotIn("Completed successfully", failed_gt.stdout)
            # Missing stages are failures, without a success message.
            (job / "s2_block/vm.igb").unlink()
            result = subprocess.run(
                ["bash", str(root / "3D_Code/efov_3d.sh")], env=env, text=True, capture_output=True
            )
            self.assertNotEqual(result.returncode, 0)
            self.assertNotIn("Completed successfully", result.stdout)
            self.assertIn("Missing output", result.stderr)


if __name__ == "__main__":
    unittest.main()
