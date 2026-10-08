"""Lightweight source/configuration checks; these tests do not start training."""
import ast
import hashlib
import importlib.util
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


class GeometryReleaseTests(unittest.TestCase):
    def test_twenty_explicit_targets_and_units(self):
        catalog = read(ROOT / "experiments.json")
        self.assertEqual(len(catalog), 20)
        self.assertEqual(sum(v["dataset"] == "QM9" for v in catalog.values()), 8)
        self.assertEqual(sum(v["dataset"] == "QM8" for v in catalog.values()), 12)
        for key, entry in catalog.items():
            with self.subTest(experiment=key):
                tree = ast.parse((ROOT / entry["files"]["train_fusion.py"]).read_text(encoding="utf-8"))
                constants = {n.targets[0].id: ast.literal_eval(n.value) for n in tree.body
                             if isinstance(n, ast.Assign) and isinstance(n.targets[0], ast.Name)
                             and n.targets[0].id in ("DATASET", "TARGET", "UNIT")}
                self.assertEqual(constants, {"DATASET": entry["dataset"], "TARGET": entry["target"], "UNIT": entry["unit"]})
                protocol = read(ROOT / entry["files"]["experiment_protocol.json"])
                self.assertEqual(protocol["target"], entry["target"])
                self.assertEqual(protocol["seeds"], [42, 2025, 3407])

    def test_source_fingerprints_and_compile(self):
        provenance = read(ROOT / "SOURCE_PROVENANCE.json")
        for records in provenance.values():
            for record in records:
                path = ROOT / record["release_path"]
                self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(), record["release_sha256"], str(path))
        for path in ROOT.rglob("*.py"):
            if "_temporary" not in path.parts:
                compile(path.read_text(encoding="utf-8-sig"), str(path), "exec")

    def test_split_disjointness_and_sizes(self):
        expected = {"qm9": (105992, 13251, 13243), "qm8": (17356, 2164, 2169)}
        for dataset, sizes in expected.items():
            split = read(ROOT / "splits" / f"{dataset}_matched.json")
            original = read(ROOT / "splits" / f"{dataset}_original.json")
            all_ids = []
            for key, size in zip(("train", "val", "test"), sizes):
                self.assertEqual(len(split[key]), size)
                retained = set(split[key])
                self.assertEqual([mid for mid in original[key] if mid in retained], split[key])
                all_ids.extend(split[key])
            self.assertEqual(len(all_ids), len(set(all_ids)))

    def test_all_twenty_default_queue_commands(self):
        for key, entry in read(ROOT / "experiments.json").items():
            with self.subTest(experiment=key), tempfile.TemporaryDirectory() as temporary:
                work = Path(temporary)
                for name, source in entry["files"].items():
                    shutil.copyfile(ROOT / source, work / name)
                manifest = {name: hashlib.sha256((work / name).read_bytes()).hexdigest()
                            for name in entry["files"]}
                (work / "release_source_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
                result = subprocess.run([sys.executable, "-B", str(ROOT / "run.py"), "train", "--experiment", key,
                                         "--work-dir", str(work), "--dry-run"], capture_output=True, text=True)
                self.assertEqual(result.returncode, 0, result.stderr)
                jobs = json.loads(result.stdout)
                self.assertEqual(len(jobs), 6)
                self.assertEqual({r["job"]["seed"] for r in jobs}, {42, 2025, 3407})
                self.assertEqual({r["job"]["geometry_condition"] for r in jobs}, {"dft", "mmff"})

    def test_conflicting_existing_units_are_rejected(self):
        from types import SimpleNamespace
        spec = importlib.util.spec_from_file_location("geometry_release_run", ROOT / "run.py")
        runner = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(runner)
        entry = read(ROOT / "experiments.json")["qm9_mu"]
        with tempfile.TemporaryDirectory() as temporary:
            folder = Path(temporary) / "input" / "dft"
            folder.mkdir(parents=True)
            (folder / "cache.npz").write_bytes(b"unused: rejected before cache loading")
            shutil.copyfile(ROOT / entry["split"], folder / "fixed_split_ids.json")
            metadata = read(ROOT / entry["metadata"]["dft"])
            metadata["cache_sha256"] = runner.digest(folder / "cache.npz")
            metadata["target_units"]["dipole_moment"] = "incorrect-unit"
            (folder / "metadata.json").write_text(json.dumps(metadata), encoding="utf-8")
            output = Path(temporary) / "output"
            args = SimpleNamespace(work_dir=output, data=folder.parent, copy_data=False, experiment="qm9_mu")
            with self.assertRaisesRegex(ValueError, "Existing target unit"):
                runner.prepare(args, entry)
            self.assertFalse(output.exists())


if __name__ == "__main__":
    unittest.main()
