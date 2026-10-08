"""Verify portable entry points and the preserved original protocol tests."""
import hashlib
import json
import subprocess
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
FUSION = ROOT / "fusion"


class FusionReleaseTests(unittest.TestCase):
    def run_python(self, arguments, cwd=ROOT):
        result = subprocess.run([sys.executable, "-X", "utf8", "-B", *arguments], cwd=cwd,
                                text=True, encoding="utf-8", capture_output=True,
                                timeout=120)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        return result.stdout

    def test_both_queues_plan_exactly_six_methods_and_three_seeds(self):
        for experiment in ("qm9_mu", "qm8_e1cc2"):
            jobs = json.loads(self.run_python([
                str(FUSION / experiment / "run_queue.py"), "--data", "unused-cache",
                "--runs", "unused-runs", "--output", "unused-queue", "--dry-run"]))
            self.assertEqual(len(jobs), 18)
            self.assertEqual({item["job"]["seed"] for item in jobs}, {42, 2025, 3407})
            self.assertEqual({item["job"]["fusion"] for item in jobs},
                             {"concat", "mean", "weighted", "attention",
                              "gate_no_interaction", "gate"})
            for item in jobs:
                command = item["command"]
                self.assertTrue(Path(command[2]).is_file())
                self.assertEqual(command[command.index("--mode") + 1], "train")
                self.assertEqual(command[command.index("--split-seed") + 1], "20260927")

    def test_one_shared_historical_model(self):
        self.assertEqual(list(FUSION.rglob("Model.py")), [FUSION / "Model.py"])
        digest = hashlib.sha256((FUSION / "Model.py").read_bytes()).hexdigest()
        self.assertEqual(digest, "86b34ebe6cbef4f7a24b06db6e179e3062f3bf4e30031b09f3b3ea548e048b78")

    def test_release_manifest_hashes(self):
        manifest = json.loads((FUSION / "SOURCE_PROVENANCE.json").read_text(encoding="utf-8"))
        for item in manifest["files"]:
            self.assertEqual(hashlib.sha256((FUSION / item["release_file"]).read_bytes()).hexdigest(),
                             item["release_sha256"], item["release_file"])

    def test_preserved_qm9_integrity_tests(self):
        self.run_python(["-m", "unittest", "test_queue", "test_train_protocol",
                         "test_summarize_fusion"], cwd=FUSION / "qm9_mu")

    def test_preserved_qm8_group_integrity_tests(self):
        self.run_python(["-m", "unittest", "test_group_protocol"], cwd=FUSION / "qm8_e1cc2")


if __name__ == "__main__":
    unittest.main()
