"""Small independent, synthetic integrity checks; no ML packages or server."""
import csv
import json
import math
import tempfile
import unittest
from pathlib import Path

import summarize_fusion as summary


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def make_run(root, fusion, seed, error=1.0):
    run = root / fusion / f"seed_{seed}"
    run.mkdir(parents=True)
    write_json(run / "status.json", {"status": "completed"})
    split = {"train": [1, 2, 3], "val": [4, 5], "test": [6, 7]}
    write_json(run / "split_ids.json", split)
    prediction_path = run / "test_predictions.csv"
    with prediction_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["id", "target_original_units", "prediction_original_units"])
        writer.writerow([6, 0, error])
        writer.writerow([7, 2, 2 + error])
    (run / "best.pt").write_bytes(f"synthetic {fusion} {seed}".encode())
    record = {
        "schema_version": 1, "status": "completed", "finalized": True,
        "fusion_type": fusion, "seed": seed, "target": "dipole_moment", "unit": "Debye",
        "diagnostic": False,
        "protocol": {"split_seed": 20260927, "max_epochs": 200, "min_epochs": 50,
                     "early_stopping_patience": 30, "cmcr": False, "mixed_precision": False,
                     "official_qm9_filtering_applied": False, "batch_size": 128},
        "provenance": {"cache_sha256": "a" * 64, "source_sha256": {"Model.py": "b" * 64},
                       "split_ids_sha256": summary.sha256_file(run / "split_ids.json"),
                       "data_fingerprint_sha256": "c" * 64,
                       "checkpoint_sha256": summary.sha256_file(run / "best.pt")},
        "metrics": {"validation_mae": 0.8, "validation_rmse": 0.9,
                    "test_mae": abs(error), "test_rmse": abs(error), "test_n": 2},
        "params": {"encoder": 1000, "fusion": 100, "total": 1100},
        "timing": {"train_total_s": 80, "val_total_s": 10, "inference_total_s": 0.1,
                   "training_wall_s": 100, "cuda_peak_allocated_bytes": 1048576},
        "epochs": {"best": 160, "completed": 200, "max_epochs": 200,
                   "termination_reason": "max_epochs"},
        "predictions_file": "test_predictions.csv",
        "predictions_sha256": summary.sha256_file(prediction_path),
        "split_ids_file": "split_ids.json", "best_checkpoint_reload_verified": True,
        "initialization_sha256": {"encoders": f"{seed:064x}", "common_head": "e" * 64},
    }
    write_json(run / "summary.json", record)
    return run


class MetricTests(unittest.TestCase):
    def test_hand_calculated_metrics(self):
        actual = summary.prediction_metrics([0.0, 2.0], [1.0, 4.0])
        self.assertEqual(actual["mae"], 1.5)
        self.assertAlmostEqual(actual["rmse"], math.sqrt(2.5))
        self.assertEqual(actual["r2"], -1.5)
        self.assertEqual(actual["n"], 2)

    def test_constant_target_r2_is_undefined(self):
        self.assertIsNone(summary.prediction_metrics([1.0, 1.0], [1.0, 2.0])["r2"])

    def test_nonfinite_metadata_is_rejected(self):
        for value in (float("nan"), float("inf"), True, "1"):
            with self.assertRaises(summary.ValidationError):
                summary.finite_number(value, "metric")

    def test_known_hash(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "empty"
            path.write_bytes(b"")
            self.assertEqual(summary.sha256_file(path),
                             "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855")


class IntegrityTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.runs = self.root / "runs"
        self.out = self.root / "comparison"

    def all_runs(self):
        for fusion in summary.FUSIONS:
            for index, seed in enumerate(summary.SEEDS):
                make_run(self.runs, fusion, seed, error=(1, -2, 3)[index])

    def mutate_summary(self, run, mutate):
        path = run / "summary.json"
        record = summary.read_json(path)
        mutate(record)
        write_json(path, record)

    def test_final_summary_and_sample_sd_not_prediction_ensemble(self):
        self.all_runs()
        result = summary.summarize(self.runs, self.out)
        self.assertEqual(result["validated_runs"], 18)
        with (self.out / "comparison.csv").open(encoding="utf-8-sig", newline="") as handle:
            rows = list(csv.DictReader(handle))
        self.assertEqual(len(rows), 6)
        self.assertEqual(float(rows[0]["test_mae_Debye_mean"]), 2.0)
        self.assertEqual(float(rows[0]["test_mae_Debye_sample_sd"]), 1.0)
        self.assertEqual(float(rows[0]["delta_mae_vs_gate_Debye_mean"]), 0.0)
        self.assertEqual(float(rows[0]["training_wall_s_mean"]), 100.0)
        self.assertIn("三个独立训练运行", (self.out / "comparison.md").read_text(encoding="utf-8"))

    def test_missing_run_refuses_final_output(self):
        make_run(self.runs, "gate", 42)
        with self.assertRaisesRegex(summary.ValidationError, "all 18"):
            summary.summarize(self.runs, self.out)
        self.assertFalse(self.out.exists())

    def test_progress_only_does_not_publish_metrics(self):
        make_run(self.runs, "gate", 42)
        result = summary.summarize(self.runs, self.out, allow_partial=True)
        self.assertEqual(result["status"], "progress_only")
        self.assertEqual(result["completed"], 1)
        self.assertFalse((self.out / "comparison.csv").exists())
        self.assertTrue((self.out / "progress.csv").exists())
        self.assertNotIn("MAE", (self.out / "progress.md").read_text(encoding="utf-8"))

    def test_unfinalized_trained_run_is_not_completed(self):
        run = make_run(self.runs, "gate", 42)
        write_json(run / "status.json", {"status": "trained"})
        result = summary.summarize(self.runs, self.out, allow_partial=True)
        self.assertEqual(result["completed"], 0)

    def test_wrong_source_hash_rejected(self):
        self.all_runs()
        run = self.runs / "gate" / "seed_42"
        self.mutate_summary(run, lambda value: value["provenance"]["source_sha256"].update(
            {"Model.py": "d" * 64}))
        with self.assertRaisesRegex(summary.ValidationError, "provenance differs"):
            summary.summarize(self.runs, self.out)

    def test_test_id_change_rejected_even_when_hash_updated(self):
        run = make_run(self.runs, "gate", 42)
        path = run / "test_predictions.csv"
        path.write_text(path.read_text(encoding="utf-8").replace("6,0,", "8,0,"), encoding="utf-8")
        self.mutate_summary(run, lambda value: value.update(predictions_sha256=summary.sha256_file(path)))
        with self.assertRaisesRegex(summary.ValidationError, "IDs/order"):
            summary.load_run(run, "gate", 42)

    def test_saved_metrics_must_match_predictions(self):
        run = make_run(self.runs, "gate", 42)
        self.mutate_summary(run, lambda value: value["metrics"].update(test_mae=0.001))
        with self.assertRaisesRegex(summary.ValidationError, "disagrees"):
            summary.load_run(run, "gate", 42)

    def test_checkpoint_tamper_rejected(self):
        run = make_run(self.runs, "gate", 42)
        (run / "best.pt").write_bytes(b"wrong weights")
        with self.assertRaisesRegex(summary.ValidationError, "checkpoint SHA256"):
            summary.load_run(run, "gate", 42)

    def test_split_overlap_rejected_even_when_hash_updated(self):
        run = make_run(self.runs, "gate", 42)
        path = run / "split_ids.json"
        split = summary.read_json(path)
        split["train"] = [1, 2, 6]
        write_json(path, split)
        self.mutate_summary(run, lambda value: value["provenance"].update(
            split_ids_sha256=summary.sha256_file(path)))
        with self.assertRaisesRegex(summary.ValidationError, "leaking splits"):
            summary.load_run(run, "gate", 42)

    def test_wrong_same_seed_initialization_rejected(self):
        self.all_runs()
        run = self.runs / "gate" / "seed_42"
        self.mutate_summary(run, lambda value: value["initialization_sha256"].update(encoders="f" * 64))
        with self.assertRaisesRegex(summary.ValidationError, "initialization differs"):
            summary.summarize(self.runs, self.out)

    def test_truncated_training_rejected(self):
        run = make_run(self.runs, "gate", 42)
        self.mutate_summary(run, lambda value: value["epochs"].update(completed=199))
        with self.assertRaisesRegex(summary.ValidationError, "Truncated"):
            summary.load_run(run, "gate", 42)

    def test_diagnostic_run_is_rejected(self):
        run = make_run(self.runs, "gate", 42)
        self.mutate_summary(run, lambda value: value.update(diagnostic=True))
        with self.assertRaisesRegex(summary.ValidationError, "Diagnostic"):
            summary.load_run(run, "gate", 42)


if __name__ == "__main__":
    unittest.main()
