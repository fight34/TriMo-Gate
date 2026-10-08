"""Tests of trainer's data/protocol/recovery functions without requiring CUDA.

The production function definitions are extracted unchanged because this local
runtime has NumPy but no PyTorch/PyG. End-to-end trainer checks run on the server.
"""
import argparse
import ast
import contextlib
import hashlib
import io
import json
import math
import os
from pathlib import Path
import tempfile
import unittest

import numpy as np


TRAINER = Path(__file__).with_name("train_fusion.py")
FUNCTION_NAMES = {"atomic_json", "read_json", "sha256", "make_split", "data_fingerprint", "metrics",
                  "parse_args", "verify_split", "prepare_dataset", "recover_best_checkpoint"}
tree = ast.parse(TRAINER.read_text(encoding="utf-8"))
nodes = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in FUNCTION_NAMES]
scope = {"np": np, "argparse": argparse, "Path": Path, "math": math, "os": os,
         "json": json, "hashlib": hashlib, "__doc__": "trainer protocol tests",
         "FORMAL_SEEDS": (42, 2025, 3407),
         "FUSION_CLASSES": dict.fromkeys(("concat", "mean", "weighted", "attention", "gate_no_interaction", "gate"))}
exec(compile(ast.Module(body=nodes, type_ignores=[]), str(TRAINER), "exec"), scope)


class ProtocolTests(unittest.TestCase):
    def test_fixed_split_is_disjoint_exhaustive_and_order_invariant(self):
        ids = np.arange(1, 134, dtype=np.int64)
        reversed_ids = ids[::-1]
        split = scope["make_split"](ids, 20260927)
        reverse_split = scope["make_split"](reversed_ids, 20260927)
        self.assertEqual([len(split[key]) for key in ("train", "val", "test")], [106, 13, 14])
        for key in split:
            self.assertEqual(set(ids[split[key]]), set(reversed_ids[reverse_split[key]]))
        joined = np.concatenate(list(split.values()))
        np.testing.assert_array_equal(np.sort(joined), np.arange(len(ids)))
        self.assertEqual(len(np.unique(joined)), len(ids))

    def test_duplicate_ids_and_empty_partitions_fail(self):
        for ids in ([1, 1, 2], [1, 2, 3]):
            with self.assertRaises(ValueError):
                scope["make_split"](ids, 42)

    def test_target_scaling_uses_only_training_rows(self):
        args = argparse.Namespace(split_seed=20260927)
        ids = np.arange(1, 101)
        split = scope["make_split"](ids, args.split_seed)
        targets = np.ones(100, dtype=np.float64)
        targets[split["train"]] = np.arange(80)
        targets[split["val"]] = 10000
        targets[split["test"]] = -10000
        scope["load_data"] = lambda options: (object(), ids, targets, object())
        scaling = scope["prepare_dataset"](args)[-1]
        self.assertEqual(scaling["mean"], 39.5)
        self.assertEqual(scaling["std"], float(np.arange(80).std(ddof=0)))
        self.assertEqual(scaling["n_fit"], 80)
        self.assertEqual(scaling["fit_split"], "train")

    def test_sample_metrics_and_invalid_values(self):
        result = scope["metrics"]([0, 0, 9], [0, 0, 0])
        self.assertEqual(result["mae"], 3)
        self.assertEqual(result["rmse"], math.sqrt(27))
        for predictions, targets in (([], []), ([1], [1, 2]), ([float("nan")], [0]), ([0], [float("inf")])):
            with self.assertRaises((ValueError, FloatingPointError)):
                scope["metrics"](predictions, targets)

    def test_formal_protocol_is_bounded_and_requires_explicit_diagnostics(self):
        base = ["--data", "cache", "--out", "run"]
        options = scope["parse_args"](base)
        self.assertEqual((options.epochs, options.min_epochs, options.patience), (200, 50, 30))
        for alterations in (["--epochs", "2", "--min-epochs", "1"], ["--subset", "100"],
                            ["--seed", "17"], ["--max-minutes", "421"], ["--split-seed", "42"]):
            with self.assertRaises(SystemExit), contextlib.redirect_stderr(io.StringIO()):
                scope["parse_args"](base + alterations)
        options = scope["parse_args"](base + ["--diagnostic", "--epochs", "2", "--min-epochs", "1", "--subset", "100"])
        self.assertTrue(options.diagnostic)
        self.assertEqual(options.subset, 100)

    def test_persisted_split_is_checked_by_values_and_exact_bytes(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            split = {"train": [1, 2], "val": [3], "test": [4]}
            scope["atomic_json"](root / "split_ids.json", split)
            digest = scope["verify_split"](root, split)
            self.assertEqual(digest, scope["sha256"](root / "split_ids.json"))
            scope["atomic_json"](root / "split_ids.json", {**split, "test": [5]})
            with self.assertRaises(ValueError):
                scope["verify_split"](root, split, digest)
            scope["atomic_json"](root / "split_ids.json", split)
            with (root / "split_ids.json").open("a", encoding="utf-8") as handle:
                handle.write("\n")
            with self.assertRaises(ValueError):
                scope["verify_split"](root, split, digest)

    def test_best_recovery_uses_last_committed_epoch(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            last = {"config": {"seed": 42}, "best_epoch": 3, "best_val": 0.3}
            committed = {"config": last["config"], "epoch": 3, "best_val": 0.3}
            uncommitted = {"config": last["config"], "epoch": 4, "best_val": 0.2}
            scope["atomic_json"](root / "best_previous.pt", committed)
            scope["atomic_json"](root / "best.pt", uncommitted)
            scope["load_checkpoint"] = scope["read_json"]
            recovered = scope["recover_best_checkpoint"](root, last)
            self.assertEqual(recovered, committed)
            self.assertEqual(scope["read_json"](root / "best.pt"), committed)
            self.assertFalse((root / "best_previous.pt").exists())
            scope["atomic_json"](root / "best.pt", uncommitted)
            with self.assertRaises(ValueError):
                scope["recover_best_checkpoint"](root, last)


if __name__ == "__main__":
    unittest.main()
