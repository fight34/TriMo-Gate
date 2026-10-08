"""Structural split and resume integrity checks without Torch, PyG, or training."""
import argparse
import ast
import hashlib
import json
import math
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

import numpy as np
import summarize_fusion


ROOT = Path(__file__).resolve().parent
TRAINER = ROOT / "train_fusion.py"
FUNCTIONS = {"make_split", "prepare_dataset", "protocol", "atomic_json", "read_json", "sha256",
             "source_hashes", "data_fingerprint", "verify_split", "verify_split_groups",
             "verify_config", "parse_args"}
tree = ast.parse(TRAINER.read_text(encoding="utf-8"))
scope = dict(np=np, argparse=argparse, hashlib=hashlib, json=json, math=math, os=os, Path=Path,
             __file__=str(TRAINER), __doc__=__doc__, DATASET="QM8", TARGET="E1-CC2", UNIT="hartree",
             EXPECTED_SAMPLE_COUNT=21747, FORMAL_SEEDS=(42, 2025, 3407),
             FUSION_CLASSES=dict.fromkeys(summarize_fusion.FUSIONS))
exec(compile(ast.Module(body=[node for node in tree.body if isinstance(node, ast.FunctionDef)
                             and node.name in FUNCTIONS], type_ignores=[]), str(TRAINER), "exec"), scope)


class GroupProtocolTests(unittest.TestCase):
    def fixture(self):
        ids = np.arange(1001, 1101, dtype=np.int64)
        groups = ids.copy()
        for begin, end in ((1, 6), (19, 22), (47, 49), (65, 69), (85, 89)):
            groups[begin:end] = ids[begin]
        return ids, groups

    def test_groups_never_leak_and_all_rows_are_covered_once(self):
        ids, groups = self.fixture()
        split = scope["make_split"](ids, 20260927, groups)
        np.testing.assert_array_equal(np.sort(np.concatenate(list(split.values()))), np.arange(len(ids)))
        group_sets = [set(groups[indices]) for indices in split.values()]
        self.assertTrue(all(group_sets))
        self.assertFalse(group_sets[0] & group_sets[1] or group_sets[0] & group_sets[2]
                         or group_sets[1] & group_sets[2])
        self.assertLessEqual(abs(len(split["train"]) - len(ids) * 0.8), 5)
        self.assertLessEqual(abs(len(split["train"]) + len(split["val"]) - len(ids) * 0.9), 5)

    def test_membership_is_invariant_to_cache_row_order(self):
        ids, groups = self.fixture()
        permutation = np.random.default_rng(987).permutation(len(ids))
        first = scope["make_split"](ids, 20260927, groups)
        second = scope["make_split"](ids[permutation], 20260927, groups[permutation])
        for name in first:
            self.assertEqual(set(ids[first[name]]), set(ids[permutation][second[name]]))

    def test_invalid_groups_and_duplicate_ids_are_rejected(self):
        for ids, groups in (([1, 2, 3], [1, 1, 1]), ([1, 2, 3], [1, 2]),
                            ([1, 1, 3], [1, 1, 3])):
            with self.assertRaises(ValueError):
                scope["make_split"](ids, 42, groups)

    def test_training_scaling_excludes_held_out_labels(self):
        ids, groups = self.fixture()
        split = scope["make_split"](ids, 20260927, groups)
        targets = np.zeros(len(ids), dtype=np.float64)
        targets[split["train"]] = np.arange(len(split["train"]))
        targets[split["val"]] = 1e9
        targets[split["test"]] = -1e9
        scope["load_data"] = lambda _: (SimpleNamespace(group_ids=groups), ids, targets, None)
        prepared = scope["prepare_dataset"](SimpleNamespace(split_seed=20260927))
        self.assertEqual(prepared[-1]["mean"], targets[split["train"]].mean())
        self.assertEqual(prepared[-1]["std"], targets[split["train"]].std())
        for part, indices in split.items():
            self.assertEqual(prepared[-2][part], groups[indices].tolist())

    def test_summary_rejects_cross_split_structural_groups(self):
        split = {"train": [1, 2], "val": [3], "test": [4]}
        with self.assertRaisesRegex(summarize_fusion.ValidationError, "crosses split"):
            summarize_fusion.validate_split_groups(split, {"train": [1, 2], "val": [1], "test": [4]})
        summarize_fusion.validate_split_groups(split, {"train": [1, 1], "val": [3], "test": [4]})

    def test_resume_checks_group_values_exact_hash_and_source(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            data, out = root / "cache", root / "run"
            data.mkdir()
            out.mkdir()
            args = scope["parse_args"](["--data", str(data), "--out", str(out), "--diagnostic"])
            ids, groups = self.fixture()
            targets = np.arange(len(ids), dtype=np.float64)
            dataset = SimpleNamespace(metadata={"cache_sha256": "a" * 64})
            # verify_config needs only len and metadata from the dataset.
            class Dataset:
                metadata = dataset.metadata
                def __len__(self):
                    return len(ids)
            dataset = Dataset()
            split = scope["make_split"](ids, args.split_seed, groups)
            split_ids = {part: ids[index].tolist() for part, index in split.items()}
            split_groups = {part: groups[index].tolist() for part, index in split.items()}
            scaling = {"mean": 1.0, "std": 2.0}
            model = {"fusion_type": args.fusion, "hidden": 256}
            write = scope["atomic_json"]
            write(data / "metadata.json", dataset.metadata)
            write(out / "split_ids.json", split_ids)
            write(out / "split_groups.json", split_groups)
            write(out / "target_scaling.json", scaling)
            config = {"fusion_type": args.fusion, "seed": args.seed, "diagnostic": True,
                      "model": model, "protocol": scope["protocol"](args, model, scaling,
                          {part: len(values) for part, values in split_ids.items()}),
                      "dataset": "QM8", "target": "E1-CC2", "unit": "hartree", "dataset_size": len(ids),
                      "provenance": {"source_sha256": scope["source_hashes"](), "cache_sha256": "a" * 64,
                                     "metadata_sha256": scope["sha256"](data / "metadata.json"),
                                     "split_ids_sha256": scope["sha256"](out / "split_ids.json"),
                                     "split_groups_sha256": scope["sha256"](out / "split_groups.json"),
                                     "data_fingerprint_sha256": scope["data_fingerprint"](ids, targets)}}
            write(out / "config.json", config)
            arguments = (args, out, dataset, ids, targets, scaling, model, split_ids, split_groups)
            self.assertEqual(scope["verify_config"](*arguments), config)
            wrong = {part: list(values) for part, values in split_groups.items()}
            wrong["train"][0] = -1
            write(out / "split_groups.json", wrong)
            with self.assertRaisesRegex(ValueError, "structural groups differ"):
                scope["verify_config"](*arguments)
            write(out / "split_groups.json", split_groups)
            with (out / "split_groups.json").open("a", encoding="utf-8") as handle:
                handle.write("\n")
            with self.assertRaisesRegex(ValueError, "groups SHA256"):
                scope["verify_config"](*arguments)
            write(out / "split_groups.json", split_groups)
            config["provenance"]["source_sha256"]["train_fusion.py"] = "0" * 64
            write(out / "config.json", config)
            with self.assertRaisesRegex(ValueError, "source_sha256"):
                scope["verify_config"](*arguments)


if __name__ == "__main__":
    unittest.main(verbosity=2)
