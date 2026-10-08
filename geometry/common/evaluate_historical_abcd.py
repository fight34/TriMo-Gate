"""Frozen A'/B'/C'/D inference on the existing matched test cohorts (CPU only)."""
from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
import platform
import sys
import time

for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[name] = "4"

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


class GeometryView(Dataset):
    def __init__(self, dataset, points):
        self.dataset = dataset
        self.points = torch.from_numpy(np.asarray(points).copy())

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        sample = self.dataset[index]
        sample["points"] = self.points[index]
        return sample


def validate_points(points, lengths, reference):
    assert points.shape == reference.shape and points.shape[2] == 9
    assert points.dtype == np.float32 and np.isfinite(points).all()
    mask = np.arange(points.shape[1])[None, :] < lengths[:, None]
    assert np.count_nonzero(points[~mask]) == 0
    types = points[:, :, 4:]
    assert np.isin(types, (0, 1)).all()
    np.testing.assert_array_equal(types.sum(axis=2), mask.astype(np.float32))
    # Point clouds need the same elemental composition, not a DFT atom ordering.
    np.testing.assert_array_equal(types.sum(axis=1), reference[:, :, 4:].sum(axis=1))
    np.testing.assert_allclose(points[:, :, :3].sum(axis=1) / lengths[:, None], 0, atol=2e-6)
    np.testing.assert_allclose(points[:, :, 3], np.linalg.norm(points[:, :, :3], axis=2), atol=2e-6, rtol=2e-6)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", choices=("qm8", "qm9"))
    parser.add_argument("--pilot", action="store_true")
    parser.add_argument("--source", type=Path, required=True, help="Prepared experiment with data/ and runs/")
    parser.add_argument("--coordinates", type=Path, required=True, help="Full generated coordinate directory")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    work = args.coordinates.resolve()
    source = args.source.resolve()
    sys.path.insert(0, str(source))
    import train_fusion as trainer
    import data

    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    torch.manual_seed(0)
    device = torch.device("cpu")
    dataset_class = data.QM8Dataset if args.dataset == "qm8" else data.QM9Dataset
    collate = data.collate_qm8 if args.dataset == "qm8" else data.collate_qm9
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=False)
    dataset = dataset_class(source / "data/dft")
    arrays = dataset.arrays
    split_ids = dataset.fixed_split_ids
    rows_by_id = {int(value): index for index, value in enumerate(arrays["ids"])}
    test_ids = split_ids["test"][:128] if args.pilot else split_ids["test"]
    rows = np.array([rows_by_id[value] for value in test_ids], dtype=np.int64)
    dataset.indices = rows
    dataset.ids = arrays["ids"][rows]
    dataset.targets = dataset._all_targets[rows]
    dft_points = arrays["points"][rows]
    lengths = arrays["point_lengths"][rows]
    mmff_metadata = read(source / "data/mmff/metadata.json")
    assert trainer.sha256(source / "data/mmff/cache.npz") == mmff_metadata["cache_sha256"]
    assert read(source / "data/mmff/fixed_split_ids.json") == split_ids
    with np.load(source / "data/mmff/cache.npz", allow_pickle=False) as archive:
        assert set(archive.files) == set(arrays)
        for key in archive.files:
            if key != "points":
                np.testing.assert_array_equal(archive[key], arrays[key], err_msg=key)
        mmff_points = archive["points"][rows]
    if args.dataset == "qm8":
        coordinate_path = work / "etkdg_all.npz"
        with np.load(coordinate_path, allow_pickle=False) as archive:
            np.testing.assert_array_equal(archive["ids"], arrays["ids"])
            assert archive["available"][rows].all()
            np.testing.assert_array_equal(archive["lengths"][rows], lengths)
            etkdg_points = archive["points"][rows]
    else:
        coordinate_path = work / "etkdg_points.npy"
        with np.load(work / "availability.npz", allow_pickle=False) as archive:
            np.testing.assert_array_equal(archive["ids"], arrays["ids"])
            assert archive["etkdg"][rows].all() and archive["mmff"][rows].all()
        etkdg_points = np.load(coordinate_path, mmap_mode="r", allow_pickle=False)[rows]
    for points in (dft_points, etkdg_points, mmff_points):
        validate_points(points, lengths, dft_points)
    np.testing.assert_array_equal(etkdg_points[:, :, 4:], mmff_points[:, :, 4:])
    definitions = {
        "A_prime": ("dft", "DFT", dft_points),
        "B_prime": ("dft", "ETKDGv3 (no additional MMFF)", etkdg_points),
        "C_prime": ("dft", "ETKDGv3 + MMFF94", mmff_points),
        "D": ("mmff", "ETKDGv3 + MMFF94", mmff_points),
    }
    protocol = {
        "dataset": trainer.DATASET, "target": trainer.TARGET, "unit": trainer.UNIT,
        "purpose": "Frozen geometry inference only; no training, checkpoint selection or target fitting",
        "device": "cpu", "torch": torch.__version__, "numpy": np.__version__,
        "python": platform.python_version(), "batch_size": 128, "threads": 4,
        "seeds": [42, 2025, 3407], "n": len(rows),
        "split_sizes": {key: len(value) for key, value in split_ids.items()},
        "point_features": dataset.metadata["point_features"],
        "dft_cache_sha256": dataset.metadata["cache_sha256"],
        "mmff_cache_sha256": mmff_metadata["cache_sha256"],
        "etkdg_file": str(coordinate_path), "etkdg_sha256": trainer.sha256(coordinate_path),
        "source_sha256": trainer.source_hashes(), "runner_sha256": trainer.sha256(__file__),
        "fixed_split_sha256": dataset.metadata["fixed_split_sha256"],
        "original_results_retained": True,
        "split_note": "QM8 structural groups retained" if args.dataset == "qm8" else "Original QM9 ungrouped random split retained; official exclusion list not applied",
    }
    trainer.atomic_json(out / "protocol.json", protocol)
    records, drift = [], []
    started = time.perf_counter()
    for seed in ([42] if args.pilot else protocol["seeds"]):
        for train_geometry in (["dft"] if args.pilot else ["dft", "mmff"]):
            run = source / "runs" / train_geometry / f"seed_{seed}"
            config, summary = read(run / "config.json"), read(run / "summary.json")
            scaling = read(run / "target_scaling.json")
            assert summary["status"] == "completed" and config["seed"] == seed
            assert config["geometry_condition"] == train_geometry
            assert read(run / "split_ids.json") == split_ids
            assert scaling == config["protocol"]["target_scaling"]
            assert scaling == read(source / "runs/dft" / f"seed_{seed}" / "target_scaling.json")
            assert config["protocol"]["batch_size"] == protocol["batch_size"]
            for name, digest in summary["provenance"]["source_sha256"].items():
                assert trainer.sha256(source / name) == digest
            checkpoint_sha = trainer.sha256(run / "best.pt")
            assert checkpoint_sha == summary["provenance"]["checkpoint_sha256"]
            checkpoint = trainer.load_checkpoint(run / "best.pt")
            assert checkpoint["config"] == config and checkpoint["split_ids"] == split_ids
            model = trainer.MultimodalModel(**config["model"]).to(device)
            model.load_state_dict(checkpoint["model_state"], strict=True)
            del checkpoint
            for label, (required_train, test_geometry, points) in definitions.items():
                if required_train != train_geometry or (args.pilot and label != "A_prime"):
                    continue
                loader = DataLoader(GeometryView(dataset, points), batch_size=128, shuffle=False,
                                    num_workers=0, collate_fn=collate)
                result = trainer.evaluate(model, loader, device, scaling)
                np.testing.assert_array_equal(result["ids"], test_ids)
                np.testing.assert_array_equal(result["targets"], dataset.targets)
                assert np.isfinite(result["predictions"]).all()
                folder = out / label / f"seed_{seed}"
                folder.mkdir(parents=True)
                trainer.write_predictions(folder / "test_predictions.csv", result)
                record = {"label": label, "seed": seed, "train_geometry": train_geometry,
                          "test_geometry": test_geometry, **result["metrics"],
                          "inference_seconds": result["seconds"], "checkpoint_sha256": checkpoint_sha,
                          "config_sha256": trainer.sha256(run / "config.json"),
                          "predictions_sha256": trainer.sha256(folder / "test_predictions.csv")}
                trainer.atomic_json(folder / "summary.json", record)
                records.append(record)
                if label != "B_prime":
                    original = run / "test_predictions.csv" if label != "C_prime" else source / "cross_inference/dft_to_mmff" / f"seed_{seed}" / "test_predictions.csv"
                    with original.open(encoding="utf-8", newline="") as handle:
                        old = list(csv.DictReader(handle))[:len(rows)]
                    np.testing.assert_array_equal([int(row["id"]) for row in old], test_ids)
                    np.testing.assert_array_equal([float(row["target_original_units"]) for row in old], result["targets"])
                    old_prediction = np.array([float(row["prediction_original_units"]) for row in old])
                    delta = np.abs(old_prediction - result["predictions"])
                    old_error = old_prediction - np.asarray(result["targets"])
                    drift.append({"label": label, "seed": seed, "max_prediction_abs_delta": float(delta.max()),
                                  "mean_prediction_abs_delta": float(delta.mean()),
                                  "original_mae": float(np.abs(old_error).mean()),
                                  "original_rmse": float(np.sqrt(np.mean(old_error ** 2))),
                                  "mae_delta": record["mae"] - float(np.abs(old_error).mean())})
                trainer.atomic_json(out / "status.json", {"completed": len(records), "expected": 1 if args.pilot else 12,
                                                            "last": record, "elapsed_seconds": time.perf_counter() - started})
                print(json.dumps({"dataset": trainer.DATASET, **record}), flush=True)
            del model
    aggregate = {}
    for label in definitions:
        runs = [row for row in records if row["label"] == label]
        if runs:
            aggregate[label] = {f"{metric}_{stat}": float(function([row[metric] for row in runs]))
                                for metric in ("mae", "rmse") for stat, function in
                                (("mean", np.mean), ("sd", lambda a: np.std(a, ddof=1) if len(a) > 1 else 0))}
    paired = []
    if not args.pilot:
        for left, right in (("B_prime", "A_prime"), ("C_prime", "B_prime"), ("C_prime", "A_prime"), ("D", "C_prime"), ("D", "A_prime")):
            for metric in ("mae", "rmse"):
                values = []
                for seed in protocol["seeds"]:
                    indexed = {row["label"]: row for row in records if row["seed"] == seed}
                    values.append(100 * (indexed[left][metric] / indexed[right][metric] - 1))
                paired.append({"comparison": f"{left}_vs_{right}", "metric": metric,
                               "per_seed_pct": values, "mean_pct": float(np.mean(values)),
                               "sd_pct": float(np.std(values, ddof=1))})
    trainer.atomic_json(out / "comparison.json", {"status": "completed", **protocol, "per_seed": records,
                                                  "aggregate": aggregate, "paired_changes": paired,
                                                  "original_backend_drift": drift,
                                                  "elapsed_seconds": time.perf_counter() - started})


if __name__ == "__main__":
    main()
