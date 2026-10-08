"""Evaluate frozen DFT-trained checkpoints on the matched MMFF test coordinates."""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import time
import numpy as np
import torch
import train_fusion as trainer

ROOT = Path(__file__).resolve().parent
SEEDS = (42, 2025, 3407)


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def atomic(path, value):
    trainer.atomic_json(Path(path), value)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    if args.device != "cuda:0":
        raise ValueError("Frozen cross-inference protocol requires cuda:0")
    mmff_args = trainer.parse_args(["--data", str(ROOT/"data/mmff"),
                                     "--out", str(ROOT/"cross_inference/unused")])
    args.cpu_threads = mmff_args.cpu_threads
    device = trainer.prepare_device(args)
    dataset, ids, targets, collate, split, split_ids, mmff_scaling = trainer.prepare_dataset(mmff_args)
    for condition in ('dft', 'mmff'):
        data_dir = ROOT/'data'/condition
        metadata = read(data_dir/'metadata.json')
        assert trainer.sha256(data_dir/'cache.npz') == metadata['cache_sha256']
        assert trainer.sha256(data_dir/'fixed_split_ids.json') == metadata['fixed_split_sha256']
        for seed in SEEDS:
            folder = ROOT/'runs'/condition/f'seed_{seed}'
            config = read(folder/'config.json')
            assert read(folder/'summary.json')['status'] == 'completed'
            assert not config['diagnostic'] and config['geometry_condition'] == condition
            assert config['provenance']['source_sha256'] == trainer.source_hashes()
            assert config['provenance']['metadata_sha256'] == trainer.sha256(data_dir/'metadata.json')
            assert config['provenance']['cache_sha256'] == metadata['cache_sha256']
            assert config['provenance']['fixed_split_sha256'] == metadata['fixed_split_sha256']
    loaders = trainer.make_loaders(dataset, split, collate, mmff_args.batch_size, None, ("test",))
    out_root = ROOT/"cross_inference/dft_to_mmff"
    if out_root.exists():
        raise FileExistsError("Cross-inference output already exists; inspect instead of overwriting")
    out_root.mkdir(parents=True)
    overall = []
    for seed in SEEDS:
        source = ROOT/"runs/dft"/f"seed_{seed}"
        paired = ROOT/"runs/mmff"/f"seed_{seed}"
        config = read(source/"config.json")
        paired_config = read(paired/'config.json')
        for key in ('model', 'protocol', 'initialization_sha256', 'target', 'unit', 'dataset_size'):
            assert config[key] == paired_config[key], (seed, key)
        dft_summary = read(source/"summary.json")
        mmff_summary = read(paired/"summary.json")
        dft_split = read(source/"split_ids.json")
        dft_scaling = read(source/"target_scaling.json")
        assert config["geometry_condition"] == "dft" and config["seed"] == seed
        assert dft_summary["status"] == mmff_summary["status"] == "completed"
        assert dft_split == split_ids == read(paired/"split_ids.json")
        assert dft_scaling == mmff_scaling == read(paired/"target_scaling.json")
        best_path = source/"best.pt"
        checkpoint_sha = trainer.sha256(best_path)
        assert checkpoint_sha == dft_summary["provenance"]["checkpoint_sha256"]
        best = trainer.load_checkpoint(best_path)
        assert best["config"] == config and best["split_ids"] == split_ids
        model = trainer.MultimodalModel(**config["model"]).to(device)
        model.load_state_dict(best["model_state"])
        result = trainer.evaluate(model, loaders["test"], device, dft_scaling)
        assert result["ids"] == split_ids["test"]
        np.testing.assert_array_equal(result["targets"], targets[split["test"]])
        dft_final = read(source/"finalization_result.json")["test"]
        mmff_final = read(paired/"finalization_result.json")["test"]
        np.testing.assert_array_equal(result["ids"], dft_final["ids"])
        np.testing.assert_array_equal(result["ids"], mmff_final["ids"])
        np.testing.assert_array_equal(result["targets"], dft_final["targets"])
        np.testing.assert_array_equal(result["targets"], mmff_final["targets"])
        folder = out_root/f"seed_{seed}"
        folder.mkdir()
        trainer.write_predictions(folder/"test_predictions.csv", result)
        record = {
            "status": "completed", "label": "C_prime", "seed": seed,
            "train_geometry": "dft", "test_geometry": "mmff",
            "dataset": "QM9", "target": "dipole_moment", "unit": "Debye",
            "n": result["metrics"]["n"], "metrics": result["metrics"],
            "inference_seconds": result["seconds"],
            "source_checkpoint": str(best_path.relative_to(ROOT)),
            "checkpoint_sha256": checkpoint_sha,
            "dft_config_sha256": trainer.sha256(source/"config.json"),
            "mmff_cache_sha256": dataset.metadata["cache_sha256"],
            "fixed_split_sha256": dataset.metadata["fixed_split_sha256"],
            "predictions_sha256": trainer.sha256(folder/"test_predictions.csv"),
            "target_scaling": dft_scaling,
            "completed_utc": datetime.now(timezone.utc).isoformat(),
        }
        atomic(folder/"summary.json", record)
        overall.append(record)
        del model
        torch.cuda.empty_cache()
        print(json.dumps(record, ensure_ascii=False), flush=True)
    atomic(out_root/"summary.json", {"status": "completed", "runs": overall})


if __name__ == "__main__":
    main()
