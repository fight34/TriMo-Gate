"""Matched-cohort QM9 zero_point_vibrational_energy geometry comparison with separate finalization.

This uses matched successful geometries and the filtered original fixed split.
The official QM9 geometry exclusion list is not applied. No CMCR or HPO is run.
Training never evaluates the test split. Use finalize only after the planned
training runs have finished, with the protocol and best epochs already fixed.
"""
from __future__ import annotations

import argparse
import contextlib
import csv
import hashlib
import importlib.metadata
import json
import math
import os
import platform
import random
import signal
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from Model import FUSION_CLASSES, MultimodalModel


DATASET = "QM9"
TARGET = "zero_point_vibrational_energy"
UNIT = "hartree"
FORMAL_SEEDS = (42, 2025, 3407)
SCOPE = "controlled internal QM9 zero_point_vibrational_energy geometry ABCD comparison; matched eligible cohort; fixed original ungrouped split; gate only"


def atomic_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


def read_json(path):
    with Path(path).open(encoding="utf-8") as handle:
        return json.load(handle)


def event(out, kind, **values):
    record = {"event": kind, "time_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), **values}
    with (Path(out) / "events.jsonl").open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")
        handle.flush()
    print(json.dumps(record, ensure_ascii=False, allow_nan=False), flush=True)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def source_hashes():
    return {name: sha256(Path(__file__).parent / name) for name in ("train_fusion.py", "Model.py", "data.py")}


def state_fingerprint(state):
    digest = hashlib.sha256()
    for name, tensor in sorted(state.items()):
        array = tensor.detach().cpu().contiguous().numpy()
        digest.update(name.encode("utf-8"))
        digest.update(str(array.dtype).encode("ascii"))
        digest.update(str(array.shape).encode("ascii"))
        digest.update(array.tobytes())
    return digest.hexdigest()


def initialization_fingerprints(model):
    state = model.state_dict()
    encoders = {name: value for name, value in state.items()
                if name.startswith(("smiles_model.", "graph_model.", "point_cloud_model."))}
    common = {name: value for name, value in state.items()
              if name.startswith(("fusion_model.norm_S.", "fusion_model.norm_G.",
                                  "fusion_model.norm_P.", "fusion_model.norm_final.", "fusion_model.fc."))}
    return {"encoders": state_fingerprint(encoders), "common_head": state_fingerprint(common)}


@contextlib.contextmanager
def directory_lock(out):
    """Process-scoped file locking; stale files do not imply stale locks."""
    path = Path(out) / ".run.lock"
    descriptor = os.open(path, os.O_CREAT | os.O_RDWR, 0o600)
    locked = False
    try:
        if os.name == "nt":
            import msvcrt
            if os.fstat(descriptor).st_size == 0:
                os.write(descriptor, b"0")
            os.lseek(descriptor, 0, os.SEEK_SET)
            msvcrt.locking(descriptor, msvcrt.LK_NBLCK, 1)
        else:
            import fcntl
            fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        locked = True
        yield
    finally:
        if locked:
            if os.name == "nt":
                import msvcrt
                os.lseek(descriptor, 0, os.SEEK_SET)
                msvcrt.locking(descriptor, msvcrt.LK_UNLCK, 1)
            else:
                import fcntl
                fcntl.flock(descriptor, fcntl.LOCK_UN)
        os.close(descriptor)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def capture_rng(generator):
    return {"python": random.getstate(), "numpy": np.random.get_state(),
            "torch": torch.get_rng_state(), "loader": generator.get_state(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None}


def restore_rng(state, generator):
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"].cpu())
    generator.set_state(state["loader"].cpu())
    if state["cuda"] is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all([value.cpu() for value in state["cuda"]])


def prepare_device(args):
    torch.set_num_threads(args.cpu_threads)
    device = torch.device(args.device)
    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but unavailable")
        torch.cuda.set_device(device)
    elif device.type != "cpu":
        raise ValueError("Only CPU or a single CUDA device is supported")
    elif not args.diagnostic and args.mode != "self-test":
        raise ValueError("CPU is supported only for explicit diagnostics")
    return device


def environment(device):
    result = {"python": sys.version, "platform": platform.platform(), "torch": str(torch.__version__),
              "torch_cuda": torch.version.cuda, "device": str(device),
              "torch_threads": torch.get_num_threads(), "cuda_visible_devices": os.getenv("CUDA_VISIBLE_DEVICES"),
              "packages": {}}
    for name in ("numpy", "torch-geometric", "torch-scatter", "rdkit"):
        try:
            result["packages"][name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            result["packages"][name] = None
    if device.type == "cuda":
        properties = torch.cuda.get_device_properties(device)
        result.update(gpu_name=properties.name, gpu_total_bytes=properties.total_memory)
    return result


def sync(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def finite(value, name):
    if not bool(torch.isfinite(value).all()):
        raise FloatingPointError("Nonfinite " + name)


def make_split(ids, seed):
    ids = np.asarray(ids, dtype=np.int64).reshape(-1)
    if len(np.unique(ids)) != len(ids):
        raise ValueError("Dataset IDs must be unique")
    order = np.argsort(ids, kind="stable")
    np.random.default_rng(seed).shuffle(order)
    ntrain, nval = int(len(ids) * 0.8), int(len(ids) * 0.1)
    split = {"train": order[:ntrain], "val": order[ntrain:ntrain + nval], "test": order[ntrain + nval:]}
    if any(len(part) == 0 for part in split.values()):
        raise ValueError("Dataset too small for three nonempty splits")
    return {name: np.sort(part) for name, part in split.items()}


def data_fingerprint(ids, targets):
    digest = hashlib.sha256()
    digest.update(np.asarray(ids, dtype="<i8").tobytes())
    digest.update(np.asarray(targets, dtype="<f8").tobytes())
    return digest.hexdigest()


def load_data(args):
    from data import QM9Dataset, collate_qm9
    dataset = QM9Dataset(args.data, target=TARGET, verify_hash=True)
    if dataset.metadata.get("target_units", {}).get(TARGET) != UNIT:
        raise ValueError("Cache target unit is not the expected hartree")
    ids = np.asarray(dataset.ids, dtype=np.int64).reshape(-1)
    targets = np.asarray(dataset.targets, dtype=np.float64).reshape(-1)
    if len(ids) != len(dataset) or len(targets) != len(ids) or not np.isfinite(targets).all():
        raise ValueError("Invalid ID/target alignment or nonfinite target")
    if not args.diagnostic and len(dataset.arrays["ids"]) != 133885:
        raise ValueError("Formal protocol expects all 133885 original QM9 cache rows before eligibility filtering")
    if args.subset:
        if args.subset > len(ids):
            raise ValueError("Diagnostic subset exceeds cache size")
        order = np.argsort(ids, kind="stable")
        np.random.default_rng(args.split_seed + 1000003).shuffle(order)
        selected = np.sort(order[:args.subset])
        base = dataset
        dataset = Subset(base, selected.tolist())
        dataset.dims, dataset.metadata = base.dims, base.metadata
        dataset.fixed_split_ids = base.fixed_split_ids
        ids, targets = ids[selected], targets[selected]
    return dataset, ids, targets, collate_qm9


def model_config(args, dims):
    return {"vocab_size": int(dims["vocab_size"]), "smiles_length": int(dims["smiles_length"]),
            "d_smiles": args.hidden_dim, "smiles_num_heads": args.heads, "smiles_num_layers": args.smiles_layers,
            "cnn_kernels": args.hidden_dim, "d_node": int(dims["node_dim"]), "d_edge": int(dims["edge_dim"]),
            "d_hidden": args.hidden_dim, "graph_num_layers": args.graph_layers,
            "d_point_input": int(dims["point_dim"]), "d_point_cloud": args.point_dim,
            "d_point_output": args.hidden_dim, "d_output": 1,
            "use_smiles": True, "use_graph": True, "use_point_cloud": True,
            "dropout": 0.1, "fusion_type": args.fusion,
            "fusion_hidden_dim": None, "fusion_attention_heads": 4}


def protocol(args, config, scaling):
    common = {key: value for key, value in config.items() if key != "fusion_type"}
    return {"split_seed": args.split_seed, "max_epochs": args.epochs, "min_epochs": args.min_epochs,
            "early_stopping_patience": args.patience, "batch_size": args.batch_size,
            "learning_rate": args.learning_rate, "weight_decay": args.weight_decay, "grad_clip": args.grad_clip,
            "optimizer": "AdamW", "scheduler": "ReduceLROnPlateau", "scheduler_factor": 0.5,
            "scheduler_patience": 10, "scheduler_min_lr": 1e-6, "scheduler_threshold": 0.0,
            "scheduler_threshold_mode": "abs", "loss": "sample-mean L1 on train-standardized target",
            "cmcr": False, "mixed_precision": False, "num_workers": 0,
            "split_strategy": "original fixed split filtered by matched geometry success; original ID order; no resplitting",
            "split_grouping": "original QM9 split is ungrouped; no structural group separation claim",
            "official_qm9_filtering_applied": False, "model_common": common, "target_scaling": scaling,
            "cpu_threads": args.cpu_threads, "max_minutes": args.max_minutes,
            "diagnostic_subset": args.subset, "diagnostic_max_steps": args.max_steps,
            "training_rng_offset": 1729, "early_stopping_min_delta": 0.0}


def make_loaders(dataset, split, collate, batch_size, generator, names):
    # Caller must explicitly request test; train() passes only train and val.
    return {name: DataLoader(Subset(dataset, split[name].tolist()), batch_size=batch_size,
                            shuffle=name == "train", collate_fn=collate, num_workers=0,
                            generator=generator if name == "train" else None)
            for name in names}


def model_forward(model, batch, device):
    prediction = model(batch["smiles_ids"].to(device), batch["graph"].to(device),
                       batch["points"].to(device), batch["point_lengths"].to(device)).reshape(-1)
    finite(prediction, "prediction")
    return prediction


def metrics(predictions, targets):
    prediction = np.asarray(predictions, dtype=np.float64)
    truth = np.asarray(targets, dtype=np.float64)
    if prediction.ndim != 1 or prediction.shape != truth.shape or not len(prediction):
        raise ValueError("Evaluation arrays must be aligned nonempty vectors")
    if not np.isfinite(prediction).all() or not np.isfinite(truth).all():
        raise FloatingPointError("Nonfinite evaluation arrays")
    errors = prediction - truth
    return {"n": len(prediction), "mae": float(np.abs(errors).mean()),
            "rmse": float(np.sqrt(np.square(errors).mean()))}


@torch.no_grad()
def evaluate(model, loader, device, scaling):
    model.eval()
    predictions, normalized, targets, ids = [], [], [], []
    sync(device)
    start = time.perf_counter()
    for batch in loader:
        pred = model_forward(model, batch, device).double().cpu().numpy()
        normalized.extend(pred.tolist())
        predictions.extend((pred * scaling["std"] + scaling["mean"]).tolist())
        targets.extend(batch["y"].reshape(-1).double().cpu().numpy().tolist())
        ids.extend(batch["id"].reshape(-1).tolist())
    sync(device)
    return {"ids": ids, "targets": targets, "predictions": predictions,
            "normalized_predictions": normalized, "metrics": metrics(predictions, targets),
            "seconds": time.perf_counter() - start}


def write_predictions(path, result):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["id", "target_original_units", "prediction_original_units"])
        writer.writerows(zip(result["ids"], result["targets"], result["predictions"]))
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


def load_checkpoint(path):
    # Only this experiment's locally generated trusted checkpoints are accepted.
    return torch.load(path, map_location="cpu", weights_only=False)


def save_checkpoint(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(value, temporary)
    temporary.replace(path)


def recover_best_checkpoint(out, last):
    """Recover an interrupted best/last commit without changing completed history."""
    out = Path(out)
    best_path = out / "best.pt"
    for candidate in (best_path, out / "best_previous.pt"):
        if not candidate.is_file():
            continue
        checkpoint = load_checkpoint(candidate)
        if (checkpoint["config"] == last["config"] and checkpoint["epoch"] == last["best_epoch"]
                and checkpoint["best_val"] == last["best_val"]):
            if candidate != best_path:
                candidate.replace(best_path)
            return checkpoint
    raise ValueError("No best checkpoint matches the last completed epoch")


def write_epoch_history(out, records):
    # Regenerate from checkpoint-owned history: retries cannot double-count epochs.
    path = Path(out) / "epochs.jsonl"
    temporary = path.with_suffix(".jsonl.tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, allow_nan=False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


def verify_split(out, split_ids, expected_sha256=None):
    path = Path(out) / "split_ids.json"
    if read_json(path) != split_ids:
        raise ValueError("Persisted split IDs differ from the fixed split protocol")
    actual_hash = sha256(path)
    if expected_sha256 is not None and actual_hash != expected_sha256:
        raise ValueError("Persisted split file SHA256 mismatch")
    return actual_hash


def training_summary(config, checkpoint, reason):
    history = checkpoint["history"]
    return {"schema_version": 1, "status": "trained", "finalized": False,
            "scope": config["scope"], "diagnostic": config["diagnostic"],
            "dataset": DATASET, "geometry_condition": config["geometry_condition"],
            "fusion_type": config["fusion_type"], "seed": config["seed"], "target": TARGET, "unit": UNIT,
            "protocol": config["protocol"], "provenance": config["provenance"],
            "params": config["params"], "initialization_sha256": config["initialization_sha256"],
            "epochs": {"best": checkpoint["best_epoch"], "completed": checkpoint["epoch"],
                       "max_epochs": config["protocol"]["max_epochs"], "termination_reason": reason},
            "metrics": {"best_validation_mae": checkpoint["best_val"]},
            "timing": {"train_total_s": sum(row["train_seconds"] for row in history),
                       "val_total_s": sum(row["validation_seconds"] for row in history),
                       "training_wall_s": checkpoint["training_wall_s"],
                       "cuda_peak_allocated_bytes": max((row["cuda_peak_allocated_bytes"] or 0 for row in history), default=0)},
            "test_evaluated": False}


def verify_config(args, out, dataset, ids, targets, scaling, net_config, split_ids):
    config = read_json(Path(out) / "config.json")
    checks = {"fusion_type": args.fusion, "seed": args.seed, "diagnostic": args.diagnostic,
              "model": net_config, "protocol": protocol(args, net_config, scaling),
              "dataset": DATASET, "geometry_condition": dataset.metadata["geometry_condition"],
              "target": TARGET, "unit": UNIT, "dataset_size": len(dataset)}
    for key, expected in checks.items():
        if config[key] != expected:
            raise ValueError("Configuration mismatch: " + key)
    provenance = config["provenance"]
    for key, expected in {"source_sha256": source_hashes(),
                          "cache_sha256": dataset.metadata["cache_sha256"],
                          "metadata_sha256": sha256(Path(args.data) / "metadata.json"),
                          "fixed_split_sha256": dataset.metadata["fixed_split_sha256"],
                          "data_fingerprint_sha256": data_fingerprint(ids, targets)}.items():
        if provenance[key] != expected:
            raise ValueError("Provenance mismatch: " + key)
    verify_split(out, split_ids, provenance["split_ids_sha256"])
    if read_json(Path(out) / "target_scaling.json") != scaling:
        raise ValueError("Saved training target scaling differs")
    return config


def prepare_dataset(args):
    dataset, ids, targets, collate = load_data(args)
    row_by_id = {int(identifier): row for row, identifier in enumerate(ids)}
    split = {name: np.asarray([row_by_id[identifier] for identifier in dataset.fixed_split_ids[name]
                               if identifier in row_by_id], dtype=np.int64)
             for name in ("train", "val", "test")}
    if any(not len(indices) for indices in split.values()):
        raise ValueError("Every fixed split must remain nonempty, including diagnostic subsets")
    split_ids = {name: ids[index].tolist() for name, index in split.items()}
    scaling = {"mean": float(targets[split["train"]].mean()),
               "std": float(targets[split["train"]].std(ddof=0)),
               "fit_split": "train", "n_fit": len(split["train"]), "ddof": 0}
    if not math.isfinite(scaling["std"]) or scaling["std"] <= 0:
        raise ValueError("Invalid training-only target standard deviation")
    return dataset, ids, targets, collate, split, split_ids, scaling


def train(args):
    started = time.perf_counter()
    out = Path(args.out)
    args._attempt_started = started
    budget = read_json(out / "budget.json") if (out / "budget.json").exists() else {}
    args._previous_seconds = float(budget.get("training_wall_s", 0.0))
    device = prepare_device(args)
    dataset, ids, targets, collate, split, split_ids, scaling = prepare_dataset(args)
    net_config = model_config(args, dataset.dims)
    set_seed(args.seed)
    model = MultimodalModel(**net_config).to(device)
    initial_hashes = initialization_fingerprints(model)
    counts = model.get_parameter_counts()
    generator = torch.Generator().manual_seed(args.seed)
    loaders = make_loaders(dataset, split, collate, args.batch_size, generator, ("train", "val"))
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=10, threshold=0.0, threshold_mode="abs", min_lr=1e-6)
    start_epoch, best_epoch, bad_epochs = 0, 0, 0
    best_val, history, previous_seconds = float("inf"), [], args._previous_seconds
    if args.resume:
        config = verify_config(args, out, dataset, ids, targets, scaling, net_config, split_ids)
        args._owns_run = True
        checkpoint = load_checkpoint(args.resume)
        if checkpoint["config"] != config or checkpoint["split_ids"] != split_ids:
            raise ValueError("Checkpoint configuration or split mismatch")
        if initial_hashes != config["initialization_sha256"]:
            raise ValueError("Recreated initial model does not match recorded initialization")
        if Path(args.resume).name != "last.pt":
            raise ValueError("Resume only last.pt; best.pt is reserved for final evaluation")
        model.load_state_dict(checkpoint["model_state"])
        optimizer.load_state_dict(checkpoint["optimizer_state"])
        scheduler.load_state_dict(checkpoint["scheduler_state"])
        restore_rng(checkpoint["rng_state"], generator)
        start_epoch, best_epoch, best_val = checkpoint["epoch"], checkpoint["best_epoch"], checkpoint["best_val"]
        history, bad_epochs = checkpoint["history"], checkpoint["bad_epochs"]
        previous_seconds = max(previous_seconds, checkpoint["training_wall_s"])
        args._previous_seconds = previous_seconds
        if len(history) != start_epoch or [row["epoch"] for row in history] != list(range(1, start_epoch + 1)):
            raise ValueError("Checkpoint epoch history is inconsistent")
        recover_best_checkpoint(out, checkpoint)
        write_epoch_history(out, history)
        reason = checkpoint.get("termination_reason")
        if reason in ("max_epochs", "early_stopping"):
            summary = training_summary(config, checkpoint, reason)
            atomic_json(out / "training_summary.json", summary)
            atomic_json(out / "status.json", {"status": "trained", "epoch": start_epoch, "best_epoch": best_epoch,
                                               "termination_reason": reason, "diagnostic": args.diagnostic})
            event(out, "training_already_finished", epoch=start_epoch)
            return
    else:
        atomic_json(out / "split_ids.json", split_ids)
        atomic_json(out / "target_scaling.json", scaling)
        config = {"schema_version": 1, "scope": SCOPE, "diagnostic": args.diagnostic,
                  "dataset": DATASET, "geometry_condition": dataset.metadata["geometry_condition"],
                  "fusion_type": args.fusion, "seed": args.seed, "target": TARGET, "unit": UNIT,
                  "data": str(Path(args.data).resolve()), "dataset_size": len(dataset), "model": net_config,
                  "protocol": protocol(args, net_config, scaling), "initialization_sha256": initial_hashes,
                  "provenance": {"source_sha256": source_hashes(), "cache_sha256": dataset.metadata["cache_sha256"],
                                 "metadata_sha256": sha256(Path(args.data) / "metadata.json"),
                                 "fixed_split_sha256": dataset.metadata["fixed_split_sha256"],
                                 "split_ids_sha256": sha256(out / "split_ids.json"),
                                 "data_fingerprint_sha256": data_fingerprint(ids, targets)},
                  "params": {"encoder": counts["smiles_encoder"] + counts["graph_encoder"] + counts["point_encoder"],
                             "fusion": counts["fusion"], "total": counts["total"]},
                  "parameter_count_by_module": counts, "environment": environment(device),
                  "dataset_metadata": dataset.metadata,
                  "known_limitations": ["No official QM9 geometry exclusions; controlled internal comparison only",
                                        "Original SMILES GroupNorm retains padding dependence",
                                        "CUDA scatter reductions may be nondeterministic",
                                        "Original ungrouped split is retained; structural overlap is not excluded",
                                        "MMFF-success filtering defines the matched comparison cohort"]}
        atomic_json(out / "config.json", config)
        # Private fusion layer sizes must not change the initial training RNG stream.
        set_seed(args.seed + 1729)
    atomic_json(out / "status.json", {"status": "running", "stage": "training", "pid": os.getpid(),
                                      "diagnostic": args.diagnostic, "epoch": start_epoch})
    event(out, "prepared", fusion=args.fusion, geometry_condition=dataset.metadata["geometry_condition"],
          seed=args.seed, split_seed=args.split_seed,
          split_sizes={name: len(indices) for name, indices in split.items()}, params=config["params"],
          resumed_epoch=start_epoch, initialization_sha256=initial_hashes)
    for epoch in range(start_epoch + 1, args.epochs + 1):
        if previous_seconds + time.perf_counter() - started >= args.max_minutes * 60:
            raise TimeoutError("Cumulative wall-time budget reached; completed-epoch checkpoint retained")
        atomic_json(out / "status.json", {"status": "running", "stage": "training", "epoch": epoch,
                                          "pid": os.getpid(), "diagnostic": args.diagnostic})
        model.train()
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        sync(device)
        epoch_started = time.perf_counter()
        total_error, samples_seen, steps = 0.0, 0, 0
        learning_rate = optimizer.param_groups[0]["lr"]
        for batch in loaders["train"]:
            optimizer.zero_grad(set_to_none=True)
            pred = model_forward(model, batch, device)
            target = batch["y"].to(device).reshape(-1)
            finite(target, "target")
            normalized = ((target.double() - scaling["mean"]) / scaling["std"]).to(pred.dtype)
            loss = (pred - normalized).abs().mean()
            finite(loss, "training loss")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip, error_if_nonfinite=True)
            optimizer.step()
            total_error += float(loss.detach()) * len(target)
            samples_seen += len(target)
            steps += 1
            if args.max_steps and steps >= args.max_steps:
                break
            if previous_seconds + time.perf_counter() - started >= args.max_minutes * 60:
                raise TimeoutError("Cumulative wall-time budget reached during epoch; last completed epoch retained")
        sync(device)
        train_seconds = time.perf_counter() - epoch_started
        if not samples_seen:
            raise RuntimeError("No training samples were processed")
        if not args.diagnostic and samples_seen != len(split["train"]):
            raise RuntimeError("Formal training epoch was incomplete")
        val = evaluate(model, loaders["val"], device, scaling)
        val_mae = val["metrics"]["mae"]
        improved = val_mae < best_val
        if improved:
            best_val, best_epoch, bad_epochs = val_mae, epoch, 0
        else:
            bad_epochs += 1
        scheduler.step(val_mae)
        reason = "max_epochs" if epoch == args.epochs else None
        if epoch >= args.min_epochs and bad_epochs >= args.patience and epoch < args.epochs:
            reason = "early_stopping"
        row = {"epoch": epoch, "train_steps": steps, "train_samples": samples_seen,
               "train_complete_epoch": samples_seen == len(split["train"]),
               "train_mae_standardized": total_error / samples_seen,
               "train_mae_original_units": total_error / samples_seen * scaling["std"],
               "validation_mae": val_mae, "validation_rmse": val["metrics"]["rmse"],
               "validation_n": val["metrics"]["n"], "learning_rate": learning_rate,
               "next_learning_rate": optimizer.param_groups[0]["lr"], "improved": improved,
               "bad_epochs": bad_epochs, "train_seconds": train_seconds, "validation_seconds": val["seconds"],
               "cuda_peak_allocated_bytes": torch.cuda.max_memory_allocated(device) if device.type == "cuda" else None}
        history.append(row)
        checkpoint = {"format_version": 1, "config": config, "epoch": epoch, "best_epoch": best_epoch,
                      "best_val": best_val, "bad_epochs": bad_epochs, "history": history,
                      "training_wall_s": previous_seconds + time.perf_counter() - started,
                      "termination_reason": reason, "model_state": model.state_dict(),
                      "optimizer_state": optimizer.state_dict(), "scheduler_state": scheduler.state_dict(),
                      "rng_state": capture_rng(generator), "split_ids": split_ids, "validation_reference": val}
        # Commit best first; last then contains the complete resumable epoch state.
        if improved:
            if (out / "best.pt").exists():
                (out / "best.pt").replace(out / "best_previous.pt")
            save_checkpoint(out / "best.pt", checkpoint)
        save_checkpoint(out / "last.pt", checkpoint)
        write_epoch_history(out, history)
        event(out, "epoch_complete", **row)
        if reason:
            summary = training_summary(config, checkpoint, reason)
            atomic_json(out / "training_summary.json", summary)
            atomic_json(out / "status.json", {"status": "trained", "epoch": epoch, "best_epoch": best_epoch,
                                              "termination_reason": reason, "diagnostic": args.diagnostic})
            event(out, "trained", epoch=epoch, best_epoch=best_epoch, best_validation_mae=best_val,
                  termination_reason=reason, test_evaluated=False)
            return
    raise RuntimeError("No valid training termination state")


def finalize(args):
    out = Path(args.out)
    device = prepare_device(args)
    dataset, ids, targets, collate, split, split_ids, scaling = prepare_dataset(args)
    config = verify_config(args, out, dataset, ids, targets, scaling, model_config(args, dataset.dims), split_ids)
    args._owns_run = True
    complete_path = out / "summary.json"
    if complete_path.exists():
        complete = read_json(complete_path)
        if (complete["status"] != "completed" or not complete["finalized"]
                or complete["provenance"]["checkpoint_sha256"] != sha256(out / "best.pt")
                or complete["predictions_sha256"] != sha256(out / "test_predictions.csv")):
            raise ValueError("Existing finalized artifacts do not match their recorded hashes")
        atomic_json(out / "status.json", {"status": "completed", "finalized": True, "diagnostic": args.diagnostic})
        event(out, "already_finalized", test_evaluated_again=False)
        return
    last = load_checkpoint(out / "last.pt")
    if last["config"] != config or last.get("termination_reason") not in ("max_epochs", "early_stopping"):
        raise ValueError("Finalization requires completed training with the unchanged protocol")
    if last["epoch"] < args.min_epochs or last["epoch"] > args.epochs:
        raise ValueError("Invalid completed epoch count")
    if last["termination_reason"] == "max_epochs" and last["epoch"] != args.epochs:
        raise ValueError("Maximum-epoch termination has inconsistent epoch count")
    if not args.diagnostic and not all(row["train_complete_epoch"] for row in last["history"]):
        raise ValueError("Formal table cannot use truncated training epochs")
    best = load_checkpoint(out / "best.pt")
    if (best["config"] != config or best["epoch"] != last["best_epoch"]
            or best["best_val"] != last["best_val"] or best["split_ids"] != split_ids):
        raise ValueError("Best checkpoint identity or split mismatch")
    checkpoint_hash = sha256(out / "best.pt")
    atomic_json(out / "status.json", {"status": "finalizing", "pid": os.getpid(), "diagnostic": args.diagnostic})
    result_path = out / "finalization_result.json"
    if result_path.exists():
        # Reuse an already committed held-out evaluation after an export failure.
        result = read_json(result_path)
        if result["checkpoint_sha256"] != checkpoint_hash or result["config_sha256"] != sha256(out / "config.json"):
            raise ValueError("Persisted finalization result provenance mismatch")
    else:
        loaders = make_loaders(dataset, split, collate, args.batch_size, None, ("val", "test"))
        model = MultimodalModel(**config["model"]).to(device)
        model.load_state_dict(best["model_state"])
        val = evaluate(model, loaders["val"], device, scaling)
        reference = best["validation_reference"]
        np.testing.assert_array_equal(val["ids"], reference["ids"])
        np.testing.assert_array_equal(val["targets"], reference["targets"])
        np.testing.assert_allclose(val["normalized_predictions"], reference["normalized_predictions"], rtol=1e-5, atol=1e-6)
        difference = float(np.max(np.abs(np.asarray(val["normalized_predictions"]) - reference["normalized_predictions"])))
        test = evaluate(model, loaders["test"], device, scaling)
        result = {"checkpoint_sha256": checkpoint_hash, "config_sha256": sha256(out / "config.json"),
                  "validation": val, "test": test, "reload_max_normalized_difference": difference}
        atomic_json(result_path, result)
    val, test = result["validation"], result["test"]
    np.testing.assert_array_equal(val["ids"], split_ids["val"])
    np.testing.assert_array_equal(test["ids"], split_ids["test"])
    np.testing.assert_array_equal(test["targets"], targets[split["test"]])
    if metrics(test["predictions"], test["targets"]) != test["metrics"]:
        raise ValueError("Persisted test metrics inconsistent with predictions")
    write_predictions(out / "validation_predictions.csv", val)
    write_predictions(out / "test_predictions.csv", test)
    summary = training_summary(config, last, last["termination_reason"])
    summary.update(status="completed", finalized=True, test_evaluated=True,
                   split_ids_file="split_ids.json", predictions_file="test_predictions.csv",
                   predictions_sha256=sha256(out / "test_predictions.csv"),
                   best_checkpoint_reload_verified=True,
                   checkpoint_reload_max_normalized_difference=result["reload_max_normalized_difference"])
    summary["provenance"] = {**config["provenance"], "checkpoint_sha256": checkpoint_hash,
                             "config_sha256": sha256(out / "config.json")}
    summary["metrics"] = {"validation_mae": val["metrics"]["mae"], "validation_rmse": val["metrics"]["rmse"],
                          "validation_n": val["metrics"]["n"], "test_mae": test["metrics"]["mae"],
                          "test_rmse": test["metrics"]["rmse"], "test_n": test["metrics"]["n"]}
    summary["timing"].update(inference_total_s=test["seconds"], validation_reload_s=val["seconds"])
    atomic_json(out / "summary.json", summary)
    atomic_json(out / "status.json", {"status": "completed", "finalized": True, "diagnostic": args.diagnostic,
                                      "best_epoch": best["epoch"], "completed_epochs": last["epoch"]})
    event(out, "completed", fusion=args.fusion, seed=args.seed, diagnostic=args.diagnostic, **summary["metrics"])


def self_test(args):
    from torch_geometric.data import Batch, Data
    device = prepare_device(args)
    if metrics([0, 0, 9], [0, 0, 0])["mae"] != 3.0:
        raise AssertionError("MAE must weight samples, not batches")
    sample_ids = np.arange(100, dtype=np.int64)
    split_a, split_b = make_split(sample_ids, 7), make_split(sample_ids[::-1], 7)
    for name in split_a:
        if set(sample_ids[split_a[name]]) != set(sample_ids[::-1][split_b[name]]):
            raise AssertionError("Split identity depends on input row order")
    args.hidden_dim, args.heads, args.smiles_layers, args.graph_layers, args.point_dim = 16, 4, 1, 1, 32
    dims = {"vocab_size": 10, "smiles_length": 8, "node_dim": 9, "edge_dim": 4, "point_dim": 9}
    set_seed(777)
    graphs = [Data(x=torch.randn(3, 9), edge_index=torch.tensor([[0, 1, 1, 2], [1, 0, 2, 1]]),
                   edge_attr=torch.randn(4, 4)) for _ in range(3)]
    batch = {"smiles_ids": torch.tensor([[1, 2, 3, 4, 5, 6, 7, 0]] * 3),
             "graph": Batch.from_data_list(graphs), "points": torch.randn(3, 5, 9),
             "point_lengths": torch.tensor([5, 4, 3])}
    expected_hashes = None
    for fusion in FUSION_CLASSES:
        args.fusion = fusion
        set_seed(42)
        config = model_config(args, dims)
        model = MultimodalModel(**config).to(device)
        hashes = initialization_fingerprints(model)
        if expected_hashes is None:
            expected_hashes = hashes
        elif hashes != expected_hashes:
            raise AssertionError("Fusion variants changed common encoder/head initialization")
        optimizer = torch.optim.AdamW(model.parameters(), lr=2e-4)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10)
        generator = torch.Generator().manual_seed(42)
        prediction = model_forward(model, batch, device)
        prediction.abs().mean().backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5, error_if_nonfinite=True)
        optimizer.step()
        scheduler.step(float(prediction.detach().abs().mean()))
        model.eval()
        with torch.no_grad():
            before = model_forward(model, batch, device).cpu()
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "check.pt"
            save_checkpoint(path, {"model": model.state_dict(), "optimizer": optimizer.state_dict(),
                                   "scheduler": scheduler.state_dict(), "rng": capture_rng(generator)})
            saved = load_checkpoint(path)
            other = MultimodalModel(**config).to(device)
            other.load_state_dict(saved["model"])
            optimizer2 = torch.optim.AdamW(other.parameters(), lr=2e-4)
            optimizer2.load_state_dict(saved["optimizer"])
            scheduler2 = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer2, patience=10)
            scheduler2.load_state_dict(saved["scheduler"])
            if scheduler2.state_dict() != scheduler.state_dict():
                raise AssertionError("Scheduler state failed checkpoint round-trip")
            restore_rng(saved["rng"], generator)
            other.eval()
            with torch.no_grad():
                after = model_forward(other, batch, device).cpu()
            torch.testing.assert_close(before, after, rtol=1e-5, atol=1e-6)
        print(json.dumps({"self_test": "passed", "fusion": fusion, "device": str(device),
                          "initialization_sha256": hashes}), flush=True)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("train", "finalize", "self-test"), default="train")
    parser.add_argument("--data")
    parser.add_argument("--out")
    parser.add_argument("--fusion", choices=("gate",), default="gate")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--split-seed", type=int, default=20260927)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--min-epochs", type=int, default=50)
    parser.add_argument("--patience", type=int, default=30)
    parser.add_argument("--max-minutes", type=float, default=420)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--heads", type=int, default=8)
    parser.add_argument("--smiles-layers", type=int, default=6)
    parser.add_argument("--graph-layers", type=int, default=6)
    parser.add_argument("--point-dim", type=int, default=1024)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--grad-clip", type=float, default=0.5)
    parser.add_argument("--cpu-threads", type=int, default=4)
    parser.add_argument("--resume")
    parser.add_argument("--diagnostic", action="store_true")
    parser.add_argument("--subset", type=int, default=0)
    parser.add_argument("--max-steps", type=int, default=0)
    args = parser.parse_args(argv)
    if args.mode == "self-test":
        return args
    if not args.data or not args.out:
        parser.error("--data and --out are required")
    if not 1 <= args.min_epochs <= args.epochs <= 200 or args.patience < 1:
        parser.error("Require 1 <= min-epochs <= epochs <= 200 and positive patience")
    if not math.isfinite(args.max_minutes) or not 0 < args.max_minutes <= 420:
        parser.error("--max-minutes must be finite and in (0, 420]")
    if min(args.batch_size, args.hidden_dim, args.heads, args.smiles_layers, args.graph_layers,
           args.point_dim, args.cpu_threads) < 1 or min(args.subset, args.max_steps) < 0:
        parser.error("Invalid dimensions, batch size, threads or diagnostic limits")
    if args.hidden_dim % args.heads or args.hidden_dim % 4:
        parser.error("Hidden dimension must be divisible by encoder heads and four fusion heads")
    if not 0 <= args.seed <= 2**32 - 1730 or not 0 <= args.split_seed <= 2**32 - 1000004:
        parser.error("Seeds outside supported range")
    for name in ("learning_rate", "weight_decay", "grad_clip"):
        if not math.isfinite(getattr(args, name)) or getattr(args, name) <= 0:
            parser.error(name + " must be finite and positive")
    if not args.diagnostic:
        expected = {"epochs": 200, "min_epochs": 50, "patience": 30, "batch_size": 128,
                    "hidden_dim": 256, "heads": 8, "smiles_layers": 6, "graph_layers": 6,
                    "point_dim": 1024, "learning_rate": 2e-4, "weight_decay": 1e-4,
                    "grad_clip": 0.5, "cpu_threads": 4, "split_seed": 20260927,
                    "subset": 0, "max_steps": 0}
        changed = [name for name, value in expected.items() if getattr(args, name) != value]
        if changed or args.seed not in FORMAL_SEEDS:
            parser.error("Formal protocol is fixed; --diagnostic required for altered settings: " + ", ".join(changed))
    if args.resume and (args.mode != "train" or Path(args.resume).resolve().parent != Path(args.out).resolve()):
        parser.error("--resume must be the original run's last.pt in train mode")
    return args


def main():
    args = parse_args()
    if args.mode == "self-test":
        self_test(args)
        return
    out = Path(args.out)
    if args.mode == "finalize" and not out.is_dir():
        raise ValueError("Finalization requires an existing training output directory")
    out.mkdir(parents=True, exist_ok=True)
    def interrupted(signum, frame):
        raise InterruptedError("Training process received signal " + str(signum))

    signal.signal(signal.SIGTERM, interrupted)
    signal.signal(signal.SIGINT, interrupted)
    with directory_lock(out):
        contents = [path for path in out.iterdir() if path.name != ".run.lock"]
        if args.mode == "train" and not args.resume and contents:
            raise ValueError("New run requires an empty output directory; refusing to overwrite existing artifacts")
        if args.resume and not Path(args.resume).is_file():
            raise ValueError("Resume checkpoint does not exist")
        args._owns_run = args.mode == "train" and not args.resume
        try:
            {"train": train, "finalize": finalize}[args.mode](args)
        except Exception as exc:
            if not args._owns_run:
                raise
            atomic_json(out / "status.json", {"status": "failed", "stage": args.mode,
                                              "diagnostic": args.diagnostic,
                                              "error_type": type(exc).__name__, "error": str(exc)})
            event(out, "failed", stage=args.mode, error_type=type(exc).__name__, error=str(exc))
            raise
        finally:
            if args._owns_run and args.mode == "train" and hasattr(args, "_attempt_started"):
                atomic_json(out / "budget.json", {
                    "training_wall_s": args._previous_seconds + time.perf_counter() - args._attempt_started,
                    "max_minutes": args.max_minutes})


if __name__ == "__main__":
    main()
