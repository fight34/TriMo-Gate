# -*- coding: utf-8 -*-
"""Random-split TriMo-Gate training with per-seed Bayesian optimization.

Protocol implemented from the manuscript:
1. Randomly split the dataset into train/validation/test subsets at 8:1:1.
2. Repeat the experiment with three random seeds.
3. For each seed, run 15 Bayesian-optimization trials on the training and
   validation subsets.
4. Select the hyperparameter configuration with the lowest supervised
   validation MAE.
5. Retrain a fresh model with the selected configuration under the same split,
   then evaluate the held-out test set once.

Dependency note for GitHub/requirements.txt:
    optuna>=3.0
"""

import copy
import json
import os
import random
import re
import time
from typing import Any, Dict, Iterable, List, Optional, Tuple

import matplotlib
import numpy as np
import pandas as pd
import torch
from Model import MultimodalModel
from Graph_dataloader import edge_dim, graphs, node_dim, valid_smiles, y
from data_config import get_config
if get_config()["dataset"] == "qm8":
    from QM8_xyz_to_3d_coords import d_input, point_cloud, point_cloud_lengths
else:
    from QM9_xyz_to_3d_coords import d_input, point_cloud, point_cloud_lengths
from SMILES_dataloader_atom import indices, max_length, vocb_size
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader, Dataset, Subset
from torch_geometric.data import Batch

try:
    import optuna
except ImportError as exc:
    raise ImportError(
        "This script requires Optuna for Bayesian optimization. "
        "Install it with: pip install optuna"
    ) from exc

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


# =========================
# Basic configuration
# =========================

RESULT_DIR = get_config()["result_dir"]

SEEDS = [42, 44, 46]
TRAIN_FRAC = 0.8
VAL_FRAC = 0.1
TEST_FRAC = 0.1

BAYES_OPT_N_TRIALS = 15
BO_MAX_EPOCHS = 200
FINAL_MAX_EPOCHS = 200
BATCH_LOG_INTERVAL = 50
NUM_WORKERS = 0
PIN_MEMORY = torch.cuda.is_available()


DEFAULT_TRAINING_CONFIG: Dict[str, Any] = {
    "hidden_dim": 256,
    "smiles_num_heads": 8,
    "smiles_num_layers": 6,
    "graph_num_layers": 6,
    "d_point_cloud": 1024,
    "learning_rate": 2e-4,
    "weight_decay": 1e-4,
    "batch_size": 128,
    "scheduler_factor": 0.5,
    "scheduler_patience": 8,
    "early_stop_patience": 10,
    "early_stop_min_delta": 1e-4,
    "grad_clip_norm": 0.5,
    "adam_beta1": 0.9,
    "adam_beta2": 0.999,
}


# =========================
# Dataset and utilities
# =========================


class MultiModalDataset(Dataset):
    """Synchronized SMILES, graph, point-cloud, and target dataset."""

    def __init__(self, graph, index, point_clouds, point_cloud_length, target):
        self.graphs = graph
        self.indices = torch.tensor(index, dtype=torch.long)
        self.point_clouds = point_clouds
        self.point_cloud_length = point_cloud_length
        self.targets = torch.tensor(target, dtype=torch.float)

    def __len__(self):
        return len(self.graphs)

    def __getitem__(self, idx):
        return {
            "graph": self.graphs[idx],
            "point_cloud": self.point_clouds[idx],
            "point_cloud_length": self.point_cloud_length[idx],
            "smiles_idx": self.indices[idx],
            "target": self.targets[idx],
        }


class EarlyStopper:
    """Early stopping based on supervised validation loss."""

    def __init__(self, patience: int, min_delta: float):
        self.patience = int(patience)
        self.min_delta = float(min_delta)
        self.counter = 0
        self.best_loss = float("inf")

    def __call__(self, val_loss: float) -> bool:
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
            return False
        self.counter += 1
        print(f"EarlyStopping counter: {self.counter}/{self.patience}")
        return self.counter >= self.patience


def infer_modalities_from_result_dir(result_dir: str) -> Dict[str, bool]:
    """Infer active modalities from names such as 1D+2D+3D in RESULT_DIR."""
    result_name = os.path.basename(os.path.normpath(result_dir))
    prefix = "RandomSplit_3_Seeds_"
    suffix = "_Results"

    if not (result_name.startswith(prefix) and result_name.endswith(suffix)):
        raise ValueError(
            "RESULT_DIR name must start with 'RandomSplit_3_Seeds_' "
            "and end with '_Results'. "
            f"Got: {result_name}"
        )

    modality_part = result_name[len(prefix): -len(suffix)]
    valid_tokens = {"1D", "2D", "3D"}
    modality_tokens = set(re.findall(r"\b(?:1D|2D|3D)\b", modality_part))

    if not modality_tokens or not modality_tokens.issubset(valid_tokens):
        raise ValueError(
            f"Cannot infer modalities from RESULT_DIR={result_name}. "
            f"Allowed modality tokens are {sorted(valid_tokens)}."
        )

    return {
        "use_smiles": "1D" in modality_tokens,
        "use_graph": "2D" in modality_tokens,
        "use_point_cloud": "3D" in modality_tokens,
    }


MODALITY_FLAGS = {
    "use_smiles": "S" in get_config()["modalities"],
    "use_graph": "G" in get_config()["modalities"],
    "use_point_cloud": "P" in get_config()["modalities"],
}


def collate_fn(batchs):
    graph_batch = Batch.from_data_list([item["graph"] for item in batchs])
    return {
        "graph": graph_batch,
        "point_cloud": torch.stack([item["point_cloud"] for item in batchs]),
        "point_cloud_length": torch.stack(
            [item["point_cloud_length"] for item in batchs]
        ),
        "smiles_idx": torch.stack([item["smiles_idx"] for item in batchs]),
        "target": torch.stack([item["target"] for item in batchs]),
    }


def set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True


def seed_worker(worker_id: int) -> None:
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def random_split_indices(
    n_samples: int,
    seed: int,
    train_frac: float = 0.8,
    val_frac: float = 0.1,
    test_frac: float = 0.1,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    if not np.isclose(train_frac + val_frac + test_frac, 1.0):
        raise ValueError("train_frac + val_frac + test_frac must equal 1.0")

    if n_samples <= 0:
        raise ValueError("n_samples must be positive")

    rng = np.random.default_rng(seed)
    all_indices = np.arange(n_samples, dtype=int)
    rng.shuffle(all_indices)

    n_train = int(train_frac * n_samples)
    n_val = int(val_frac * n_samples)
    n_test = n_samples - n_train - n_val

    if n_train <= 0 or n_val <= 0 or n_test <= 0:
        raise ValueError(
            f"Invalid split sizes: train={n_train}, val={n_val}, test={n_test}. "
            "Check dataset size and split fractions."
        )

    train_idx = np.array(sorted(all_indices[:n_train]), dtype=int)
    val_idx = np.array(sorted(all_indices[n_train: n_train + n_val]), dtype=int)
    test_idx = np.array(sorted(all_indices[n_train + n_val:]), dtype=int)

    train_set, val_set, test_set = set(train_idx), set(val_idx), set(test_idx)
    assert train_set.isdisjoint(val_set)
    assert train_set.isdisjoint(test_set)
    assert val_set.isdisjoint(test_set)
    assert len(train_set | val_set | test_set) == n_samples

    return train_idx, val_idx, test_idx


def check_dataset_lengths() -> None:
    if not (
        len(graphs)
        == len(indices)
        == len(point_cloud)
        == len(point_cloud_lengths)
        == len(y)
        == len(valid_smiles)
    ):
        raise ValueError(
            "Length mismatch: "
            f"graphs={len(graphs)}, "
            f"indices={len(indices)}, "
            f"point_cloud={len(point_cloud)}, "
            f"point_cloud_lengths={len(point_cloud_lengths)}, "
            f"y={len(y)}, "
            f"valid_smiles={len(valid_smiles)}"
        )


def make_dataset(target_values: Iterable[float]) -> MultiModalDataset:
    return MultiModalDataset(
        graph=graphs,
        index=indices,
        point_clouds=point_cloud,
        point_cloud_length=point_cloud_lengths,
        target=target_values,
    )


def make_dataloaders(
    dataset: MultiModalDataset,
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    test_idx: np.ndarray,
    batch_size: int,
    seed: int,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    generator = torch.Generator()
    generator.manual_seed(seed)

    train_loader = DataLoader(
        Subset(dataset, train_idx),
        batch_size=int(batch_size),
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
        worker_init_fn=seed_worker,
        generator=generator,
    )
    val_loader = DataLoader(
        Subset(dataset, val_idx),
        batch_size=int(batch_size),
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
        worker_init_fn=seed_worker,
    )
    test_loader = DataLoader(
        Subset(dataset, test_idx),
        batch_size=int(batch_size),
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
        worker_init_fn=seed_worker,
    )
    return train_loader, val_loader, test_loader


# =========================
# Bayesian-optimization setup
# =========================


def suggest_hyperparameters(trial: "optuna.Trial") -> Dict[str, Any]:
    """Hyperparameter search space for the 15 BO iterations per seed."""
    hidden_dim = trial.suggest_categorical("hidden_dim", [128, 256, 512])

    return {
        "hidden_dim": hidden_dim,
        "smiles_num_heads": trial.suggest_categorical(
            "smiles_num_heads", [4, 8]
        ),
        "smiles_num_layers": trial.suggest_categorical(
            "smiles_num_layers", [3, 4, 6]
        ),
        "graph_num_layers": trial.suggest_categorical(
            "graph_num_layers", [3, 4, 6]
        ),
        "d_point_cloud": trial.suggest_categorical("d_point_cloud", [512, 1024]),
        "learning_rate": trial.suggest_float(
            "learning_rate", 1e-5, 1e-3, log=True
        ),
        "weight_decay": trial.suggest_float("weight_decay", 1e-7, 1e-3, log=True),
        "batch_size": trial.suggest_categorical("batch_size", [64, 128, 256]),
        "scheduler_factor": trial.suggest_categorical(
            "scheduler_factor", [0.3, 0.5, 0.7]
        ),
        "scheduler_patience": trial.suggest_categorical(
            "scheduler_patience", [5, 8, 12]
        ),
        "early_stop_patience": trial.suggest_categorical(
            "early_stop_patience", [8, 10, 15]
        ),
        "grad_clip_norm": trial.suggest_categorical(
            "grad_clip_norm", [0.3, 0.5, 1.0]
        ),
    }


def build_training_config(params: Dict[str, Any], max_epochs: int) -> Dict[str, Any]:
    config = copy.deepcopy(DEFAULT_TRAINING_CONFIG)
    config.update(params)
    config["max_epochs"] = int(max_epochs)
    return config


def make_json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): make_json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [make_json_safe(v) for v in value]
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    return value


# =========================
# Model, training, and evaluation
# =========================


def make_model(config: Dict[str, Any]) -> MultimodalModel:
    hidden_dim = int(config["hidden_dim"])
    return MultimodalModel(
        vocab_size=vocb_size,
        smiles_length=max_length,
        d_smiles=hidden_dim,
        smiles_num_heads=int(config["smiles_num_heads"]),
        smiles_num_layers=int(config["smiles_num_layers"]),
        cnn_kernels=hidden_dim,
        d_node=node_dim,
        d_edge=edge_dim,
        d_hidden=hidden_dim,
        graph_num_layers=int(config["graph_num_layers"]),
        d_point_input=d_input,
        d_point_cloud=int(config["d_point_cloud"]),
        d_point_output=hidden_dim,
        d_output=1,
        use_smiles=MODALITY_FLAGS["use_smiles"],
        use_graph=MODALITY_FLAGS["use_graph"],
        use_point_cloud=MODALITY_FLAGS["use_point_cloud"],
        fusion_type=get_config()["fusion"],
    )


def count_parameters(model: torch.nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def print_model_parameter_report(model: torch.nn.Module) -> None:
    print(f"Total trainable parameters: {count_parameters(model):,}")
    print("\nStatistics of parameters for each module:")
    for name, module in model.named_children():
        module_params = sum(p.numel() for p in module.parameters() if p.requires_grad)
        print(f"{name}: {module_params:,} parameters")

    print("\nDetailed parameter statistics:")
    total_params = 0
    for name, param in model.named_parameters():
        if param.requires_grad:
            print(f"{name}: {param.numel():,} parameters")
            total_params += param.numel()
    print(f"Detailed total: {total_params:,} parameters")


def move_batch_to_device(batch: Dict[str, Any], device: torch.device) -> Tuple:
    smiles_batch = batch["smiles_idx"].to(device)
    graphs_batch = batch["graph"].to(device)
    point_cloud_batch = batch["point_cloud"].to(device)
    point_cloud_length_batch = batch["point_cloud_length"].to(device)
    targets_batch = batch["target"].to(device)
    return (
        smiles_batch,
        graphs_batch,
        point_cloud_batch,
        point_cloud_length_batch,
        targets_batch,
    )


def train_one_model(
    train_loader: DataLoader,
    val_loader: DataLoader,
    config: Dict[str, Any],
    device: torch.device,
    seed: int,
    run_name: str,
    trial: Optional["optuna.Trial"] = None,
    report_model: bool = False,
) -> Dict[str, Any]:
    """Train one model and return the best validation checkpoint."""
    set_global_seed(seed)
    model = make_model(config).to(device)

    if report_model:
        print_model_parameter_report(model)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(config["learning_rate"]),
        weight_decay=float(config["weight_decay"]),
        betas=(float(config["adam_beta1"]), float(config["adam_beta2"])),
    )
    criterion = torch.nn.L1Loss()
    scheduler = ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=float(config["scheduler_factor"]),
        patience=int(config["scheduler_patience"]),
    )
    early_stopper = EarlyStopper(
        patience=int(config["early_stop_patience"]),
        min_delta=float(config["early_stop_min_delta"]),
    )

    start_time = time.time()
    train_history = {"loss": [], "val_loss": []}
    best_val_loss = float("inf")
    best_model_state = None
    best_epoch = 0
    max_epochs = int(config["max_epochs"])

    for epoch in range(max_epochs):
        model.train()
        train_loss = 0.0
        train_batches = 0

        for batch_idx, batch in enumerate(train_loader):
            optimizer.zero_grad(set_to_none=True)
            (
                smiles_batch,
                graphs_batch,
                point_cloud_batch,
                point_cloud_length_batch,
                targets_batch,
            ) = move_batch_to_device(batch, device)

            output = model(
                smiles_batch,
                graphs_batch,
                point_cloud_batch,
                point_cloud_length_batch,
            )
            loss = criterion(output.view(-1), targets_batch)
            loss.backward()

            torch.nn.utils.clip_grad_norm_(
                model.parameters(), max_norm=float(config["grad_clip_norm"])
            )
            optimizer.step()

            train_loss += float(loss.item())
            train_batches += 1

            should_log_batch = (
                batch_idx == 0
                or (batch_idx + 1) % BATCH_LOG_INTERVAL == 0
                or (batch_idx + 1) == len(train_loader)
            )
            if should_log_batch:
                print(
                    f"{run_name} | Epoch {epoch + 1}/{max_epochs} | "
                    f"Batch {batch_idx + 1}/{len(train_loader)} | "
                    f"Loss: {loss.item():.6f}"
                )

        model.eval()
        val_loss = 0.0
        val_batches = 0
        with torch.no_grad():
            for batch in val_loader:
                (
                    smiles_batch,
                    graphs_batch,
                    point_cloud_batch,
                    point_cloud_length_batch,
                    targets_batch,
                ) = move_batch_to_device(batch, device)

                output = model(
                    smiles_batch,
                    graphs_batch,
                    point_cloud_batch,
                    point_cloud_length_batch,
                )
                loss = criterion(output.view(-1), targets_batch)
                val_loss += float(loss.item())
                val_batches += 1

        avg_train_loss = train_loss / max(train_batches, 1)
        avg_val_loss = val_loss / max(val_batches, 1)
        scheduler.step(avg_val_loss)
        current_lr = optimizer.param_groups[0]["lr"]
        train_history["loss"].append(avg_train_loss)
        train_history["val_loss"].append(avg_val_loss)

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_model_state = copy.deepcopy(model.state_dict())
            best_epoch = epoch

        print(
            f"{run_name} | Epoch {epoch + 1}/{max_epochs} | "
            f"Train Loss: {avg_train_loss:.6f} | "
            f"Val Loss: {avg_val_loss:.6f} | LR: {current_lr:.2e}"
        )

        if trial is not None:
            trial.report(avg_val_loss, epoch)
            if trial.should_prune():
                raise optuna.TrialPruned(
                    f"Pruned at epoch {epoch + 1} with val_loss={avg_val_loss:.6f}"
                )

        if early_stopper(avg_val_loss):
            print(f"Early stopping triggered at epoch {epoch + 1}")
            break

    training_time_min = (time.time() - start_time) / 60.0

    if best_model_state is None:
        best_model_state = copy.deepcopy(model.state_dict())

    model.load_state_dict(best_model_state)
    model.eval()

    return {
        "model": model,
        "best_model_state": best_model_state,
        "best_val_loss": float(best_val_loss),
        "best_epoch": int(best_epoch + 1),
        "history": train_history,
        "training_time_min": float(training_time_min),
    }


def evaluate_model(
    model: torch.nn.Module,
    test_loader: DataLoader,
    device: torch.device,
    train_mean: float,
    train_std: float,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, float]]:
    seed_targets, seed_predictions = [], []

    model.eval()
    with torch.no_grad():
        for batch in test_loader:
            (
                smiles_batch,
                graphs_batch,
                point_cloud_batch,
                point_cloud_length_batch,
                targets_batch,
            ) = move_batch_to_device(batch, device)

            output = model(
                smiles_batch,
                graphs_batch,
                point_cloud_batch,
                point_cloud_length_batch,
            )
            seed_predictions.append(output.view(-1).detach().cpu())
            seed_targets.append(targets_batch.detach().cpu())

    predictions_norm = torch.cat(seed_predictions).numpy()
    targets_norm = torch.cat(seed_targets).numpy()

    predictions_np = predictions_norm * train_std + train_mean
    targets_np = targets_norm * train_std + train_mean

    metrics = {
        "MAE": float(mean_absolute_error(targets_np, predictions_np)),
        "RMSE": float(np.sqrt(mean_squared_error(targets_np, predictions_np))),
        "R2": float(r2_score(targets_np, predictions_np)),
    }
    return targets_np, predictions_np, metrics


def run_bayesian_optimization_for_seed(
    seed: int,
    dataset_seed: MultiModalDataset,
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    test_idx: np.ndarray,
    device: torch.device,
    result_dir: str,
) -> Tuple[Dict[str, Any], float, float]:
    """Run exactly BAYES_OPT_N_TRIALS Optuna/TPE trials for one split."""
    sampler = optuna.samplers.TPESampler(
        seed=seed,
        n_startup_trials=min(5, BAYES_OPT_N_TRIALS),
        multivariate=True,
        group=True,
    )
    pruner = optuna.pruners.MedianPruner(
        n_startup_trials=min(5, BAYES_OPT_N_TRIALS),
        n_warmup_steps=10,
    )
    study = optuna.create_study(
        direction="minimize",
        sampler=sampler,
        pruner=pruner,
        study_name=f"trimogate_seed_{seed}",
    )

    bo_start = time.time()

    def objective(trial: "optuna.Trial") -> float:
        params = suggest_hyperparameters(trial)
        config = build_training_config(params, max_epochs=BO_MAX_EPOCHS)

        train_loader, val_loader, _ = make_dataloaders(
            dataset=dataset_seed,
            train_idx=train_idx,
            val_idx=val_idx,
            test_idx=test_idx,
            batch_size=int(config["batch_size"]),
            seed=seed + trial.number,
        )

        run_name = f"Seed {seed} | BO Trial {trial.number + 1}/{BAYES_OPT_N_TRIALS}"
        try:
            result = train_one_model(
                train_loader=train_loader,
                val_loader=val_loader,
                config=config,
                device=device,
                seed=seed + trial.number,
                run_name=run_name,
                trial=trial,
                report_model=False,
            )
            return float(result["best_val_loss"])
        finally:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    study.optimize(objective, n_trials=BAYES_OPT_N_TRIALS, gc_after_trial=True)
    bo_time_min = (time.time() - bo_start) / 60.0

    trials_df = study.trials_dataframe()
    trials_df.to_excel(
        os.path.join(result_dir, f"bayesian_optimization_trials_seed_{seed}.xlsx"),
        index=False,
    )

    best_config = build_training_config(study.best_params, max_epochs=FINAL_MAX_EPOCHS)
    best_value = float(study.best_value)
    best_payload = {
        "seed": seed,
        "bayes_opt_n_trials": BAYES_OPT_N_TRIALS,
        "bo_max_epochs": BO_MAX_EPOCHS,
        "final_max_epochs": FINAL_MAX_EPOCHS,
        "best_supervised_val_loss": best_value,
        "best_params": make_json_safe(study.best_params),
        "best_final_training_config": make_json_safe(best_config),
    }
    with open(
        os.path.join(result_dir, f"best_hyperparameters_seed_{seed}.json"),
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(best_payload, f, indent=2, ensure_ascii=False)

    return best_config, best_value, float(bo_time_min)


# =========================
# Reporting and visualization
# =========================


def save_split_file(
    seed: int,
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    test_idx: np.ndarray,
    result_dir: str,
) -> pd.DataFrame:
    split_name_by_idx = {}
    for idx in train_idx:
        split_name_by_idx[int(idx)] = "train"
    for idx in val_idx:
        split_name_by_idx[int(idx)] = "val"
    for idx in test_idx:
        split_name_by_idx[int(idx)] = "test"

    split_df = pd.DataFrame(
        {
            "Seed": seed,
            "Index": list(range(len(valid_smiles))),
            "SMILES": valid_smiles,
            "Split": [split_name_by_idx[i] for i in range(len(valid_smiles))],
        }
    )
    split_df.to_excel(
        os.path.join(result_dir, f"random_split_seed_{seed}.xlsx"), index=False
    )
    return split_df


def save_history_file(
    seed: int,
    history: Dict[str, List[float]],
    result_dir: str,
) -> None:
    history_df = pd.DataFrame(
        {
            "Epoch": range(1, len(history["loss"]) + 1),
            "Train_Loss": history["loss"],
            "Val_Loss": history["val_loss"],
        }
    )
    history_df.to_excel(
        os.path.join(result_dir, f"training_history_seed_{seed}.xlsx"), index=False
    )


def save_prediction_file(
    seed: int,
    seed_smiles: List[str],
    targets_np: np.ndarray,
    predictions_np: np.ndarray,
    result_dir: str,
) -> None:
    pred_df = pd.DataFrame(
        {
            "SMILES": seed_smiles,
            "True_Values": targets_np,
            "Predicted_Values": predictions_np,
        }
    )
    pred_df.to_excel(
        os.path.join(result_dir, f"predictions_seed_{seed}.xlsx"), index=False
    )


def save_publication_figures(
    all_results_df: pd.DataFrame,
    fold_metrics: Dict[str, List[Any]],
    result_dir: str,
) -> None:
    plt.rcParams.update(
        {
            "font.family": "Arial",
            "mathtext.fontset": "stix",
            "axes.linewidth": 1.2,
            "axes.labelsize": 14,
            "axes.titlesize": 15,
            "xtick.labelsize": 12,
            "ytick.labelsize": 12,
            "legend.fontsize": 11,
            "figure.dpi": 300,
            "savefig.dpi": 600,
            "savefig.bbox": "tight",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": False,
        }
    )

    true_values = all_results_df["True_Values"].to_numpy(dtype=float)
    pred_values = all_results_df["Predicted_Values"].to_numpy(dtype=float)
    residuals = pred_values - true_values

    overall_mae = mean_absolute_error(true_values, pred_values)
    overall_rmse = np.sqrt(mean_squared_error(true_values, pred_values))
    overall_r2 = r2_score(true_values, pred_values)

    min_val = min(true_values.min(), pred_values.min())
    max_val = max(true_values.max(), pred_values.max())
    pad = 0.04 * (max_val - min_val)
    plot_min = min_val - pad
    plot_max = max_val + pad

    fig, axes = plt.subplots(2, 2, figsize=(12.5, 10.5))
    ax1, ax2, ax3, ax4 = axes.flatten()

    hb = ax1.hexbin(
        true_values,
        pred_values,
        gridsize=70,
        mincnt=1,
        bins="log",
        cmap="viridis",
        linewidths=0.0,
    )
    ax1.plot(
        [plot_min, plot_max],
        [plot_min, plot_max],
        linestyle="--",
        linewidth=1.8,
        color="black",
        alpha=0.85,
        label="Ideal",
    )

    coef = np.polyfit(true_values, pred_values, 1)
    fit_line = np.poly1d(coef)
    x_fit = np.linspace(plot_min, plot_max, 200)
    ax1.plot(
        x_fit,
        fit_line(x_fit),
        linewidth=1.8,
        color="crimson",
        alpha=0.9,
        label=f"Fit: y = {coef[0]:.3f}x + {coef[1]:.3f}",
    )

    ax1.set_xlim(plot_min, plot_max)
    ax1.set_ylim(plot_min, plot_max)
    ax1.set_aspect("equal", adjustable="box")
    ax1.set_xlabel("True value")
    ax1.set_ylabel("Predicted value")
    ax1.set_title("a) Prediction parity", loc="left", fontweight="bold")
    ax1.legend(frameon=False, loc="lower right")

    stats_text = (
        f"N = {len(true_values):,}\n"
        f"MAE = {overall_mae:.4f}\n"
        f"RMSE = {overall_rmse:.4f}\n"
        f"$R^2$ = {overall_r2:.4f}"
    )
    ax1.text(
        0.04,
        0.96,
        stats_text,
        transform=ax1.transAxes,
        ha="left",
        va="top",
        fontsize=11.5,
        bbox=dict(
            boxstyle="round,pad=0.35",
            facecolor="white",
            edgecolor="0.75",
            alpha=0.92,
        ),
    )

    cb1 = fig.colorbar(hb, ax=ax1, fraction=0.046, pad=0.025)
    cb1.set_label("log$_{10}$(point count)", fontsize=12)

    hb2 = ax2.hexbin(
        true_values,
        residuals,
        gridsize=70,
        mincnt=1,
        bins="log",
        cmap="magma",
        linewidths=0.0,
    )
    ax2.axhline(0, linestyle="--", linewidth=1.6, color="black", alpha=0.85)
    res_std = np.std(residuals)
    ax2.axhline(2 * res_std, linestyle=":", linewidth=1.2, color="gray")
    ax2.axhline(-2 * res_std, linestyle=":", linewidth=1.2, color="gray")
    ax2.set_xlabel("True value")
    ax2.set_ylabel("Residual (Predicted - True)")
    ax2.set_title("b) Residual structure", loc="left", fontweight="bold")
    ax2.text(
        0.04,
        0.96,
        f"Mean residual = {np.mean(residuals):.4f}\nSD residual = {res_std:.4f}",
        transform=ax2.transAxes,
        ha="left",
        va="top",
        fontsize=11.5,
        bbox=dict(
            boxstyle="round,pad=0.35",
            facecolor="white",
            edgecolor="0.75",
            alpha=0.92,
        ),
    )
    cb2 = fig.colorbar(hb2, ax=ax2, fraction=0.046, pad=0.025)
    cb2.set_label("log$_{10}$(point count)", fontsize=12)

    ax3.hist(
        residuals,
        bins=80,
        density=True,
        alpha=0.82,
        edgecolor="white",
        linewidth=0.3,
    )
    ax3.axvline(0, linestyle="--", linewidth=1.6, color="black", alpha=0.85)
    ax3.axvline(
        np.mean(residuals), linestyle="-", linewidth=1.6, color="crimson", alpha=0.9
    )
    ax3.set_xlabel("Residual (Predicted - True)")
    ax3.set_ylabel("Density")
    ax3.set_title("c) Residual distribution", loc="left", fontweight="bold")
    ax3.text(
        0.04,
        0.96,
        f"Median = {np.median(residuals):.4f}\n"
        f"Q$_1$ = {np.quantile(residuals, 0.25):.4f}\n"
        f"Q$_3$ = {np.quantile(residuals, 0.75):.4f}",
        transform=ax3.transAxes,
        ha="left",
        va="top",
        fontsize=11.5,
        bbox=dict(
            boxstyle="round,pad=0.35",
            facecolor="white",
            edgecolor="0.75",
            alpha=0.92,
        ),
    )

    seed_labels = [str(s) for s in fold_metrics["Seed"]]
    x = np.arange(len(seed_labels))
    mae_values = np.array(fold_metrics["MAE"], dtype=float)
    rmse_values = np.array(fold_metrics["RMSE"], dtype=float)
    r2_values = np.array(fold_metrics["R2"], dtype=float)
    width = 0.36

    ax4.bar(
        x - width / 2,
        mae_values,
        width=width,
        label="MAE",
        alpha=0.88,
        edgecolor="white",
        linewidth=0.8,
    )
    ax4.bar(
        x + width / 2,
        rmse_values,
        width=width,
        label="RMSE",
        alpha=0.88,
        edgecolor="white",
        linewidth=0.8,
    )
    ax4.set_xticks(x)
    ax4.set_xticklabels(seed_labels)
    ax4.set_xlabel("Random seed")
    ax4.set_ylabel("Error")
    ax4.set_title("d) Seed-wise test performance", loc="left", fontweight="bold")

    ax4_r = ax4.twinx()
    ax4_r.plot(
        x,
        r2_values,
        marker="o",
        linewidth=2.0,
        markersize=6,
        color="black",
        label="$R^2$",
    )
    ax4_r.set_ylabel("$R^2$")

    lines1, labels1 = ax4.get_legend_handles_labels()
    lines2, labels2 = ax4_r.get_legend_handles_labels()
    ax4.legend(lines1 + lines2, labels1 + labels2, frameon=False, loc="best")

    for i, value in enumerate(mae_values):
        ax4.text(i - width / 2, value, f"{value:.3f}", ha="center", va="bottom", fontsize=9)
    for i, value in enumerate(rmse_values):
        ax4.text(i + width / 2, value, f"{value:.3f}", ha="center", va="bottom", fontsize=9)

    fig.suptitle(
        "QM9 prediction performance across random splits with Bayesian optimization",
        fontsize=17,
        fontweight="bold",
        y=0.995,
    )
    plt.tight_layout(rect=(0.0, 0.0, 1.0, 0.975))
    fig.savefig(os.path.join(result_dir, "publication_performance_summary.png"), dpi=600)
    fig.savefig(os.path.join(result_dir, "publication_performance_summary.pdf"))
    plt.close(fig)

    fig_single, ax = plt.subplots(figsize=(7.2, 6.8))
    hb_single = ax.hexbin(
        true_values,
        pred_values,
        gridsize=80,
        mincnt=1,
        bins="log",
        cmap="viridis",
        linewidths=0.0,
    )
    ax.plot(
        [plot_min, plot_max],
        [plot_min, plot_max],
        linestyle="--",
        linewidth=1.8,
        color="black",
        alpha=0.85,
    )
    ax.plot(x_fit, fit_line(x_fit), linewidth=1.8, color="crimson", alpha=0.9)
    ax.set_xlim(plot_min, plot_max)
    ax.set_ylim(plot_min, plot_max)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("True value", fontsize=14)
    ax.set_ylabel("Predicted value", fontsize=14)
    ax.text(
        0.04,
        0.96,
        stats_text,
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=12,
        bbox=dict(
            boxstyle="round,pad=0.35",
            facecolor="white",
            edgecolor="0.75",
            alpha=0.92,
        ),
    )
    cb = fig_single.colorbar(hb_single, ax=ax, fraction=0.046, pad=0.025)
    cb.set_label("log$_{10}$(point count)", fontsize=12)
    plt.tight_layout()
    fig_single.savefig(os.path.join(result_dir, "publication_parity_plot.png"), dpi=600)
    fig_single.savefig(os.path.join(result_dir, "publication_parity_plot.pdf"))
    plt.close(fig_single)

    print("Publication-quality figures saved:")
    print(os.path.join(result_dir, "publication_performance_summary.png"))
    print(os.path.join(result_dir, "publication_performance_summary.pdf"))
    print(os.path.join(result_dir, "publication_parity_plot.png"))
    print(os.path.join(result_dir, "publication_parity_plot.pdf"))


def print_target_stats(name: str, values: np.ndarray) -> None:
    print(
        f"{name} y stats:",
        values.min(),
        values.max(),
        values.mean(),
        values.std(),
        np.mean(np.abs(values)),
    )


# =========================
# Main experiment
# =========================


def main() -> None:
    print(f"RESULT_DIR: {RESULT_DIR}")
    print(
        "Inferred modalities: "
        f"use_smiles={MODALITY_FLAGS['use_smiles']}, "
        f"use_graph={MODALITY_FLAGS['use_graph']}, "
        f"use_point_cloud={MODALITY_FLAGS['use_point_cloud']}"
    )
    print(
        "Bayesian optimization protocol: "
        f"{BAYES_OPT_N_TRIALS} trials per seed, "
        f"BO_MAX_EPOCHS={BO_MAX_EPOCHS}, "
        f"FINAL_MAX_EPOCHS={FINAL_MAX_EPOCHS}"
    )

    os.makedirs(RESULT_DIR, exist_ok=True)
    check_dataset_lengths()

    raw_dataset = make_dataset(y)
    y_np = np.asarray(y, dtype=float)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    fold_metrics: Dict[str, List[Any]] = {
        "Seed": [],
        "Train_Size": [],
        "Val_Size": [],
        "Test_Size": [],
        "MAE": [],
        "RMSE": [],
        "R2": [],
        "BO_Best_Val_Loss": [],
        "Final_Best_Val_Loss": [],
        "Best_Epoch": [],
        "BO_Time(min)": [],
        "Training_Time(min)": [],
        "Total_Time(min)": [],
        "Best_Config": [],
    }

    all_smiles: List[str] = []
    all_predictions: List[float] = []
    all_true_values: List[float] = []
    all_split_records: List[pd.DataFrame] = []
    all_best_configs: List[Dict[str, Any]] = []

    for seed_idx, seed in enumerate(SEEDS):
        set_global_seed(seed)

        print("\n" + "=" * 60)
        print(f"Random Split Seed {seed} ({seed_idx + 1}/{len(SEEDS)})")
        print("=" * 60)

        train_idx, val_idx, test_idx = random_split_indices(
            n_samples=len(raw_dataset),
            seed=seed,
            train_frac=TRAIN_FRAC,
            val_frac=VAL_FRAC,
            test_frac=TEST_FRAC,
        )
        print(f"Train/Val/Test sizes: {len(train_idx)}/{len(val_idx)}/{len(test_idx)}")

        print_target_stats("Train", y_np[train_idx])
        print_target_stats("Val", y_np[val_idx])
        print_target_stats("Test", y_np[test_idx])

        train_mean = float(y_np[train_idx].mean())
        train_std = float(y_np[train_idx].std())
        if train_std < 1e-12:
            raise ValueError("Training target std is too small; cannot standardize target.")

        y_norm = (y_np - train_mean) / train_std
        dataset_seed = make_dataset(y_norm)

        print(f"Target standardization: mean={train_mean:.6f}, std={train_std:.6f}")
        print(
            "Mean predictor MAE:",
            "train=", np.mean(np.abs(y_np[train_idx] - train_mean)),
            "val=", np.mean(np.abs(y_np[val_idx] - train_mean)),
            "test=", np.mean(np.abs(y_np[test_idx] - train_mean)),
        )

        split_df = save_split_file(seed, train_idx, val_idx, test_idx, RESULT_DIR)
        all_split_records.append(split_df)

        best_config, bo_best_value, bo_time_min = run_bayesian_optimization_for_seed(
            seed=seed,
            dataset_seed=dataset_seed,
            train_idx=train_idx,
            val_idx=val_idx,
            test_idx=test_idx,
            device=device,
            result_dir=RESULT_DIR,
        )
        all_best_configs.append(
            {
                "Seed": seed,
                "BO_Best_Val_Loss": bo_best_value,
                **make_json_safe(best_config),
            }
        )

        print("\nSelected hyperparameters for final retraining:")
        print(json.dumps(make_json_safe(best_config), indent=2, ensure_ascii=False))

        train_loader, val_loader, test_loader = make_dataloaders(
            dataset=dataset_seed,
            train_idx=train_idx,
            val_idx=val_idx,
            test_idx=test_idx,
            batch_size=int(best_config["batch_size"]),
            seed=seed,
        )

        final_run_name = f"Seed {seed} | Final retraining"
        final_result = train_one_model(
            train_loader=train_loader,
            val_loader=val_loader,
            config=best_config,
            device=device,
            seed=seed,
            run_name=final_run_name,
            trial=None,
            report_model=(seed_idx == 0),
        )

        targets_np, predictions_np, metrics = evaluate_model(
            model=final_result["model"],
            test_loader=test_loader,
            device=device,
            train_mean=train_mean,
            train_std=train_std,
        )

        seed_smiles = [valid_smiles[i] for i in test_idx]
        all_smiles.extend(seed_smiles)
        all_predictions.extend(predictions_np.tolist())
        all_true_values.extend(targets_np.tolist())

        save_prediction_file(seed, seed_smiles, targets_np, predictions_np, RESULT_DIR)
        save_history_file(seed, final_result["history"], RESULT_DIR)

        total_time_min = bo_time_min + float(final_result["training_time_min"])
        fold_metrics["Seed"].append(seed)
        fold_metrics["Train_Size"].append(len(train_idx))
        fold_metrics["Val_Size"].append(len(val_idx))
        fold_metrics["Test_Size"].append(len(test_idx))
        fold_metrics["MAE"].append(metrics["MAE"])
        fold_metrics["RMSE"].append(metrics["RMSE"])
        fold_metrics["R2"].append(metrics["R2"])
        fold_metrics["BO_Best_Val_Loss"].append(bo_best_value)
        fold_metrics["Final_Best_Val_Loss"].append(final_result["best_val_loss"])
        fold_metrics["Best_Epoch"].append(final_result["best_epoch"])
        fold_metrics["BO_Time(min)"].append(bo_time_min)
        fold_metrics["Training_Time(min)"].append(final_result["training_time_min"])
        fold_metrics["Total_Time(min)"].append(total_time_min)
        fold_metrics["Best_Config"].append(json.dumps(make_json_safe(best_config)))

        print(f"\nSeed {seed} Test Results:")
        print(
            f"MAE: {metrics['MAE']:.4f}, "
            f"RMSE: {metrics['RMSE']:.4f}, "
            f"R2: {metrics['R2']:.4f}, "
            f"BO time: {bo_time_min:.2f} min, "
            f"final training time: {final_result['training_time_min']:.2f} min"
        )

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    fold_metrics_df = pd.DataFrame(fold_metrics)
    fold_metrics_df.to_excel(
        os.path.join(RESULT_DIR, "random_3_seeds_metrics.xlsx"), index=False
    )

    best_configs_df = pd.DataFrame(all_best_configs)
    best_configs_df.to_excel(
        os.path.join(RESULT_DIR, "best_hyperparameters_all_seeds.xlsx"), index=False
    )

    all_splits_df = pd.concat(all_split_records, ignore_index=True)
    all_splits_df.to_excel(os.path.join(RESULT_DIR, "all_random_splits.xlsx"), index=False)

    avg_mae = float(np.mean(fold_metrics["MAE"]))
    std_mae = float(np.std(fold_metrics["MAE"]))
    avg_rmse = float(np.mean(fold_metrics["RMSE"]))
    std_rmse = float(np.std(fold_metrics["RMSE"]))
    avg_r2 = float(np.mean(fold_metrics["R2"]))
    std_r2 = float(np.std(fold_metrics["R2"]))

    print("\n" + "=" * 60)
    print("Random Split 3-Seed Summary with Bayesian Optimization")
    print("=" * 60)
    print(f"Average MAE: {avg_mae:.4f} +/- {std_mae:.4f}")
    print(f"Average RMSE: {avg_rmse:.4f} +/- {std_rmse:.4f}")
    print(f"Average R2: {avg_r2:.4f} +/- {std_r2:.4f}")
    print(
        "Average final training time per seed: "
        f"{np.mean(fold_metrics['Training_Time(min)']):.2f} min"
    )
    print(
        "Average BO time per seed: "
        f"{np.mean(fold_metrics['BO_Time(min)']):.2f} min"
    )

    all_results_df = pd.DataFrame(
        {
            "SMILES": all_smiles,
            "True_Values": all_true_values,
            "Predicted_Values": all_predictions,
        }
    )[["SMILES", "True_Values", "Predicted_Values"]]
    all_results_df.to_excel(os.path.join(RESULT_DIR, "all_predictions.xlsx"), index=False)

    save_publication_figures(all_results_df, fold_metrics, RESULT_DIR)


if __name__ == "__main__":
    main()


