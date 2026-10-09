import os
import re
import time
import copy
import torch
import random
import matplotlib
import numpy as np
import pandas as pd
import torch.nn.functional as F  # noqa
from Model import MultimodalModel
from torch_geometric.data import Batch
from torch.optim.lr_scheduler import ReduceLROnPlateau
from Graph_dataloader import graphs, y, node_dim, edge_dim, valid_smiles
from SMILES_dataloader_atom import indices, vocb_size, max_length
from torch.utils.data import DataLoader, Dataset, Subset
from data_config import get_config
if get_config()["dataset"] == "qm8":
    from QM8_xyz_to_3d_coords import d_input, point_cloud, point_cloud_lengths
else:
    from QM9_xyz_to_3d_coords import d_input, point_cloud, point_cloud_lengths
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


# =========================
# Basic configuration
# =========================


RESULT_DIR = get_config()["result_dir"]


def infer_modalities_from_result_dir(result_dir):
    result_name = os.path.basename(os.path.normpath(result_dir))
    prefix = "RandomSplit_3_Seeds_"
    suffix = "_Results"

    if not (result_name.startswith(prefix) and result_name.endswith(suffix)):
        raise ValueError(
            "RESULT_DIR name must look like "
            "RandomSplit_3_Seeds_1D+2D+3D_Results, "
            "RandomSplit_3_Seeds_1D+2D_Results, or RandomSplit_3_Seeds_3D_Results. "
            f"Got: {result_name}"
        )

    modality_part = result_name[len(prefix):-len(suffix)]
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
print(f"RESULT_DIR: {RESULT_DIR}")
print(
    "Inferred modalities: "
    f"use_smiles={MODALITY_FLAGS['use_smiles']}, "
    f"use_graph={MODALITY_FLAGS['use_graph']}, "
    f"use_point_cloud={MODALITY_FLAGS['use_point_cloud']}"
)

os.makedirs(RESULT_DIR, exist_ok=True)

SEEDS = [42, 44, 46]
# SEEDS = [42]
TRAIN_FRAC = 0.8
VAL_FRAC = 0.1
TEST_FRAC = 0.1


class MultiModalDataset(Dataset):
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
            'graph': self.graphs[idx],
            'point_cloud': self.point_clouds[idx],
            'point_cloud_length': self.point_cloud_length[idx],
            'smiles_idx': self.indices[idx],
            'target': self.targets[idx]
        }


class EarlyStopper:
    def __init__(self, patience, min_delta):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = float('inf')

    def __call__(self, val_loss):
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
            return False
        self.counter += 1
        print(f"EarlyStopping counter: {self.counter}/{self.patience}")
        return self.counter >= self.patience


def collate_fn(batchs):
    graph_batch = Batch.from_data_list([item['graph'] for item in batchs])
    return {
        'graph': graph_batch,
        'point_cloud': torch.stack([item['point_cloud'] for item in batchs]),
        'point_cloud_length': torch.stack([item['point_cloud_length'] for item in batchs]),
        'smiles_idx': torch.stack([item['smiles_idx'] for item in batchs]),
        'target': torch.stack([item['target'] for item in batchs])
    }


def rmse_loss(pred, target):
    mse = torch.nn.MSELoss()(pred, target)
    return torch.sqrt(mse)


def set_global_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True


def random_split_indices(n_samples, seed, train_frac=0.8, val_frac=0.1, test_frac=0.1):
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
    val_idx = np.array(sorted(all_indices[n_train:n_train + n_val]), dtype=int)
    test_idx = np.array(sorted(all_indices[n_train + n_val:]), dtype=int)

    # Verify no index leakage.
    train_set, val_set, test_set = set(train_idx), set(val_idx), set(test_idx)
    assert train_set.isdisjoint(val_set)
    assert train_set.isdisjoint(test_set)
    assert val_set.isdisjoint(test_set)
    assert len(train_set | val_set | test_set) == n_samples

    return train_idx, val_idx, test_idx


# =========================
# Dataset
# =========================
if not (len(graphs) == len(indices) == len(point_cloud) == len(point_cloud_lengths) == len(y) == len(valid_smiles)):
    raise ValueError(
        "Length mismatch: "
        f"graphs={len(graphs)}, "
        f"indices={len(indices)}, "
        f"point_cloud={len(point_cloud)}, "
        f"point_cloud_lengths={len(point_cloud_lengths)}, "
        f"y={len(y)}, "
        f"valid_smiles={len(valid_smiles)}"
    )

dataset = MultiModalDataset(
    graph=graphs,
    index=indices,
    point_clouds=point_cloud,
    point_cloud_length=point_cloud_lengths,
    target=y
)

fold_metrics = {
    'Seed': [],
    'Train_Size': [],
    'Val_Size': [],
    'Test_Size': [],
    'MAE': [],
    'RMSE': [],
    'R2': [],
    'Best_Epoch': [],
    'Training_Time(min)': []
}

all_smiles = []
all_predictions = []
all_true_values = []
seed_results = []
all_split_records = []


# =========================
# Random split + 3 seeds
# =========================
for seed_idx, seed in enumerate(SEEDS):
    set_global_seed(seed)

    print(f"\n{'=' * 50}")
    print(f"Random Split Seed {seed} ({seed_idx + 1}/{len(SEEDS)})")
    print(f"{'=' * 50}")

    train_idx, val_idx, test_idx = random_split_indices(
        n_samples=len(dataset),
        seed=seed,
        train_frac=TRAIN_FRAC,
        val_frac=VAL_FRAC,
        test_frac=TEST_FRAC,
    )

    print(f"Train/Val/Test sizes: {len(train_idx)}/{len(val_idx)}/{len(test_idx)}")

    y_np = np.asarray(y, dtype=float)

    print("Train y stats:",
          y_np[train_idx].min(),
          y_np[train_idx].max(),
          y_np[train_idx].mean(),
          y_np[train_idx].std(),
          np.mean(np.abs(y_np[train_idx])))

    print("Val y stats:",
          y_np[val_idx].min(),
          y_np[val_idx].max(),
          y_np[val_idx].mean(),
          y_np[val_idx].std(),
          np.mean(np.abs(y_np[val_idx])))

    print("Test y stats:",
          y_np[test_idx].min(),
          y_np[test_idx].max(),
          y_np[test_idx].mean(),
          y_np[test_idx].std(),
          np.mean(np.abs(y_np[test_idx])))

    # print("Zero predictor MAE:",
    #       "train=", np.mean(np.abs(y_np[train_idx])),
    #       "val=", np.mean(np.abs(y_np[val_idx])),
    #       "test=", np.mean(np.abs(y_np[test_idx])))

    train_mean = y_np[train_idx].mean()
    train_std = y_np[train_idx].std()

    if train_std < 1e-12:
        raise ValueError("Training target std is too small; cannot standardize target.")

    y_norm = (y_np - train_mean) / train_std

    print(f"Target standardization: mean={train_mean:.6f}, std={train_std:.6f}")
    print("Mean predictor MAE:",
          "train=", np.mean(np.abs(y_np[train_idx] - train_mean)),
          "val=", np.mean(np.abs(y_np[val_idx] - train_mean)),
          "test=", np.mean(np.abs(y_np[test_idx] - train_mean)))

    split_name_by_idx = {}
    for idx in train_idx:
        split_name_by_idx[int(idx)] = 'train'
    for idx in val_idx:
        split_name_by_idx[int(idx)] = 'val'
    for idx in test_idx:
        split_name_by_idx[int(idx)] = 'test'

    split_df = pd.DataFrame({
        'Seed': seed,
        'Index': list(range(len(valid_smiles))),
        'SMILES': valid_smiles,
        'Split': [split_name_by_idx[i] for i in range(len(valid_smiles))]
    })
    split_df.to_excel(os.path.join(RESULT_DIR, f'random_split_seed_{seed}.xlsx'), index=False)
    all_split_records.append(split_df)

    # train_subset = Subset(dataset, train_idx)
    # val_subset = Subset(dataset, val_idx)
    # test_subset = Subset(dataset, test_idx)

    dataset_seed = MultiModalDataset(
        graph=graphs,
        index=indices,
        point_clouds=point_cloud,
        point_cloud_length=point_cloud_lengths,
        target=y_norm
    )

    train_subset = Subset(dataset_seed, train_idx)
    val_subset = Subset(dataset_seed, val_idx)
    test_subset = Subset(dataset_seed, test_idx)

    train_loader = DataLoader(train_subset, batch_size=128, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_subset, batch_size=128, shuffle=False, collate_fn=collate_fn)
    test_loader = DataLoader(test_subset, batch_size=128, shuffle=False, collate_fn=collate_fn)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = MultimodalModel(
        vocab_size=vocb_size, smiles_length=max_length, d_smiles=256,
        smiles_num_heads=8, smiles_num_layers=6, cnn_kernels=256,
        d_node=node_dim, d_edge=edge_dim, d_hidden=256, graph_num_layers=6,
        d_point_input=d_input, d_point_cloud=1024, d_point_output=256,
        d_output=1,
        use_smiles=MODALITY_FLAGS["use_smiles"],
        use_graph=MODALITY_FLAGS["use_graph"],
        use_point_cloud=MODALITY_FLAGS["use_point_cloud"],
        fusion_type=get_config()["fusion"]
    ).to(device)

    if seed_idx == 0:
        def count_parameters(models):
            return sum(p.numel() for p in models.parameters() if p.requires_grad)

        total_params = count_parameters(model)
        print(f"Total number of model parameters: {total_params:,}")
        print(f"Number of trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

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

    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-4, weight_decay=1e-4, betas=(0.9, 0.999))
    criterion = torch.nn.L1Loss()
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=8)
    early_stopper = EarlyStopper(patience=10, min_delta=1e-4)

    start_time = time.time()
    train_history = {'loss': [], 'val_loss': []}
    best_val_loss = float('inf')
    best_model_state = None
    best_epoch = 0
    epochs = 200

    for epoch in range(epochs):
        model.train()
        train_loss = 0
        train_batches = 0

        for batch_idx, batch in enumerate(train_loader):
            optimizer.zero_grad()
            smiles_batch = batch['smiles_idx'].to(device)
            graphs_batch = batch['graph'].to(device)
            point_cloud_batch = batch['point_cloud'].to(device)
            point_cloud_length_batch = batch['point_cloud_length'].to(device)
            targets_batch = batch['target'].to(device)

            output = model(smiles_batch, graphs_batch, point_cloud_batch, point_cloud_length_batch)
            loss = criterion(output.view(-1), targets_batch)
            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
            optimizer.step()

            train_loss += loss.item()
            train_batches += 1

            print(f'Seed {seed} | Epoch {epoch + 1}/{epochs} | '
                  f'Batch {batch_idx + 1}/{len(train_loader)} | '
                  f'Loss: {loss.item():.4f}')

        model.eval()
        val_loss = 0
        val_batches = 0
        with torch.no_grad():
            for batch in val_loader:
                smiles_batch = batch['smiles_idx'].to(device)
                graphs_batch = batch['graph'].to(device)
                point_cloud_batch = batch['point_cloud'].to(device)
                point_cloud_length_batch = batch['point_cloud_length'].to(device)
                targets_batch = batch['target'].to(device)

                output = model(smiles_batch, graphs_batch, point_cloud_batch, point_cloud_length_batch)
                loss = criterion(output.view(-1), targets_batch)
                val_loss += loss.item()
                val_batches += 1

        avg_train_loss = train_loss / train_batches
        avg_val_loss = val_loss / val_batches
        scheduler.step(avg_val_loss)
        current_lr = optimizer.param_groups[0]['lr']
        train_history['loss'].append(avg_train_loss)
        train_history['val_loss'].append(avg_val_loss)

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_model_state = copy.deepcopy(model.state_dict())
            best_epoch = epoch

        print(f'Seed {seed} | Epoch {epoch + 1}/{epochs} | '
              f'Train Loss: {avg_train_loss:.6f} | Val Loss: {avg_val_loss:.6f} | LR: {current_lr:.2e}')

        if early_stopper(avg_val_loss):
            print(f"Early stopping triggered at epoch {epoch + 1}")
            break

    training_time = time.time() - start_time
    minutes = training_time / 60

    if best_model_state is None:
        best_model_state = copy.deepcopy(model.state_dict())

    model.load_state_dict(best_model_state)
    model.eval()
    seed_targets, seed_predictions = [], []
    with torch.no_grad():
        for batch in test_loader:
            smiles_batch = batch['smiles_idx'].to(device)
            graphs_batch = batch['graph'].to(device)
            point_cloud_batch = batch['point_cloud'].to(device)
            point_cloud_length_batch = batch['point_cloud_length'].to(device)
            targets_batch = batch['target'].to(device)

            output = model(smiles_batch, graphs_batch, point_cloud_batch, point_cloud_length_batch)
            seed_predictions.append(output.view(-1))
            seed_targets.append(targets_batch)

    seed_predictions = torch.cat(seed_predictions)
    seed_targets = torch.cat(seed_targets)

    # predictions_np = seed_predictions.cpu().numpy()
    # targets_np = seed_targets.cpu().numpy()

    predictions_norm = seed_predictions.cpu().numpy()
    targets_norm = seed_targets.cpu().numpy()

    predictions_np = predictions_norm * train_std + train_mean
    targets_np = targets_norm * train_std + train_mean

    seed_smiles = [valid_smiles[i] for i in test_idx]
    all_smiles.extend(seed_smiles)

    MAE = mean_absolute_error(targets_np, predictions_np)
    RMSE = np.sqrt(mean_squared_error(targets_np, predictions_np))
    R2 = r2_score(targets_np, predictions_np)

    seed_results.append({'Seed': seed, 'True_Values': targets_np, 'Predicted_Values': predictions_np})
    all_predictions.extend(predictions_np.tolist())
    all_true_values.extend(targets_np.tolist())

    fold_metrics['Seed'].append(seed)
    fold_metrics['Train_Size'].append(len(train_idx))
    fold_metrics['Val_Size'].append(len(val_idx))
    fold_metrics['Test_Size'].append(len(test_idx))
    fold_metrics['MAE'].append(MAE)
    fold_metrics['RMSE'].append(RMSE)
    fold_metrics['R2'].append(R2)
    fold_metrics['Best_Epoch'].append(best_epoch + 1)
    fold_metrics['Training_Time(min)'].append(minutes)

    print(f"\nSeed {seed} Test Results:")
    print(f"MAE: {MAE:.4f}, RMSE: {RMSE:.4f}, R2: {R2:.4f}, Time: {minutes:.2f} min")

    pred_df = pd.DataFrame({
        'SMILES': seed_smiles,
        'True_Values': targets_np,
        'Predicted_Values': predictions_np
    })
    pred_df.to_excel(os.path.join(RESULT_DIR, f'predictions_seed_{seed}.xlsx'), index=False)

    history_df = pd.DataFrame({
        'Epoch': range(1, len(train_history['loss']) + 1),
        'Train_Loss': train_history['loss'],
        'Val_Loss': train_history['val_loss']
    })
    history_df.to_excel(os.path.join(RESULT_DIR, f'training_history_seed_{seed}.xlsx'), index=False)


# =========================
# Save metrics and summaries
# =========================
fold_metrics_df = pd.DataFrame(fold_metrics)
fold_metrics_df.to_excel(os.path.join(RESULT_DIR, 'random_3_seeds_metrics.xlsx'), index=False)

all_splits_df = pd.concat(all_split_records, ignore_index=True)
all_splits_df.to_excel(os.path.join(RESULT_DIR, 'all_random_splits.xlsx'), index=False)

avg_mae = float(np.mean(fold_metrics['MAE']))
std_mae = float(np.std(fold_metrics['MAE']))
avg_rmse = float(np.mean(fold_metrics['RMSE']))
std_rmse = float(np.std(fold_metrics['RMSE']))
avg_r2 = float(np.mean(fold_metrics['R2']))
std_r2 = float(np.std(fold_metrics['R2']))

print("\n" + "=" * 50)
print("Random Split 3-Seed Summary")
print("=" * 50)
print(f"Average MAE: {avg_mae:.4f} +/- {std_mae:.4f}")
print(f"Average RMSE: {avg_rmse:.4f} +/- {std_rmse:.4f}")
print(f"Average R2: {avg_r2:.4f} +/- {std_r2:.4f}")
print(f"Average Training Time per seed: {np.mean(fold_metrics['Training_Time(min)']):.2f} min")

all_results_df = pd.DataFrame({
    'SMILES': all_smiles,
    'True_Values': all_true_values,
    'Predicted_Values': all_predictions
})[['SMILES', 'True_Values', 'Predicted_Values']]
all_results_df.to_excel(os.path.join(RESULT_DIR, 'all_predictions.xlsx'), index=False)


# =========================
# Publication-quality visualization
# =========================

plt.rcParams.update({
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
})

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

# ---------- a) True vs predicted parity plot ----------
hb = ax1.hexbin(
    true_values,
    pred_values,
    gridsize=70,
    mincnt=1,
    bins="log",
    cmap="viridis",
    linewidths=0.0
)

ax1.plot(
    [plot_min, plot_max],
    [plot_min, plot_max],
    linestyle="--",
    linewidth=1.8,
    color="black",
    alpha=0.85,
    label="Ideal"
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
    label=f"Fit: y = {coef[0]:.3f}x + {coef[1]:.3f}"
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
        alpha=0.92
    )
)

cb1 = fig.colorbar(hb, ax=ax1, fraction=0.046, pad=0.025)
cb1.set_label("log$_{10}$(point count)", fontsize=12)

# ---------- b) Residual vs true value ----------
hb2 = ax2.hexbin(
    true_values,
    residuals,
    gridsize=70,
    mincnt=1,
    bins="log",
    cmap="magma",
    linewidths=0.0
)

ax2.axhline(0, linestyle="--", linewidth=1.6, color="black", alpha=0.85)

res_std = np.std(residuals)
ax2.axhline(2 * res_std, linestyle=":", linewidth=1.2, color="gray")
ax2.axhline(-2 * res_std, linestyle=":", linewidth=1.2, color="gray")

ax2.set_xlabel("True value")
ax2.set_ylabel("Residual (Predicted 鈭?True)")
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
        alpha=0.92
    )
)

cb2 = fig.colorbar(hb2, ax=ax2, fraction=0.046, pad=0.025)
cb2.set_label("log$_{10}$(point count)", fontsize=12)

# ---------- c) Residual distribution ----------
ax3.hist(
    residuals,
    bins=80,
    density=True,
    alpha=0.82,
    edgecolor="white",
    linewidth=0.3
)

ax3.axvline(0, linestyle="--", linewidth=1.6, color="black", alpha=0.85)
ax3.axvline(np.mean(residuals), linestyle="-", linewidth=1.6, color="crimson", alpha=0.9)

ax3.set_xlabel("Residual (Predicted 鈭?True)")
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
        alpha=0.92
    )
)

# ---------- d) Seed-wise performance ----------
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
    linewidth=0.8
)

ax4.bar(
    x + width / 2,
    rmse_values,
    width=width,
    label="RMSE",
    alpha=0.88,
    edgecolor="white",
    linewidth=0.8
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
    label="$R^2$"
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
    "QM9 prediction performance across random splits",
    fontsize=17,
    fontweight="bold",
    y=0.995
)

plt.tight_layout(rect=(0.0, 0.0, 1.0, 0.975))

fig.savefig(os.path.join(RESULT_DIR, "publication_performance_summary.png"), dpi=600)
fig.savefig(os.path.join(RESULT_DIR, "publication_performance_summary.pdf"))

plt.close(fig)

# ---------- Standalone parity plot for manuscript ----------
fig_single, ax = plt.subplots(figsize=(7.2, 6.8))

hb_single = ax.hexbin(
    true_values,
    pred_values,
    gridsize=80,
    mincnt=1,
    bins="log",
    cmap="viridis",
    linewidths=0.0
)

ax.plot(
    [plot_min, plot_max],
    [plot_min, plot_max],
    linestyle="--",
    linewidth=1.8,
    color="black",
    alpha=0.85
)

ax.plot(
    x_fit,
    fit_line(x_fit),
    linewidth=1.8,
    color="crimson",
    alpha=0.9
)

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
        alpha=0.92
    )
)

cb = fig_single.colorbar(hb_single, ax=ax, fraction=0.046, pad=0.025)
cb.set_label("log$_{10}$(point count)", fontsize=12)

plt.tight_layout()

fig_single.savefig(os.path.join(RESULT_DIR, "publication_parity_plot.png"), dpi=600)
fig_single.savefig(os.path.join(RESULT_DIR, "publication_parity_plot.pdf"))

plt.close(fig_single)

print("Publication-quality figures saved:")
print(os.path.join(RESULT_DIR, "publication_performance_summary.png"))
print(os.path.join(RESULT_DIR, "publication_performance_summary.pdf"))
print(os.path.join(RESULT_DIR, "publication_parity_plot.png"))
print(os.path.join(RESULT_DIR, "publication_parity_plot.pdf"))



