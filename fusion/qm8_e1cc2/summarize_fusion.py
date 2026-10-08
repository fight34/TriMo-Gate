"""Validate and summarize the fixed QM8 E1-CC2 fusion experiment.

The final comparison requires six methods and all three predeclared seeds.
``--allow-partial`` writes a progress report only, never a ranked comparison.
Only the Python standard library is needed; test predictions are never averaged
across independently trained models when calculating the reported metrics.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import statistics


FUSIONS = ("concat", "mean", "weighted", "attention", "gate_no_interaction", "gate")
SEEDS = (42, 2025, 3407)
LABELS = {
    "concat": "拼接融合", "mean": "均匀平均", "weighted": "标量加权",
    "attention": "注意力融合", "gate_no_interaction": "无成对交互门控",
    "gate": "完整门控",
}


class ValidationError(ValueError):
    """An artifact is missing, inconsistent, or not a finalized test result."""


def read_json(path):
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise ValidationError(f"Cannot read {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ValidationError(f"Expected a JSON object: {path}")
    return value


def sha256_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def finite_number(value, label, minimum=None):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValidationError(f"{label} must be numeric")
    if not math.isfinite(value) or (minimum is not None and value < minimum):
        raise ValidationError(f"Invalid {label}: {value!r}")
    return float(value)


def positive_integer(value, label):
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValidationError(f"{label} must be a positive integer")
    return value


def prediction_metrics(targets, predictions):
    if not targets or len(targets) != len(predictions):
        raise ValidationError("Predictions must have a nonempty, matching target vector")
    errors = [prediction - target for target, prediction in zip(targets, predictions)]
    count = len(errors)
    target_mean = statistics.fmean(targets)
    sse = math.fsum(error * error for error in errors)
    sst = math.fsum((target - target_mean) ** 2 for target in targets)
    return {
        "mae": math.fsum(abs(error) for error in errors) / count,
        "rmse": math.sqrt(sse / count),
        "r2": 1.0 - sse / sst if sst > 0 else None,
        "n": count,
    }


def save_csv(path, fieldnames, rows):
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def require(condition, message):
    if not condition:
        raise ValidationError(message)


def check_digest(value, label):
    require(isinstance(value, str) and len(value) == 64 and
            all(char in "0123456789abcdef" for char in value),
            f"Invalid SHA256: {label}")
    return value


def read_predictions(path, expected_ids):
    identifiers, targets, predictions = [], [], []
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        expected_columns = {"id", "target_original_units", "prediction_original_units"}
        require(reader.fieldnames is not None and set(reader.fieldnames) == expected_columns,
                f"Unexpected prediction columns: {path}")
        for row in reader:
            try:
                identifier = int(row["id"])
                target = float(row["target_original_units"])
                prediction = float(row["prediction_original_units"])
            except (KeyError, TypeError, ValueError) as exc:
                raise ValidationError(f"Invalid prediction row in {path}") from exc
            require(math.isfinite(target) and math.isfinite(prediction),
                    f"Nonfinite test prediction/target in {path}")
            identifiers.append(identifier)
            targets.append(target)
            predictions.append(prediction)
    require(len(set(identifiers)) == len(identifiers), f"Duplicate test ID: {path}")
    require(identifiers == expected_ids,
            f"Test prediction IDs/order differ from the frozen test split: {path}")
    return targets, predictions


def safe_artifact(run, filename, label):
    require(isinstance(filename, str) and filename and Path(filename).name == filename,
            f"{label} must be a plain filename")
    path = run / filename
    require(path.is_file(), f"Missing {label}: {path}")
    return path


def validate_split_groups(split, split_groups):
    require(set(split_groups) == set(split), "Unexpected structural group split sections")
    owners, minima = {}, {}
    for part, identifiers in split.items():
        groups = split_groups[part]
        require(isinstance(groups, list) and len(groups) == len(identifiers) and
                all(isinstance(value, int) and not isinstance(value, bool) for value in groups),
                f"Invalid or misaligned structural groups: {part}")
        for identifier, group in zip(identifiers, groups):
            require(group not in owners or owners[group] == part,
                    f"Structural group crosses split boundaries: {group}")
            owners[group] = part
            minima[group] = min(minima.get(group, identifier), identifier)
    require(all(group == minimum for group, minimum in minima.items()),
            "Structural group ID differs from its minimum stable molecule ID")


def load_run(run, fusion, seed):
    status = read_json(run / "status.json")
    require(status.get("status") == "completed", f"Run is not finalized: {run}")
    record = read_json(run / "summary.json")
    require(record.get("schema_version") == 1, f"Unsupported result schema: {run}")
    require(record.get("status") == "completed" and record.get("finalized") is True,
            f"Test result is not finalized: {run}")
    require(record.get("diagnostic") is False, f"Diagnostic runs cannot enter comparison: {run}")
    require(record.get("fusion_type") == fusion and record.get("seed") == seed,
            f"Run identity mismatch: {run}")
    require(record.get("dataset") == "QM8" and record.get("dataset_size") == 21747,
            f"Unexpected dataset or record count: {run}")
    require(record.get("target") == "E1-CC2" and record.get("unit") == "hartree",
            f"Unexpected target or unit: {run}")
    require(record.get("best_checkpoint_reload_verified") is True,
            f"Best-checkpoint reload was not verified: {run}")
    for key in ("protocol", "provenance", "metrics", "params", "timing", "epochs",
                "initialization_sha256"):
        require(isinstance(record.get(key), dict), f"Missing result object {key}: {run}")
    protocol = record["protocol"]
    require(protocol.get("cmcr") is False, f"CMCR must be disabled: {run}")
    require(protocol.get("mixed_precision") is False, f"Expected fixed FP32 protocol: {run}")
    require(protocol.get("split_seed") == 20260927, f"Unexpected split seed: {run}")
    require(protocol.get("dataset") == "QM8" and protocol.get("target") == "E1-CC2"
            and protocol.get("unit") == "hartree" and protocol.get("expected_sample_count") == 21747,
            f"Unexpected dataset protocol: {run}")
    provenance = record["provenance"]
    for key in ("cache_sha256", "metadata_sha256", "split_ids_sha256", "split_groups_sha256", "data_fingerprint_sha256",
                "checkpoint_sha256"):
        check_digest(provenance.get(key), f"{run}: {key}")
    sources = provenance.get("source_sha256")
    require(isinstance(sources, dict) and sources, f"Missing source hashes: {run}")
    for name, digest in sources.items():
        check_digest(digest, f"source {name}")
    for name in ("encoders", "common_head"):
        check_digest(record["initialization_sha256"].get(name), f"initialization {name}")

    split_path = safe_artifact(run, record.get("split_ids_file"), "split IDs")
    split = read_json(split_path)
    require(set(split) == {"train", "val", "test"}, f"Unexpected split sections: {run}")
    all_ids = []
    for part in ("train", "val", "test"):
        identifiers = split[part]
        require(isinstance(identifiers, list) and identifiers and
                all(isinstance(value, int) and not isinstance(value, bool) for value in identifiers),
                f"Invalid {part} IDs: {run}")
        all_ids.extend(identifiers)
    require(len(all_ids) == len(set(all_ids)), f"Duplicate IDs or leaking splits: {run}")
    require(len(all_ids) == 21747 and
            {part: len(values) for part, values in split.items()} == protocol.get("split_sizes"),
            f"Split sizes disagree with the complete QM8 structural-group protocol: {run}")
    # The trainer hashes the saved split file, including its serialization.
    require(sha256_file(split_path) == provenance["split_ids_sha256"],
            f"Split SHA256 mismatch: {run}")
    group_path = safe_artifact(run, record.get("split_groups_file"), "split groups")
    split_groups = read_json(group_path)
    require(sha256_file(group_path) == provenance["split_groups_sha256"],
            f"Split groups SHA256 mismatch: {run}")
    validate_split_groups(split, split_groups)

    prediction_path = safe_artifact(run, record.get("predictions_file"), "test predictions")
    expected_hash = check_digest(record.get("predictions_sha256"), "test predictions")
    require(sha256_file(prediction_path) == expected_hash, f"Prediction SHA256 mismatch: {run}")
    targets, predictions = read_predictions(prediction_path, split["test"])
    metrics = prediction_metrics(targets, predictions)
    saved_metrics = record["metrics"]
    require(saved_metrics.get("test_n") == metrics["n"], f"Test count mismatch: {run}")
    for metric in ("mae", "rmse"):
        saved = finite_number(saved_metrics.get(f"test_{metric}"), f"test_{metric}", 0)
        require(math.isclose(saved, metrics[metric], rel_tol=1e-6, abs_tol=1e-8),
                f"Recorded {metric} disagrees with test predictions: {run}")
    validation_mae = finite_number(saved_metrics.get("validation_mae"), "validation MAE", 0)
    validation_rmse = finite_number(saved_metrics.get("validation_rmse"), "validation RMSE", 0)

    parameters = record["params"]
    for name in ("encoder", "fusion", "total"):
        positive_integer(parameters.get(name), f"{name} parameter count")
    require(parameters["encoder"] + parameters["fusion"] == parameters["total"],
            f"Parameter decomposition mismatch: {run}")
    timing = record["timing"]
    for name in ("train_total_s", "val_total_s", "inference_total_s", "training_wall_s",
                 "cuda_peak_allocated_bytes"):
        finite_number(timing.get(name), name, 0)
    epochs = record["epochs"]
    for name in ("best", "completed", "max_epochs"):
        positive_integer(epochs.get(name), f"{name} epoch")
    require(epochs["best"] <= epochs["completed"] <= epochs["max_epochs"],
            f"Invalid epoch accounting: {run}")
    require(epochs["max_epochs"] == protocol.get("max_epochs"),
            f"Maximum epochs disagree with protocol: {run}")
    reason = epochs.get("termination_reason")
    require(reason in ("max_epochs", "early_stopping"), f"Incomplete training reason: {run}")
    if reason == "max_epochs":
        require(epochs["completed"] == epochs["max_epochs"], f"Truncated training: {run}")
    else:
        require(epochs["completed"] >= protocol.get("min_epochs", math.inf),
                f"Early stopping before minimum epochs: {run}")

    # Optional large weight files need not be copied with a portable report.
    # Where available, validate their identity as an additional integrity check.
    checkpoint = run / "best.pt"
    checkpoint_checked = checkpoint.is_file()
    if checkpoint_checked:
        require(sha256_file(checkpoint) == provenance["checkpoint_sha256"],
                f"Best checkpoint SHA256 mismatch: {run}")
    row = {
        "fusion_type": fusion, "fusion_name": LABELS[fusion], "seed": seed,
        "test_n": metrics["n"], "test_mae_hartree": metrics["mae"],
        "test_rmse_hartree": metrics["rmse"], "test_r2": metrics["r2"],
        "validation_mae_hartree": validation_mae, "validation_rmse_hartree": validation_rmse,
        "encoder_params": parameters["encoder"], "fusion_params": parameters["fusion"],
        "total_params": parameters["total"], "best_epoch": epochs["best"],
        "completed_epochs": epochs["completed"], "termination_reason": reason,
        "training_wall_s": timing["training_wall_s"], "train_total_s": timing["train_total_s"],
        "val_total_s": timing["val_total_s"], "inference_total_s": timing["inference_total_s"],
        "inference_ms_per_molecule": timing["inference_total_s"] * 1000 / metrics["n"],
        "cuda_peak_allocated_MiB": timing["cuda_peak_allocated_bytes"] / (1024 ** 2),
        "predictions_sha256": expected_hash, "checkpoint_sha256": provenance["checkpoint_sha256"],
    }
    return {"row": row, "record": record, "split": split, "split_groups": split_groups, "targets": targets,
            "checkpoint_hash_checked": checkpoint_checked}


def verify_shared_protocol(results):
    if not results:
        return
    reference = results[0]
    reference_record = reference["record"]
    initializations = {}
    for item in results:
        row, record = item["row"], item["record"]
        label = f"{row['fusion_type']}/seed_{row['seed']}"
        require(record["protocol"] == reference_record["protocol"],
                f"Training protocols differ: {label}")
        for key in ("cache_sha256", "metadata_sha256", "source_sha256", "split_ids_sha256",
                    "split_groups_sha256", "data_fingerprint_sha256"):
            require(record["provenance"][key] == reference_record["provenance"][key],
                    f"Shared provenance differs ({key}): {label}")
        require(item["split"] == reference["split"], f"Frozen splits differ: {label}")
        require(item["split_groups"] == reference["split_groups"], f"Frozen structural groups differ: {label}")
        require(item["targets"] == reference["targets"], f"Test labels differ: {label}")
        require(row["encoder_params"] == reference["row"]["encoder_params"],
                f"Encoder parameter count differs: {label}")
        seed = row["seed"]
        initialization = record["initialization_sha256"]
        if seed in initializations:
            require(initialization == initializations[seed],
                    f"Shared encoder/head initialization differs within seed {seed}: {label}")
        else:
            initializations[seed] = initialization


def aggregate_rows(results):
    rows = [item["row"] for item in results]
    gate_rows = {row["seed"]: row for row in rows if row["fusion_type"] == "gate"}
    aggregate = []
    for fusion in FUSIONS:
        selected = [row for row in rows if row["fusion_type"] == fusion]
        require(len(selected) == len(SEEDS), f"Cannot aggregate incomplete {fusion}")
        combined = {"fusion_type": fusion, "fusion_name": LABELS[fusion], "n_seeds": len(SEEDS)}
        for name in ("test_n", "encoder_params", "fusion_params", "total_params"):
            values = {row[name] for row in selected}
            require(len(values) == 1, f"{name} differs across seeds for {fusion}")
            combined[name] = selected[0][name]
        for metric in ("test_mae_hartree", "test_rmse_hartree", "test_r2", "best_epoch",
                       "completed_epochs", "training_wall_s", "inference_ms_per_molecule",
                       "cuda_peak_allocated_MiB"):
            values = [row[metric] for row in selected]
            combined[f"{metric}_mean"] = statistics.fmean(values) if None not in values else None
            combined[f"{metric}_sample_sd"] = statistics.stdev(values) if None not in values else None
        differences = [row["test_mae_hartree"] - gate_rows[row["seed"]]["test_mae_hartree"]
                       for row in selected]
        combined["delta_mae_vs_gate_hartree_mean"] = statistics.fmean(differences)
        combined["delta_mae_vs_gate_hartree_sample_sd"] = statistics.stdev(differences)
        combined["summed_training_wall_s"] = math.fsum(row["training_wall_s"] for row in selected)
        aggregate.append(combined)
    return aggregate


def number(value, decimals=6):
    return "未定义" if value is None else f"{value:.{decimals}f}"


def write_report(out, results, aggregate):
    record = results[0]["record"]
    protocol = record["protocol"]
    split = results[0]["split"]
    lines = [
        "# QM8 E1-CC2：六种融合方法的比较", "",
        "本表比较同一模型的六种融合方法。每种方法均从头训练，使用预先固定的三个种子："
        "42、2025、3407。测试集仅用于训练结束后评估验证集所选的最佳检查点。", "",
        f"训练 / 验证 / 测试分子数：{len(split['train']):,} / {len(split['val']):,} / "
        f"{len(split['test']):,}；固定划分种子：{protocol['split_seed']}。",
        "按预先缓存的结构等价组整体随机划分，累计分子数取最接近 80% / 90% 的整组边界；"
        "各组不跨训练、验证或测试集，实际比例可能略偏离 80/10/10。分组不使用目标数值。",
        f"最大训练轮数：{protocol['max_epochs']}；最少训练轮数：{protocol['min_epochs']}；"
        f"早停耐心：{protocol['early_stopping_patience']}。CMCR 关闭，使用 FP32。", "",
        "MAE 和 RMSE 的单位为 hartree，越低越好。均值和标准差均基于三个独立训练运行；"
        "标准差采用样本标准差（分母 n−1），不是标准误或置信区间。没有先平均多个模型的预测。", "",
        "| 融合方法 | 测试 MAE：均值 ± SD | 测试 RMSE：均值 ± SD | 融合参数 | 总参数 |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in aggregate:
        mae = f"{number(row['test_mae_hartree_mean'])} ± {number(row['test_mae_hartree_sample_sd'])}"
        rmse = f"{number(row['test_rmse_hartree_mean'])} ± {number(row['test_rmse_hartree_sample_sd'])}"
        lines.append(f"| {row['fusion_name']} | {mae} | {rmse} | {row['fusion_params']:,} | "
                     f"{row['total_params']:,} |")
    lines.extend(["", "| 融合方法 | 相对完整门控 ΔMAE：均值 ± SD | 最佳轮数均值 | "
                  "完成轮数均值 | 每次训练墙钟时间均值（分钟） | 推理毫秒/分子均值 |",
                  "|---|---:|---:|---:|---:|---:|"])
    for row in aggregate:
        difference = (f"{number(row['delta_mae_vs_gate_hartree_mean'])} ± "
                      f"{number(row['delta_mae_vs_gate_hartree_sample_sd'])}")
        lines.append(f"| {row['fusion_name']} | {difference} | {row['best_epoch_mean']:.1f} | "
                     f"{row['completed_epochs_mean']:.1f} | {row['training_wall_s_mean']/60:.2f} | "
                     f"{row['inference_ms_per_molecule_mean']:.4f} |")
    lines.extend([
        "", "ΔMAE = 对照方法 MAE − 完整门控 MAE，按相同训练种子配对；正值表示完整门控误差更低。"
        "三个种子的波动只反映本次固定划分下的训练随机性，不能据此声称统计显著或跨性质普遍有效。",
        "", "训练时间记录每个运行实际经历的墙钟时间；并发运行会相互影响。各运行时间相加不等于"
        "服务器租用时长或 GPU 计费时长。推理速度仅代表本次设备、批大小和计时实现。",
        "", "本轮采用各融合模块的自然参数量，参数数量已列出；若参数量不同，性能差异不能全部归因于"
        "融合规则。容量匹配与其他 QM8 性质需要另行验证。",
    ])
    lines.extend(["", "这些结果对应所记录的 QM8 E1-CC2 缓存与固定结构分组划分；"
                  "不应直接视为与不同目标、筛选、分组或划分协议的文献数值可比。"])
    lines.extend(["", "已核验所有 18 个运行的完成状态、数据/源代码/划分哈希、同种子公共初始化、"
                  "结构组无跨集重叠、测试分子及标签一致性、预测文件哈希，并从逐分子预测重新计算 MAE/RMSE。",
                  "", f"本地可用且通过哈希检查的最佳检查点文件："
                  f"{sum(item['checkpoint_hash_checked'] for item in results)}/18。未随报告复制的权重仍由"
                  "训练器记录的检查点哈希和重载验证标记追溯。", "",
                  "逐种子结果见 `per_seed.csv`，汇总结果见 `comparison.csv`；验证记录见 "
                  "`validation_manifest.json`。", ""])
    (out / "comparison.md").write_text("\n".join(lines), encoding="utf-8")


def summarize(runs, out, allow_partial=False):
    runs, out = Path(runs), Path(out)
    results, progress = [], []
    expected = {(fusion, seed) for fusion in FUSIONS for seed in SEEDS}
    discovered = []
    if runs.exists():
        discovered = list(runs.glob("*/seed_*/summary.json"))
    for path in discovered:
        identity = (path.parent.parent.name, path.parent.name)
        require(identity in {(fusion, f"seed_{seed}") for fusion, seed in expected},
                f"Unexpected run in comparison root: {path.parent}")
    for fusion in FUSIONS:
        for seed in SEEDS:
            run = runs / fusion / f"seed_{seed}"
            state = read_json(run / "status.json").get("status", "unknown") if (run / "status.json").exists() else "missing"
            progress.append({"fusion_type": fusion, "fusion_name": LABELS[fusion], "seed": seed,
                             "status": state})
            if state == "completed":
                results.append(load_run(run, fusion, seed))
    verify_shared_protocol(results)
    if allow_partial:
        out.mkdir(parents=True, exist_ok=True)
        save_csv(out / "progress.csv", list(progress[0]), progress)
        lines = ["# 融合实验进度（非最终结果）", "",
                 f"已完成并通过结果核验：{len(results)}/18。此文件只报告状态，不比较或排名部分结果。", "",
                 "| 融合方法 | 种子 | 状态 |", "|---|---:|---|"]
        lines.extend(f"| {row['fusion_name']} | {row['seed']} | {row['status']} |" for row in progress)
        (out / "progress.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
        return {"status": "progress_only", "completed": len(results), "required": 18}
    require(len(results) == 18, "Final comparison requires all 18 finalized runs; "
            f"found {len(results)}. Use --allow-partial for a status-only report.")
    aggregate = aggregate_rows(results)
    out.mkdir(parents=True, exist_ok=True)
    rows = [item["row"] for item in results]
    save_csv(out / "per_seed.csv", list(rows[0]), rows)
    save_csv(out / "comparison.csv", list(aggregate[0]), aggregate)
    manifest = {
        "schema_version": 1, "status": "completed", "validated_runs": len(results),
        "fusions": list(FUSIONS), "seeds": list(SEEDS),
        "dataset": "QM8", "target": "E1-CC2", "dataset_size": 21747, "unit": "hartree",
        "metric_aggregation": "mean and sample standard deviation across per-run metrics",
        "protocol": results[0]["record"]["protocol"],
        "shared_provenance": {key: results[0]["record"]["provenance"][key]
                              for key in ("cache_sha256", "metadata_sha256", "source_sha256", "split_ids_sha256", "split_groups_sha256",
                                          "data_fingerprint_sha256")},
        "runs": [{"fusion_type": item["row"]["fusion_type"], "seed": item["row"]["seed"],
                  "predictions_sha256": item["row"]["predictions_sha256"],
                  "checkpoint_sha256": item["row"]["checkpoint_sha256"],
                  "checkpoint_hash_checked_locally": item["checkpoint_hash_checked"]} for item in results],
    }
    (out / "validation_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    write_report(out, results, aggregate)
    return {"status": "completed", "validated_runs": len(results), "output": str(out.resolve())}


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--allow-partial", action="store_true")
    return parser.parse_args(argv)


if __name__ == "__main__":
    arguments = parse_args()
    try:
        result = summarize(arguments.runs, arguments.out, arguments.allow_partial)
    except (ValidationError, OSError, KeyError, TypeError) as exc:
        raise SystemExit(f"Result validation failed: {exc}") from exc
    print(json.dumps(result, ensure_ascii=False))
