"""Evaluate frozen QM9 electronic_spatial_extent A'/B'/C'/D checkpoints on one shared GPU backend."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import statistics
import time


ROOT = Path(__file__).resolve().parent
SEEDS = (42, 2025, 3407)
TARGET = 'electronic_spatial_extent'
UNIT = 'a0^2'
CONDITIONS = {
    'A_prime': ('dft', 'dft', 'DFT→DFT'),
    'B_prime': ('dft', 'etkdg', 'DFT→ETKDGv3'),
    'C_prime': ('dft', 'mmff', 'DFT→ETKDGv3+MMFF94'),
    'D': ('mmff', 'mmff', 'ETKDGv3+MMFF94→ETKDGv3+MMFF94'),
}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1 << 20), b''):
            digest.update(block)
    return digest.hexdigest()


def write_csv(path, rows):
    with Path(path).open('w', encoding='utf-8-sig', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def metrics(targets, predictions):
    import numpy as np
    truth, prediction = np.asarray(targets, dtype=np.float64), np.asarray(predictions, dtype=np.float64)
    require(truth.ndim == 1 and truth.shape == prediction.shape and len(truth) > 0, 'Invalid metric shapes')
    require(np.isfinite(truth).all() and np.isfinite(prediction).all(), 'Nonfinite target or prediction')
    error = prediction - truth
    return {'n': len(truth), 'mae': float(np.abs(error).mean()),
            'rmse': float(np.sqrt(np.square(error).mean()))}


def read_predictions(path):
    import numpy as np
    with Path(path).open(encoding='utf-8', newline='') as stream:
        rows = list(csv.DictReader(stream))
    return ([int(row['id']) for row in rows],
            np.asarray([float(row['target_original_units']) for row in rows], dtype=np.float64),
            np.asarray([float(row['prediction_original_units']) for row in rows], dtype=np.float64))


def ordered_etkdg(path, expected_hash, test_ids, lengths, dft_points):
    import numpy as np
    require(sha(path) == expected_hash, 'ETKDG test archive hash mismatch')
    with np.load(path, allow_pickle=False) as archive:
        require(set(archive.files) == {'ids', 'points'}, 'Unexpected ETKDG archive fields')
        ids, points = archive['ids'], archive['points']
    require(ids.ndim == 1 and ids.dtype.kind in 'iu' and len(np.unique(ids)) == len(ids), 'Invalid ETKDG IDs')
    require(len(ids) == len(test_ids) and set(ids.tolist()) == set(test_ids), 'ETKDG test cohort mismatch')
    require(points.shape == (len(ids), 29, 9) and points.dtype == np.float32, 'Invalid ETKDG point shape/dtype')
    index = {int(mid): row for row, mid in enumerate(ids)}
    points = points[[index[mid] for mid in test_ids]]
    require(np.isfinite(points).all(), 'Nonfinite ETKDG features')
    mask = np.arange(points.shape[1])[None, :] < lengths[:, None]
    require(not np.count_nonzero(points[~mask]), 'ETKDG padding must be zero')
    types = points[:, :, 4:]
    require(np.isin(types, (0, 1)).all(), 'ETKDG elements must be one-hot')
    np.testing.assert_array_equal(types.sum(axis=2), mask.astype(np.float32))
    np.testing.assert_array_equal(types.sum(axis=1), dft_points[:, :, 4:].sum(axis=1))
    np.testing.assert_allclose(points[:, :, :3].sum(axis=1) / lengths[:, None], 0, atol=2e-6)
    np.testing.assert_allclose(points[:, :, 3], np.linalg.norm(points[:, :, :3], axis=2), atol=2e-6, rtol=2e-6)
    return points


def completion_barrier(root):
    for geometry in ('dft', 'mmff'):
        for seed in SEEDS:
            summary = read(root / 'runs' / geometry / f'seed_{seed}' / 'summary.json')
            require(summary['status'] == 'completed' and summary['finalized'] and summary['test_evaluated'],
                    f'{geometry} seed {seed} must be fully finalized before ABCD evaluation')
            require(summary['best_checkpoint_reload_verified'], 'Finalization reload verification missing')


def aggregate_rows(rows, extra_keys=()):
    groups = {}
    for row in rows:
        key = tuple(row[name] for name in ('label', *extra_keys))
        groups.setdefault(key, []).append(row)
    aggregate = []
    for key, selected in groups.items():
        require(sorted(row['seed'] for row in selected) == list(SEEDS), 'Aggregate needs each fixed seed exactly once')
        require(len({row['n'] for row in selected}) == 1, 'Aggregate cohort sizes differ')
        record = dict(zip(('label', *extra_keys), key))
        record.update(condition=CONDITIONS[key[0]][2], n_test=selected[0]['n'], seeds=len(selected))
        for metric in ('mae', 'rmse'):
            values = [row[metric] for row in selected]
            record.update({metric + '_mean': statistics.mean(values), metric + '_sd': statistics.stdev(values)})
        aggregate.append(record)
    return aggregate


def paired_changes(rows):
    indexed = {(row['label'], row['seed']): row for row in rows}
    output = []
    for other, base in (('B_prime', 'A_prime'), ('C_prime', 'B_prime'), ('C_prime', 'A_prime'),
                        ('D', 'C_prime'), ('D', 'A_prime')):
        for metric in ('mae', 'rmse'):
            denominators = [indexed[base, seed][metric] for seed in SEEDS]
            require(all(value > 0 for value in denominators), 'Cannot report relative change from zero error')
            values = [100 * (indexed[other, seed][metric] / indexed[base, seed][metric] - 1) for seed in SEEDS]
            output.append({'comparison': other + '_vs_' + base, 'metric': metric.upper(),
                           'mean_change_pct': statistics.mean(values), 'sd_change_pct': statistics.stdev(values),
                           **{f'seed_{seed}_pct': value for seed, value in zip(SEEDS, values)}})
    return output


def verify_summary_config(summary, config):
    # Dataset size belongs to config and is checked by trainer.verify_config.
    for key in ('protocol', 'initialization_sha256', 'params', 'target', 'unit', 'seed'):
        require(key in summary and key in config and summary[key] == config[key],
                'Summary/config mismatch: ' + key)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--device', default='cuda:0', choices=('cuda:0',))
    options = parser.parse_args()
    import numpy as np
    import torch
    from torch.utils.data import DataLoader, Dataset, Subset
    import train_fusion as trainer

    require(trainer.DATASET == 'QM9' and trainer.TARGET == TARGET and trainer.UNIT == UNIT, 'Wrong experiment target/unit')
    completion_barrier(ROOT)
    output = ROOT / 'results'
    require(not output.exists(), 'Results already exist; inspect instead of overwriting')
    protocol = read(ROOT / 'experiment_protocol.json')
    require(protocol['target'] == TARGET and protocol['unit'] == UNIT and protocol['seeds'] == list(SEEDS),
            'Experiment protocol target/unit/seeds mismatch')
    categories = read(ROOT / 'identity_categories.json')
    source_hashes = trainer.source_hashes()
    prepared, arguments, configs, final_tests, checkpoint_hashes = {}, {}, {}, {}, {}
    device = None
    for geometry in ('dft', 'mmff'):
        args = trainer.parse_args(['--data', str(ROOT / 'data' / geometry), '--out', str(output / 'unused'),
                                  '--device', options.device])
        if device is None:
            device = trainer.prepare_device(args)
        arguments[geometry] = args
        prepared[geometry] = trainer.prepare_dataset(args)
    dft, dft_ids, dft_targets, collate, split, split_ids, scaling = prepared['dft']
    mmff, mmff_ids, mmff_targets, _, mmff_split, mmff_split_ids, mmff_scaling = prepared['mmff']
    require(len(dft) == len(mmff) == 132486, 'Wrong matched cohort size')
    require({name: len(values) for name, values in split_ids.items()} == {'train': 105992, 'val': 13251, 'test': 13243},
            'Wrong frozen split sizes')
    require(split_ids == mmff_split_ids and scaling == mmff_scaling, 'DFT/MMFF split or scaling mismatch')
    require(set(dft.arrays) == set(mmff.arrays), 'DFT/MMFF cache schema mismatch')
    for name in dft.arrays:
        if name != 'points':
            np.testing.assert_array_equal(dft.arrays[name], mmff.arrays[name], err_msg=name)
    np.testing.assert_array_equal(dft_ids, mmff_ids)
    np.testing.assert_array_equal(dft_targets, mmff_targets)
    test_ids, test_targets = split_ids['test'], dft_targets[split['test']]
    tags = np.asarray([categories[str(mid)] for mid in test_ids])
    for geometry in ('dft', 'mmff'):
        dataset, ids, targets, _, _, _, _ = prepared[geometry]
        for seed in SEEDS:
            folder = ROOT / 'runs' / geometry / f'seed_{seed}'
            args = trainer.parse_args(['--data', str(ROOT / 'data' / geometry), '--out', str(folder),
                                      '--seed', str(seed), '--device', options.device])
            config = trainer.verify_config(args, folder, dataset, ids, targets, scaling,
                                           trainer.model_config(args, dataset.dims), split_ids)
            require(not config['diagnostic'] and config['provenance']['source_sha256'] == source_hashes,
                    'Diagnostic run or changed training sources')
            require(config['protocol']['cmcr'] is False and config['protocol']['mixed_precision'] is False,
                    'Expected frozen FP32 protocol without CMCR')
            summary = read(folder / 'summary.json')
            verify_summary_config(summary, config)
            require(summary['provenance']['config_sha256'] == sha(folder / 'config.json'), 'Config hash mismatch')
            require(sha(folder / 'test_predictions.csv') == summary['predictions_sha256'], 'Finalized prediction hash mismatch')
            saved_ids, saved_targets, saved_prediction = read_predictions(folder / 'test_predictions.csv')
            require(saved_ids == test_ids, 'Finalized test prediction order mismatch')
            np.testing.assert_array_equal(saved_targets, test_targets)
            recorded = summary['metrics']
            require(metrics(saved_targets, saved_prediction) ==
                    {'n': recorded['test_n'], 'mae': recorded['test_mae'], 'rmse': recorded['test_rmse']},
                    'Finalized metrics disagree with predictions')
            final = read(folder / 'finalization_result.json')
            require(final['config_sha256'] == sha(folder / 'config.json'), 'Finalization config hash mismatch')
            require(final['test']['ids'] == test_ids, 'Finalization test IDs mismatch')
            np.testing.assert_array_equal(final['test']['targets'], test_targets)
            np.testing.assert_array_equal(final['test']['predictions'], saved_prediction)
            for filename in ('best.pt', 'last.pt'):
                path = folder / filename
                digest = sha(path)
                checkpoint_hashes[path.relative_to(ROOT).as_posix()] = digest
                checkpoint = trainer.load_checkpoint(path)
                require(checkpoint['config'] == config and checkpoint['split_ids'] == split_ids,
                        filename + ': checkpoint config/split mismatch')
                if filename == 'best.pt':
                    require(digest == summary['provenance']['checkpoint_sha256'] == final['checkpoint_sha256'],
                            'Best checkpoint hash mismatch')
                    require(checkpoint['epoch'] == summary['epochs']['best'], 'Best epoch mismatch')
                else:
                    require(checkpoint['termination_reason'] in ('max_epochs', 'early_stopping'), 'Unfinished last checkpoint')
                    require(checkpoint['epoch'] == summary['epochs']['completed'] and
                            checkpoint['best_epoch'] == summary['epochs']['best'], 'Last checkpoint epoch mismatch')
                    require(all(row['train_complete_epoch'] for row in checkpoint['history']), 'Truncated training epoch')
                del checkpoint
            configs[geometry, seed], final_tests[geometry, seed] = config, final['test']
    for seed in SEEDS:
        for key in ('model', 'protocol', 'initialization_sha256', 'target', 'unit', 'dataset_size'):
            require(configs['dft', seed][key] == configs['mmff', seed][key], f'Paired initialization/protocol mismatch: {key}')
    raw_rows = dft.indices[split['test']]
    dft_points = dft.arrays['points'][raw_rows]
    lengths = dft.arrays['point_lengths'][raw_rows]
    etkdg_path = ROOT / 'data/etkdg_test.npz'
    etkdg = ordered_etkdg(etkdg_path, protocol['etkdg_test_sha256'], test_ids, lengths, dft_points)
    np.testing.assert_array_equal(etkdg[:, :, 4:], mmff.arrays['points'][mmff.indices[mmff_split['test']], :, 4:])

    class ETKDGTestView(Dataset):
        def __init__(self):
            self.original = Subset(dft, split['test'].tolist())
            self.points = torch.from_numpy(etkdg)

        def __len__(self):
            return len(self.original)

        def __getitem__(self, index):
            sample = self.original[index]
            sample['points'] = self.points[index]
            return sample

    loaders = {
        'dft': trainer.make_loaders(dft, split, collate, 128, None, ('test',))['test'],
        'mmff': trainer.make_loaders(mmff, mmff_split, collate, 128, None, ('test',))['test'],
        'etkdg': DataLoader(ETKDGTestView(), batch_size=128, shuffle=False, num_workers=0, collate_fn=collate),
    }
    output.mkdir()
    details, identity, drift = [], [], []
    started = time.perf_counter()
    environment = trainer.environment(device)
    for seed in SEEDS:
        for geometry in ('dft', 'mmff'):
            config = configs[geometry, seed]
            trainer.set_seed(seed)
            model = trainer.MultimodalModel(**config['model']).to(device)
            require(trainer.initialization_fingerprints(model) == config['initialization_sha256'],
                    'Reconstructed initialization mismatch')
            source = ROOT / 'runs' / geometry / f'seed_{seed}'
            checkpoint = trainer.load_checkpoint(source / 'best.pt')
            model.load_state_dict(checkpoint['model_state'], strict=True)
            del checkpoint
            for label, (train_geometry, test_geometry, display) in CONDITIONS.items():
                if train_geometry != geometry:
                    continue
                result = trainer.evaluate(model, loaders[test_geometry], device, scaling)
                require(result['ids'] == test_ids, 'ABCD prediction order mismatch')
                np.testing.assert_array_equal(result['targets'], test_targets)
                actual = metrics(result['targets'], result['predictions'])
                require(actual == result['metrics'], 'ABCD computed metrics mismatch')
                folder = output / 'predictions' / label / f'seed_{seed}'
                folder.mkdir(parents=True)
                trainer.write_predictions(folder / 'test_predictions.csv', result)
                persisted_ids, persisted_y, persisted_p = read_predictions(folder / 'test_predictions.csv')
                require(persisted_ids == test_ids and metrics(persisted_y, persisted_p) == actual, 'Saved prediction audit failed')
                np.testing.assert_array_equal(persisted_y, test_targets)
                row = {'label': label, 'condition': display, 'seed': seed, 'train_geometry': geometry,
                       'test_geometry': test_geometry, **actual}
                details.append(row)
                trainer.atomic_json(folder / 'summary.json', {
                    'status': 'completed', **row, 'dataset': 'QM9', 'target': TARGET, 'unit': UNIT,
                    'checkpoint_sha256': checkpoint_hashes[(source / 'best.pt').relative_to(ROOT).as_posix()],
                    'config_sha256': sha(source / 'config.json'),
                    'predictions_sha256': sha(folder / 'test_predictions.csv'),
                    'inference_seconds': result['seconds'], 'environment': environment})
                for category in sorted(set(tags)):
                    selected = tags == category
                    identity.append({'label': label, 'seed': seed, 'identity_category': str(category),
                                     **metrics(persisted_y[selected], persisted_p[selected])})
                if label in ('A_prime', 'D'):
                    previous = final_tests[geometry, seed]
                    delta = np.abs(persisted_p - np.asarray(previous['predictions'], dtype=np.float64))
                    drift.append({'label': label, 'seed': seed, 'mean_prediction_abs_delta': float(delta.mean()),
                                  'max_prediction_abs_delta': float(delta.max()),
                                  'mae_delta': actual['mae'] - previous['metrics']['mae'],
                                  'rmse_delta': actual['rmse'] - previous['metrics']['rmse']})
                trainer.atomic_json(output / 'evaluation_status.json', {'status': 'running', 'completed': len(details),
                                    'expected': 12, 'last_label': label, 'last_seed': seed})
                print(json.dumps(row, ensure_ascii=False), flush=True)
            del model
            torch.cuda.empty_cache()
    require(len(details) == 12 and len(checkpoint_hashes) == 12, 'Incomplete evaluation/checkpoint set')
    for path, digest in checkpoint_hashes.items():
        require(sha(ROOT / path) == digest, 'Checkpoint changed during evaluation: ' + path)
    require(trainer.source_hashes() == source_hashes, 'Training sources changed during evaluation')
    require(sha(etkdg_path) == protocol['etkdg_test_sha256'], 'ETKDG archive changed during evaluation')
    aggregate = aggregate_rows(details)
    identity_aggregate = aggregate_rows(identity, ('identity_category',))
    paired = paired_changes(details)
    report = {'status': 'completed', 'dataset': 'QM9', 'target': TARGET, 'unit': UNIT,
              'test_n': len(test_ids), 'seeds': list(SEEDS), 'split_sizes': {name: len(ids) for name, ids in split_ids.items()},
              'per_seed': details, 'aggregate': aggregate, 'paired_changes': paired,
              'identity_subsets': identity, 'identity_aggregate': identity_aggregate,
              'finalization_backend_drift': drift, 'environment': environment,
              'checkpoint_sha256': checkpoint_hashes, 'source_sha256': source_hashes,
              'evaluation_script_sha256': sha(__file__), 'experiment_protocol_sha256': sha(ROOT / 'experiment_protocol.json'),
              'etkdg_test_sha256': protocol['etkdg_test_sha256'], 'target_scaling': scaling,
              'fixed_split_sha256': dft.metadata['fixed_split_sha256'],
              'interpretation': {label: display for label, (_, _, display) in CONDITIONS.items()},
              'limitations': ['Original fixed random split filtered to common successful geometries; not structure-grouped.',
                              'Official QM9 exclusions not applied; identity differences reported by fixed subsets.',
                              'B is ETKDG-only inference on the same DFT checkpoint as A/C; no ETKDG training.',
                              'All labels remain original DFT property labels; coordinate substitution is not recomputation of electronic_spatial_extent.',
                              'Three seeds on one fixed split; sample SD is not evidence of statistical significance.',
                              'FP32 and no CMCR; CUDA operations need not be bitwise deterministic.'],
              'elapsed_seconds': time.perf_counter() - started}
    for filename, rows in (('per_seed.csv', details), ('comparison.csv', aggregate), ('paired_changes.csv', paired),
                           ('identity_subsets_per_seed.csv', identity), ('identity_subsets_aggregate.csv', identity_aggregate),
                           ('finalization_backend_drift.csv', drift)):
        write_csv(output / filename, rows)
    text = ['# QM9 电子空间范围 ⟨R²⟩：坐标质量比较', '',
            f'单位：{UNIT}；共同测试集 n=13243；种子 42、2025、3407；± 为种子间样本标准差。', '',
            '| 条件 | 训练→测试坐标 | MAE（mean ± SD） | RMSE（mean ± SD） |', '|---|---|---:|---:|']
    for row in aggregate:
        text.append(f"| {row['label'].replace('_prime', '′')} | {row['condition']} | {row['mae_mean']:.6f} ± {row['mae_sd']:.6f} | {row['rmse_mean']:.6f} ± {row['rmse_sd']:.6f} |")
    text.extend(['', 'A′、B′、C′ 使用每个种子的同一 DFT 模型最佳检查点；B′ 为未经 MMFF 优化的 ETKDGv3 构象，C′ 为 ETKDGv3+MMFF94。D 在 MMFF 坐标上从零训练，再在 MMFF 坐标上测试。四条件使用同一 GPU、同一 FP32 推理后端。', '',
                 '下列百分比先在同种子内计算，再汇总均值和样本标准差；正值表示误差增加，负值表示误差降低。', ''])
    for row in paired:
        if row['metric'] == 'MAE':
            label = row['comparison'].replace('_prime', '′').replace('_vs_', ' vs ')
            text.append(f"- {label}：MAE 变化 {row['mean_change_pct']:+.2f}% ± {row['sd_change_pct']:.2f}%。")
    text.extend(['', '132486 个共同成功样本沿用原固定随机划分，训练/验证/测试为 105992/13251/13243，未重新打乱；不称为结构分组划分，未应用官方 QM9 排除名单。身份差异按冻结分类另列表，不能将全部差异单独归因于坐标误差。', '',
                 '标签始终为原数据的 DFT 性质值，不是在 ETKDG/MMFF 构象上重新计算的量子化学标签。目标标准化仅拟合共同训练集。三种子结果不单独构成统计显著性证据。原 μ 结果未被覆盖。', '',
                 '逐分子预测、检查点哈希和核验记录见 comparison.json 及 predictions/；A′、D 与训练后首次评估的数值差异另见 finalization_backend_drift.csv。', ''])
    (output / '中文结果.md').write_text('\n'.join(text), encoding='utf-8')
    trainer.atomic_json(output / 'comparison.json', report)
    trainer.atomic_json(output / 'evaluation_status.json', {'status': 'completed', 'completed': 12, 'expected': 12})
    manifest = {path.relative_to(output).as_posix(): sha(path) for path in sorted(output.rglob('*')) if path.is_file()}
    trainer.atomic_json(output / 'SHA256.json', manifest)
    print(json.dumps({'status': 'completed', 'evaluations': 12, 'test_n': len(test_ids)}, ensure_ascii=False))


if __name__ == '__main__':
    main()
