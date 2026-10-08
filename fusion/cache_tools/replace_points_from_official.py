"""Preserve the historical cache and replace only points from author QM8 XYZ."""
from __future__ import annotations

import csv
import json
import shutil
import sys
from pathlib import Path

import numpy as np

sys.dont_write_bytecode = True
from compare_official import xyz_records
from prepare_qm8 import ELEMENTS, sha256


def replace(directory):
    metadata_path = directory / "metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    old_path = directory / "cache_local_xyz_before_official.npz"
    cache_path = directory / "cache.npz"
    if not old_path.exists():
        if sha256(cache_path) != metadata["cache_sha256"]:
            raise ValueError("Current cache hash mismatch before preservation")
        shutil.copy2(cache_path, old_path)
    old_sha256 = sha256(old_path)
    if old_sha256 != "c02fbb26fedd7a1ecc00672d3976da52cfe218e6c370b8b3e06f0225f66207ca":
        raise ValueError("Historical local cache is not the validated artifact")
    with np.load(old_path, allow_pickle=False) as archive:
        previous = {key: archive[key] for key in archive.files}
    with (directory / "manifest.csv").open(encoding="utf-8", newline="") as handle:
        manifest = list(csv.DictReader(handle))
    with (directory / "official_record_comparison.csv").open(encoding="utf-8", newline="") as handle:
        official_mapping = list(csv.DictReader(handle))
    source = directory / "official_source" / "XYZ_B3LYP_631G2dfp.xyz"
    if sha256(source) != "01c20f132d6215a856955df01bf1fd65a646cd531fbda190b0a21e5c85c4c39e":
        raise ValueError("Official XYZ source hash mismatch")
    records = xyz_records(source)
    points = np.zeros_like(previous["points"])
    lengths = np.zeros_like(previous["point_lengths"])
    changes = []
    for index, (row, mapped) in enumerate(zip(manifest, official_mapping)):
        assert int(row["molecule_id"]) == int(previous["ids"][index]) == int(mapped["molecule_id"])
        record_index = int(row["original_sdf_index_0based"])
        assert record_index == int(mapped["official_record_index_0based"])
        comment, elements, positions = records[record_index]
        assert int(comment.split()[1]) == int(mapped["official_gdb9_id"])
        positions = positions.astype(np.float32)
        positions -= positions.mean(axis=0)
        one_hot = np.eye(len(ELEMENTS), dtype=np.float32)[[ELEMENTS.index(element) for element in elements]]
        features = np.concatenate([positions, np.linalg.norm(positions, axis=1, keepdims=True), one_hot], axis=1)
        lengths[index] = len(features)
        points[index, :len(features)] = features
        changes.append(float(np.max(np.abs(points[index] - previous["points"][index]))))
    assert np.array_equal(lengths, previous["point_lengths"])
    arrays = {**previous, "points": points, "point_lengths": lengths}
    np.savez_compressed(cache_path, **arrays)
    with np.load(cache_path, allow_pickle=False) as archive:
        saved = {key: archive[key] for key in archive.files}
    unchanged = []
    for key in previous:
        if key not in ("points", "point_lengths"):
            if not np.array_equal(previous[key], saved[key]):
                raise ValueError(f"Unexpected array change: {key}")
            unchanged.append(key)
    if not np.array_equal(lengths, saved["point_lengths"]) or not np.array_equal(points, saved["points"]):
        raise ValueError("Official point array changed on save")
    # Independently regenerate each coordinate block from source text, checking
    # original elements, ID mapping, float32 arithmetic, radius and zero padding.
    for index, row in enumerate(official_mapping):
        comment, elements, source_xyz = records[int(row["official_record_index_0based"])]
        n = len(elements)
        expected_xyz = np.asarray(source_xyz, dtype=np.float32)
        expected_xyz = expected_xyz - np.mean(expected_xyz, axis=0)
        assert np.array_equal(saved["points"][index, :n, :3], expected_xyz)
        assert np.array_equal(saved["points"][index, :n, 3], np.sqrt(np.sum(expected_xyz**2, axis=1)))
        expected_elements = np.asarray([[float(element == known) for known in ELEMENTS] for element in elements], dtype=np.float32)
        assert np.array_equal(saved["points"][index, :n, 4:], expected_elements)
        assert (saved["points"][index, n:] == 0).all()
        assert saved["point_lengths"][index] == n
    summary = {
        "status": "PASS", "sample_count": len(manifest), "official_coordinate_rows_exactly_reconstructed": len(manifest),
        "cache_sha256": sha256(cache_path), "previous_local_cache_sha256": old_sha256,
        "unchanged_arrays": unchanged, "point_lengths_unchanged": True,
        "official_xyz_sha256": sha256(source), "official_xyz_git_blob_sha1": "edcd8b2ff7dc9f9a91fa6cb6d83151a666ebc4c2",
        "max_abs_point_feature_change_from_local": max(changes), "records_with_point_feature_changes": sum(value != 0 for value in changes),
        "script_sha256": sha256(__file__), "no_samples_excluded_or_reordered": True,
    }
    (directory / "official_points_validation.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    metadata["cache_sha256"] = summary["cache_sha256"]
    metadata["previous_local_cache_sha256"] = old_sha256
    metadata["points_source"] = "official_source/XYZ_B3LYP_631G2dfp.xyz"
    metadata["points_source_sha256"] = summary["official_xyz_sha256"]
    metadata["points_replacement_script_sha256"] = summary["script_sha256"]
    metadata["points_validation_sha256"] = sha256(directory / "official_points_validation.json")
    metadata["preprocessing_version"] = "qm8-author-xyz-raw-smiles-v2"
    metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path, required=True)
    args = parser.parse_args()
    replace(args.directory)
