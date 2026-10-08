"""Verify source identities, generate conformers, and build paired geometry caches.

The numerical generation functions are extracted unchanged from the experiment.
This command replaces the original author-specific filesystem orchestration.
"""
from __future__ import annotations

import argparse
from collections import Counter
import csv
import json
import os
from pathlib import Path
import platform

for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[name] = "1"

import numpy as np
from rdkit import rdBase
from SmilesPE.pretokenizer import atomwise_tokenizer

import feature_reference
import primitives
import qm8
import qm9
import source_checks as checks

HERE = Path(__file__).resolve().parent.parent


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def save(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def source_rows(dataset, cache, xyz, arrays, metadata, split):
    with (cache / "manifest.csv").open(encoding="utf-8", newline="") as stream:
        records = {int(row["molecule_id"]): row for row in csv.DictReader(stream)}
    partitions = {mid: name for name, ids in split.items() for mid in ids}
    if len(partitions) != len(arrays["ids"]) or set(partitions) != set(map(int, arrays["ids"])):
        raise ValueError("Source cache and published original split population differ")
    official = checks.qm8_selected_xyz(xyz, (int(mid) - 1 for mid in arrays["ids"])) if dataset == "QM8" else None
    for i, mid in enumerate(map(int, arrays["ids"])):
        raw = records[mid]
        if dataset == "QM9":
            ref = checks.qm9_xyz(xyz / raw["xyz_filename"], raw, mid)
            smiles = raw["canonical_SMILES"]
            identity = checks.canonical_identity(smiles)
            relaxed = checks.canonical_identity(raw["isomeric_SMILES"])
            compared = checks.compare_identity(identity, relaxed)
            category = ("identical" if compared["canonical_isomeric_equal"] else
                        "stereo_only" if compared["canonical_nonstereo_equal"] else
                        "nonstereo_difference" if compared["both_parse"] else "relaxed_parse_failure")
            header_equal = smiles == ref["xyz_smiles_1"]
            tokens = atomwise_tokenizer(smiles)
            encoded = np.zeros(metadata["dims"]["smiles_length"], dtype=np.int16)
            encoded[:len(tokens)] = [metadata["token_to_idx"][t] for t in tokens]
            np.testing.assert_array_equal(encoded, arrays["smiles_ids"][i])
        else:
            ref = official[mid - 1]
            smiles = raw["raw_smiles"]
            identity = checks.canonical_identity(smiles)
            header = checks.canonical_identity(ref["header_smiles"])
            header_equal = identity["canonical_isomeric_smiles"] == header["canonical_isomeric_smiles"]
            category = "identical" if header_equal else "source_identity_warning"
        mol = feature_reference.canonical_mol(smiles, mid)
        graph = checks.graph_matches_cache(mol, i, arrays, metadata)
        if not all(graph[k] for k in ("node_features_exact", "edge_features_exact", "edge_indices_exact")):
            raise ValueError(f"Source graph does not reproduce cached graph for ID {mid}")
        dft = checks.point_features(ref["elements"], ref["coordinates"])
        length = int(arrays["point_lengths"][i])
        np.testing.assert_array_equal(dft, arrays["points"][i, :length])
        if arrays["points"][i, length:].any() or not header_equal:
            raise ValueError(f"Invalid DFT padding or source identity for ID {mid}")
        counts = dict(sorted(Counter(ref["elements"]).items()))
        yield {"dataset": dataset, "id": mid, "input_index": i, "test_index": i,
               "cache_index": i, "split": partitions[mid], "input_smiles": smiles,
               "canonical_isomeric_smiles": identity["canonical_isomeric_smiles"],
               "dft_length": length, "dft_element_counts": counts,
               "input_header_identity_equal": header_equal,
               "composition_matches_dft": identity["element_counts_with_h"] == counts,
               "identity_category": category}


def generate(dataset, rows, out, shape):
    function = qm9.generate if dataset == "QM9" else qm8.generate
    points = {key: np.lib.format.open_memmap(out / f"{key}_points.npy", mode="w+", dtype=np.float32, shape=shape)
              for key in ("etkdg", "mmff")}
    for value in points.values():
        value[:] = np.nan
    masks = {key: np.zeros(len(rows), dtype=bool) for key in points}
    primitives.warmup()
    with (out / "per_molecule.jsonl").open("x", encoding="utf-8") as stream:
        for i, row in enumerate(rows):
            result, generated = function(row, shape[1])
            result = {key: None if isinstance(value, float) and not np.isfinite(value) else value
                      for key, value in result.items()}
            for key in points:
                if result[key + "_available"]:
                    points[key][i] = generated[key]
                    masks[key][i] = True
            # Preserve QC errors in the record; do not substitute DFT/UFF coordinates.
            stream.write(json.dumps(result, ensure_ascii=False, allow_nan=False) + "\n")
            if (i + 1) % 1000 == 0:
                stream.flush()
                print(json.dumps({"generated": i + 1, "total": len(rows)}), flush=True)
    for value in points.values():
        value.flush()
    ids = np.asarray([row["id"] for row in rows], dtype=np.int64)
    np.savez_compressed(out / "availability.npz", ids=ids, **masks)
    if dataset == "QM8":
        # Historical E1-CC2 CPU evaluator consumes this original archive schema.
        np.savez_compressed(out / "etkdg_all.npz", ids=ids, points=points["etkdg"],
                            lengths=np.asarray([row["dft_length"] for row in rows]), available=masks["etkdg"])
    return points, masks


def build(dataset, original, old_meta, split, rows, points, masks, out):
    eligible = masks["etkdg"] & masks["mmff"]
    ids = original["ids"]
    kept = set(map(int, ids[eligible]))
    matched = {key: [mid for mid in values if mid in kept] for key, values in split.items()}
    expected = read(HERE / "splits" / f"{dataset.lower()}_matched.json")
    if matched != expected:
        raise ValueError("Generated success cohort differs from the published cohort. Raw records are retained; "
                         "review RDKit versions and failures. Do not silently resplit or replace failed geometries.")
    by_id = {int(mid): i for i, mid in enumerate(ids)}
    if dataset == "QM8":
        groups = {name: set(map(int, original["group_ids"][[by_id[mid] for mid in values]]))
                  for name, values in matched.items()}
        if any(groups[a] & groups[b] for a, b in (("train", "val"), ("train", "test"), ("val", "test"))):
            raise ValueError("QM8 structure group leakage")
    # Reuse the original metadata schema, while clearly recording new generation provenance.
    template_name = "qm9_mu" if dataset == "QM9" else "qm8_e1cc2"
    catalog = read(HERE / "experiments.json")
    data = out / "data"
    for condition in ("dft", "mmff"):
        folder = data / condition
        folder.mkdir(parents=True)
        template = read(HERE / catalog[template_name]["metadata"][condition])
        for key in ("dims", "atom_schema", "bond_schema", "target_names", "point_features"):
            if old_meta[key] != template[key]:
                raise ValueError(f"Original feature/target schema changed: {key}")
        save(folder / "fixed_split_ids.json", matched)
        selected_points = original["points"] if condition == "dft" else points["mmff"]
        np.savez_compressed(folder / "cache.npz", **{**original, "points": selected_points, "eligible_mask": eligible})
        template["cache_sha256"] = checks.sha256(folder / "cache.npz")
        template["fixed_split_sha256"] = checks.sha256(folder / "fixed_split_ids.json")
        template["release_generation"] = {"manifest_sha256": checks.sha256(out / "input_manifest.jsonl"),
                                           "protocol_sha256": checks.sha256(out / "generation_protocol.json")}
        save(folder / "metadata.json", template)
    test_set = set(matched["test"])
    take = np.asarray([i for i, mid in enumerate(ids) if int(mid) in test_set])
    np.savez_compressed(data / "etkdg_test.npz", ids=ids[take], points=points["etkdg"][take])
    if dataset == "QM9":
        save(data / "identity_categories.json", {str(row["id"]): row["identity_category"] for row in rows})
    save(out / "generation_summary.json", {"status": "completed", "dataset": dataset,
         "original": len(ids), "matched": len(kept), "split_sizes": {k: len(v) for k, v in matched.items()},
         "generated": {k: int(v.sum()) for k, v in masks.items()}, "no_target_based_selection": True})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, choices=("QM8", "QM9"))
    parser.add_argument("--cache", required=True, type=Path, help="Original DFT cache.npz, metadata.json and manifest.csv")
    parser.add_argument("--xyz", required=True, type=Path, help="QM9 XYZ directory or QM8 official multi-record XYZ file")
    parser.add_argument("--out", required=True, type=Path, help="New output directory on a disk with sufficient space")
    args = parser.parse_args()
    meta = read(args.cache / "metadata.json")
    if checks.sha256(args.cache / "cache.npz") != meta["cache_sha256"]:
        raise ValueError("Original cache checksum mismatch")
    if checks.sha256(args.cache / "manifest.csv") != meta["manifest_sha256"]:
        raise ValueError("Original manifest checksum mismatch")
    if args.dataset == "QM8" and checks.sha256(args.xyz) != meta["points_source_sha256"]:
        raise ValueError("QM8 official DFT XYZ checksum mismatch")
    with np.load(args.cache / "cache.npz", allow_pickle=False) as saved:
        arrays = {key: saved[key] for key in saved.files}
    split = read(HERE / "splits" / f"{args.dataset.lower()}_original.json")
    rows = list(source_rows(args.dataset, args.cache, args.xyz, arrays, meta, split))
    args.out.mkdir(parents=True, exist_ok=False)
    with (args.out / "input_manifest.jsonl").open("x", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
    save(args.out / "generation_protocol.json", {"dataset": args.dataset,
         "python": platform.python_version(), "numpy": np.__version__, "rdkit": rdBase.rdkitVersion,
         "source_cache_sha256": checks.sha256(args.cache / "cache.npz"),
         "seed_recipe": "SHA256(geometry-pilot-v1|20260928|DATASET|ID) first4 big endian mod2147483647",
         "generation": "1 ETKDGv3; one random-coordinate retry only on failure; paired MMFF94 300+700 iterations",
         "note": "Release orchestration; frozen per-molecule numerical functions are retained unchanged."})
    points, masks = generate(args.dataset, rows, args.out, arrays["points"].shape)
    build(args.dataset, arrays, meta, split, rows, points, masks, args.out)
    print(json.dumps(read(args.out / "generation_summary.json")))


if __name__ == "__main__":
    main()
