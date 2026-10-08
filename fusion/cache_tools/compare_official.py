"""Compare the retained local QM8 records to downloaded author data.

No cache values are changed. The report keeps every observed discrepancy.
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from collections import Counter
from pathlib import Path

import numpy as np
from rdkit import Chem

sys.dont_write_bytecode = True
from prepare_qm8 import TARGETS, load_table, sha256, reference


def numeric_table(path):
    headers, rows, line_numbers = [], [], []
    for line_number, text in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        tokens = text.split()
        if not tokens:
            continue
        try:
            row = [float(token.replace("D", "E")) for token in tokens]
        except ValueError:
            headers.append({"line": line_number, "text": text})
            continue
        if len(row) >= 16:
            rows.append(row)
            line_numbers.append(line_number)
        else:
            headers.append({"line": line_number, "text": text})
    widths = Counter(map(len, rows))
    if len(widths) != 1:
        raise ValueError(f"Inconsistent numeric row widths: {widths}")
    return headers, np.asarray(rows, dtype=np.float64), line_numbers


def xyz_records(path):
    records = []
    with path.open(encoding="utf-8") as handle:
        while True:
            line = handle.readline()
            if not line:
                break
            if not line.strip():
                continue
            count = int(line.strip())
            comment = handle.readline().strip()
            atoms = [handle.readline().split() for _ in range(count)]
            if not all(len(atom) >= 4 for atom in atoms):
                raise ValueError(f"Bad XYZ block after {len(records)} records")
            elements = [atom[0] for atom in atoms]
            coords = np.asarray([[float(value) for value in atom[1:4]] for atom in atoms], dtype=np.float64)
            if not np.isfinite(coords).all():
                raise ValueError("Nonfinite official coordinates")
            records.append((comment, elements, coords))
    return records


def compare(directory, official_directory):
    metadata = json.loads((directory / "metadata.json").read_text(encoding="utf-8"))
    with (directory / "manifest.csv").open(encoding="utf-8", newline="") as handle:
        manifest = list(csv.DictReader(handle))
    raw_header, raw_rows = load_table(Path(metadata["source_table"]))
    official_labels = official_directory / "gdb8_22k_elec_spec.txt"
    official_xyz = official_directory / "XYZ_B3LYP_631G2dfp.xyz"
    headers, values, label_lines = numeric_table(official_labels)
    if len(values) != len(raw_rows):
        raise ValueError(f"Official/source label row count differs: {len(values)} vs {len(raw_rows)}")
    if values.shape[1] != 17 or not np.array_equal(values[:, 0], values[:, 0].astype(np.int64)):
        raise ValueError("Expected official gdb9 index plus 16 target columns")
    official_ids = values[:, 0].astype(np.int64)
    if len(set(official_ids)) != len(official_ids):
        raise ValueError("Official gdb9 IDs are not unique")
    original = np.asarray([[float(value) for value in row[1:]] for row in raw_rows])
    # Report every corresponding numeric column without silently presuming the
    # historical local PBE0 duplicate headers identify a particular basis set.
    column_comparison = []
    for raw_col in range(original.shape[1]):
        candidates = []
        for official_col in range(values.shape[1]):
            difference = np.abs(original[:, raw_col]-values[:, official_col])
            candidates.append({"official_numeric_column_0based": official_col,
                               "exact_matches": int(np.count_nonzero(difference == 0)),
                               "max_abs_difference": float(difference.max())})
        candidates.sort(key=lambda value: (-value["exact_matches"], value["max_abs_difference"]))
        column_comparison.append({"raw_target_column_0based": raw_col, "raw_header": raw_header[raw_col+1], "best_correspondence": candidates[0], "all_exact_columns": [value["official_numeric_column_0based"] for value in candidates if value["exact_matches"] == len(values)]})
    official_records = xyz_records(official_xyz)
    if len(official_records) != len(raw_rows):
        raise ValueError(f"Official coordinate/source label row count differs: {len(official_records)} vs {len(raw_rows)}")
    results = []
    for index, row in enumerate(manifest):
        record_index = int(row["original_sdf_index_0based"])
        comment, elements, official_coords = official_records[record_index]
        fields = comment.split(maxsplit=3)
        if len(fields) != 4 or fields[0] != "Molecule":
            raise ValueError(f"Unexpected official XYZ comment: {comment}")
        official_id = int(fields[1])
        if official_id != official_ids[record_index]:
            raise ValueError(f"Official XYZ and label IDs disagree at index {record_index}")
        official_smiles = fields[3]
        official_mol = reference.canonical_mol(official_smiles, official_id)
        official_canonical = Chem.MolToSmiles(official_mol, canonical=True, isomericSmiles=True)
        official_nonstereo = Chem.MolToSmiles(official_mol, canonical=True, isomericSmiles=False)
        local_path = Path(metadata.get("historical_local_xyz_source", metadata["xyz_source"])) / row["xyz_filename"]
        local_records = xyz_records(local_path)
        assert len(local_records) == 1
        _, local_elements, local_coords = local_records[0]
        same_elements = elements == local_elements
        same_composition = Counter(elements) == Counter(local_elements)
        same_shape = official_coords.shape == local_coords.shape
        max_abs_difference = float(np.max(np.abs(official_coords-local_coords))) if same_shape and same_elements else None
        # SDF atom coordinates have four decimal places and downstream files
        # retain float32 representations; report this transformation explicitly.
        rounded_float32 = np.round(official_coords, 4).astype(np.float32).astype(np.float64)
        rounded_max_abs_difference = float(np.max(np.abs(rounded_float32-local_coords))) if same_shape and same_elements else None
        results.append({"molecule_id": int(row["molecule_id"]), "cache_row_0based": index,
                        "official_record_index_0based": record_index,
                        "official_gdb9_id": official_id, "official_smiles": official_smiles,
                        "official_canonical_isomeric_smiles": official_canonical,
                        "raw_matches_official_isomeric": official_canonical == row["raw_canonical_isomeric_smiles"],
                        "raw_matches_official_nonstereo": official_nonstereo == row["raw_canonical_nonstereo_smiles"],
                        "official_xyz_comment": comment, "official_label_line": label_lines[record_index],
                        "same_element_order": same_elements, "same_composition": same_composition,
                        "max_abs_coordinate_difference_angstrom": max_abs_difference,
                        "max_abs_difference_after_four_decimal_float32": rounded_max_abs_difference})
        if (index+1) % 5000 == 0:
            print(json.dumps({"phase": "official_coordinates", "completed": index+1}), flush=True)
    result_path = directory / "official_record_comparison.csv"
    with result_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(results[0]))
        writer.writeheader()
        writer.writerows(results)
    coord_errors = [record["max_abs_coordinate_difference_angstrom"] for record in results if record["max_abs_coordinate_difference_angstrom"] is not None]
    rounded_errors = [record["max_abs_difference_after_four_decimal_float32"] for record in results if record["max_abs_difference_after_four_decimal_float32"] is not None]
    summary = {
        "official_label_sha256": sha256(official_labels), "official_xyz_sha256": sha256(official_xyz),
        "official_label_rows": len(values), "official_numeric_columns": values.shape[1], "official_headers": headers,
        "official_xyz_records": len(official_records), "retained_records_checked": len(results),
        "raw_target_column_comparison": column_comparison,
        "retained_same_element_order": sum(record["same_element_order"] for record in results),
        "retained_raw_matches_official_isomeric": sum(record["raw_matches_official_isomeric"] for record in results),
        "retained_raw_matches_official_nonstereo": sum(record["raw_matches_official_nonstereo"] for record in results),
        "retained_xyz_id_matches_label_id": len(results),
        "official_target_column_order": ["gdb9_id", "E1-CC2", "E2-CC2", "f1-CC2", "f2-CC2", "E1-PBE0/def2SVP", "E2-PBE0/def2SVP", "f1-PBE0/def2SVP", "f2-PBE0/def2SVP", "E1-PBE0/def2TZVP", "E2-PBE0/def2TZVP", "f1-PBE0/def2TZVP", "f2-PBE0/def2TZVP", "E1-CAM", "E2-CAM", "f1-CAM", "f2-CAM"],
        "raw_CC2_four_columns_exact_rows": int(np.count_nonzero(np.all(original[:, :4] == values[:, 1:5], axis=1))),
        "retained_same_composition": sum(record["same_composition"] for record in results),
        "max_abs_coordinate_difference_angstrom": max(coord_errors) if coord_errors else None,
        "max_abs_difference_after_four_decimal_float32": max(rounded_errors) if rounded_errors else None,
        "records_with_coord_error_above_0_000051_angstrom": sum(value>0.000051 for value in coord_errors),
        "records_with_rounded_float32_error_above_5_1e_9_angstrom": sum(value>5.1e-9 for value in rounded_errors),
        "comparison_csv_sha256": sha256(result_path), "comparison_script_sha256": sha256(__file__),
        "first_xyz_comment": official_records[0][0], "last_xyz_comment": official_records[-1][0],
    }
    (directory / "official_comparison.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--directory", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--official-directory", type=Path, default=Path(__file__).resolve().parent / "official_source")
    args = parser.parse_args()
    compare(args.directory, args.official_directory)
