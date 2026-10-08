"""Attach passed author-source comparison to metadata without changing NPZ."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

from prepare_qm8 import sha256


def finalize(directory):
    metadata_path = directory / "metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    official_path = directory / "official_comparison.json"
    official = json.loads(official_path.read_text(encoding="utf-8"))
    n = metadata["sample_count"]
    required_full = ["retained_records_checked", "retained_same_element_order", "retained_same_composition",
                     "retained_raw_matches_official_nonstereo", "retained_raw_matches_official_isomeric",
                     "retained_xyz_id_matches_label_id"]
    failures = [key for key in required_full if official[key] != n]
    if official["raw_CC2_four_columns_exact_rows"] != 21786:
        failures.append("raw_CC2_four_columns_exact_rows")
    points_report_path = directory / "official_points_validation.json"
    points_report = json.loads(points_report_path.read_text(encoding="utf-8"))
    if points_report["status"] != "PASS" or points_report["official_coordinate_rows_exactly_reconstructed"] != n:
        failures.append("official_cache_points")
    if failures:
        raise ValueError(f"Official comparison requires review before metadata finalization: {failures}")
    if sha256(directory / "cache.npz") != metadata["cache_sha256"]:
        raise ValueError("Cache hash changed")
    if points_report["cache_sha256"] != metadata["cache_sha256"]:
        raise ValueError("Official points validation refers to another cache")
    download_manifest_path = directory / "official_source" / "download_manifest.json"
    downloads = json.loads(download_manifest_path.read_text(encoding="utf-8"))
    for item in downloads:
        content = (download_manifest_path.parent / item["name"]).read_bytes()
        if len(content) != item["size"] or hashlib.sha256(content).hexdigest() != item["sha256"]:
            raise ValueError(f"Official download changed: {item['name']}")
        git_blob = hashlib.sha1(f"blob {len(content)}\0".encode()+content).hexdigest()
        if git_blob != item["git_blob_sha1"]:
            raise ValueError(f"Official Git blob identity mismatch: {item['name']}")
        item["immutable_api_url"] = "https://api.github.com/repos/raghurama123/ExcitedStatesQM8/git/blobs/" + git_blob
    metadata.pop("engineering_pilot_only", None)
    metadata["experiment_protocol"] = "QM8 retained-21747 author-verified raw-SMILES with conservative group split v1"
    metadata["source_verification_status"] = "author_labels_structures_and_coordinates_verified"
    metadata["official_source_comparison"] = {
        "report": "official_comparison.json", "report_sha256": sha256(official_path),
        "record_manifest": "official_record_comparison.csv", "record_manifest_sha256": official["comparison_csv_sha256"],
        "official_label_sha256": official["official_label_sha256"], "official_xyz_sha256": official["official_xyz_sha256"],
        "matched_records": n, "raw_CC2_exact_match_rows": 21786,
        "historical_local_max_abs_coordinate_difference_angstrom": official["max_abs_coordinate_difference_angstrom"],
        "historical_local_coordinate_mismatch_records": official["records_with_coord_error_above_0_000051_angstrom"],
        "current_points_source": "author original XYZ, with only the reference float32 feature operations",
        "current_points_exact_reconstruction_rows": points_report["official_coordinate_rows_exactly_reconstructed"],
        "current_points_validation_report": points_report_path.name, "current_points_validation_sha256": sha256(points_report_path),
    }
    metadata["official_source_files"] = downloads
    metadata["scientific_citations"] = ["https://doi.org/10.1063/1.4928757", "https://doi.org/10.1038/sdata.2014.22"]
    metadata["point_coordinate_unit"] = "angstrom"
    metadata["historical_local_xyz_source"] = metadata.get("historical_local_xyz_source", metadata["xyz_source"])
    metadata["xyz_source"] = str(directory / "official_source" / "XYZ_B3LYP_631G2dfp.xyz")
    metadata["manifest_xyz_columns_role"] = "manifest.csv xyz_filename/xyz_sha256 describe historical local input only; current coordinate ID/source mapping is official_record_comparison.csv"
    metadata["point_features"][:4] = ["centered_x_angstrom", "centered_y_angstrom", "centered_z_angstrom", "rho_angstrom"]
    metadata["coordinate_source_method"] = "Author original B3LYP/6-31G(2df,p) XYZ; no SDF decimal rounding; reference float32 centering/radius/element encoding only"
    metadata["identity_policy"] = "Cache ID = original_sdf_index + 1 (local retained raw-record index); corresponding verified official gdb9 ID is saved in official_record_comparison.csv"
    metadata["target_methods"] = {name: "RI-CC2/def2TZVP" for name in ("E1-CC2", "E2-CC2", "f1-CC2", "f2-CC2")}
    for index, target in enumerate(metadata["target_names"]):
        if "CAM" in target:
            metadata["target_methods"][target] = "LR-TDDFT(CAMB3LYP/def2TZVP)"
        elif "PBE0" in target:
            raw_column = 4 + (index - 4)
            exact_cols = official["raw_target_column_comparison"][raw_column]["all_exact_columns"]
            if len(exact_cols) == 1:
                source_col = exact_cols[0]
                basis = "def2SVP" if 5 <= source_col <= 8 else "def2TZVP" if 9 <= source_col <= 12 else None
                if basis:
                    metadata["target_methods"][target] = f"LR-TDDFT(PBE0/{basis})"
    metadata["limitations"] = [
        "The preserved 21747-record cohort excludes the same 39 historical raw rows; the historical exclusion reason is not independently recovered.",
        "Historical local XYZ differs from author coordinates in 36 records beyond rounding tolerance; the current cache uses author XYZ for every retained record and preserves the historical cache separately.",
        "Conservative raw-or-clean canonical grouping prevents known duplicate candidates crossing splits; this is a specified local protocol, not a claim to reproduce an official benchmark split.",
    ]
    unresolved_targets = [name for name in metadata["target_names"] if name not in metadata["target_methods"]]
    if unresolved_targets:
        metadata["limitations"].append("Official method/basis attribution remains unresolved for: " + ", ".join(unresolved_targets))
    metadata["metadata_finalizer_sha256"] = sha256(__file__)
    metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"status": metadata["source_verification_status"], "cache_sha256_unchanged": metadata["cache_sha256"], "metadata_sha256": sha256(metadata_path)}, indent=2))


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path, required=True)
    args = parser.parse_args()
    finalize(args.directory)
