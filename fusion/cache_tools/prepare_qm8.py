"""Build a portable QM8 cache without changing sources or excluding retained rows."""
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import platform
import re
import sys
import time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
from rdkit import Chem, rdBase

sys.dont_write_bytecode = True
FEATURE_SOURCE = Path(__file__).with_name("prepare_data.py")
spec = importlib.util.spec_from_file_location("qm9_feature_reference", FEATURE_SOURCE)
reference = importlib.util.module_from_spec(spec)
spec.loader.exec_module(reference)
ELEMENTS = reference.ELEMENTS
VERSION = "qm8-raw-smiles-dft-v1"
TARGETS = ["E1-CC2", "E2-CC2", "f1-CC2", "f2-CC2", "E1-PBE0", "E2-PBE0", "f1-PBE0", "f2-PBE0", "E1-CAM", "E2-CAM", "f1-CAM", "f2-CAM"]
sha256 = reference.sha256


def load_table(path):
    values = list(reference.read_xlsx(path))
    return values[0], values[1:]


def log(phase, **values):
    print(json.dumps({"phase": phase, **values}, ensure_ascii=False), flush=True)


def load_xyz(item):
    row, xyz_dir = item
    clean_id = int(row["clean_data_row_1based"])
    original_index = int(row["raw_excel_row"]) - 2
    filename = f"qm8_valid_{clean_id:05d}.xyz"
    raw = (xyz_dir / filename).read_bytes()
    lines = raw.decode("utf-8").splitlines()
    count = int(lines[0])
    header = re.fullmatch(r"original_sdf_index=(\d+);\s*xyz_index=(\d+);\s*smiles=(.*)", lines[1])
    if not header or (int(header[1]), int(header[2]), header[3]) != (original_index, clean_id, row["clean_smiles"]):
        raise ValueError(f"XYZ identity mismatch: {filename}")
    atoms = [line.split() for line in lines[2:] if line.strip()]
    if len(atoms) != count or any(len(atom) != 4 for atom in atoms):
        raise ValueError(f"XYZ row count/format mismatch: {filename}")
    elements = [atom[0] for atom in atoms]
    if set(elements) - set(ELEMENTS):
        raise ValueError(f"XYZ unsupported elements: {filename}")
    coords = np.asarray([[float(x) for x in atom[1:]] for atom in atoms], dtype=np.float32)
    if not np.isfinite(coords).all():
        raise ValueError(f"XYZ nonfinite coordinates: {filename}")
    coords -= coords.mean(axis=0)
    one_hot = np.eye(len(ELEMENTS), dtype=np.float32)[[ELEMENTS.index(e) for e in elements]]
    features = np.concatenate([coords, np.linalg.norm(coords, axis=1, keepdims=True), one_hot], axis=1)
    return features, Counter(elements), hashlib.sha256(raw).hexdigest()


def build(raw_table, clean_table, xyz_dir, output, workers, audit_dir):
    started = time.monotonic()
    output.mkdir(parents=True, exist_ok=True)
    raw_header, raw_rows = load_table(raw_table)
    clean_header, clean_rows = load_table(clean_table)
    if len(raw_rows) != 21786 or len(clean_rows) != 21747 or clean_header[1:] != TARGETS:
        raise ValueError("Source row counts or explicit target names differ from audited inputs")
    with (audit_dir / "qm8_row_mapping.csv").open(encoding="utf-8-sig", newline="") as handle:
        mapping = list(csv.DictReader(handle))
    with (audit_dir / "qm8_xyz_identity_manifest.csv").open(encoding="utf-8-sig", newline="") as handle:
        prior_xyz = list(csv.DictReader(handle))
    if len(mapping) != 21747 or len(prior_xyz) != 21747:
        raise ValueError("Unexpected mapping or prior manifest size")
    audited_tables = json.loads((audit_dir / "qm8_table_summary.json").read_text(encoding="utf-8"))
    expected_hashes = {path: audited_tables["books"][path.name]["sha256"] for path in (raw_table, clean_table)}
    for path, expected in expected_hashes.items():
        if sha256(path) != expected:
            raise ValueError(f"Audited source hash changed: {path}")
    source_cols = [raw_header.index(name) for name in TARGETS]
    targets = []
    ids = []
    for idx, row in enumerate(mapping):
        if int(row["clean_data_row_1based"]) != idx + 1 or int(row["clean_excel_row"]) != idx + 2:
            raise ValueError("Mapping reordered or malformed")
        original_index = int(row["raw_excel_row"]) - 2
        raw_row = raw_rows[original_index]
        clean_row = clean_rows[idx]
        if raw_row[0] != row["raw_smiles"] or clean_row[0] != row["clean_smiles"]:
            raise ValueError(f"Mapping SMILES mismatch at clean row {idx+1}")
        raw_y = [float(raw_row[col]) for col in source_cols]
        if raw_y != [float(value) for value in clean_row[1:]]:
            raise ValueError(f"Raw/clean target mismatch at clean row {idx+1}")
        targets.append(raw_y)
        ids.append(original_index + 1)
    ids = np.asarray(ids, dtype=np.int64)
    targets = np.asarray(targets, dtype=np.float64)
    if len(np.unique(ids)) != len(ids) or not np.all(np.diff(ids) > 0) or not np.isfinite(targets).all():
        raise ValueError("Bad stable IDs or targets")
    log("sources_verified", rows=len(ids))
    atom_names = ["element", "degree", "total_h", "formal_charge", "hybridization", "chirality", "atom_ring_size"]
    bond_names = ["bond_type", "stereo", "bond_ring_size"]
    atom_sets = [set() for _ in atom_names]
    bond_sets = [set() for _ in bond_names]
    mols, tokens, formulas, canonical, nonstereo = [], [], [], [], []
    clean_nonstereo = []
    changes = []
    node_counts, edge_counts = [], []
    raw_groups, clean_groups = defaultdict(list), defaultdict(list)
    for idx, row in enumerate(mapping):
        smiles = row["raw_smiles"]
        if smiles != smiles.replace("\xa0", "").replace(" ", ""):
            raise ValueError(f"Whitespace would alter raw SMILES {ids[idx]}")
        seq = reference.atomwise_tokenizer(smiles)
        if "".join(seq) != smiles:
            raise ValueError(f"Tokenizer lost raw SMILES text {ids[idx]}")
        mol = reference.canonical_mol(smiles, ids[idx])
        clean_mol = reference.canonical_mol(row["clean_smiles"], ids[idx])
        mols.append(mol)
        tokens.append(seq)
        formula = Counter(a.GetSymbol() for a in Chem.AddHs(mol).GetAtoms())
        clean_formula = Counter(a.GetSymbol() for a in Chem.AddHs(clean_mol).GetAtoms())
        formulas.append(formula)
        can = Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True)
        plain = Chem.MolToSmiles(mol, canonical=True, isomericSmiles=False)
        clean_can = Chem.MolToSmiles(clean_mol, canonical=True, isomericSmiles=True)
        clean_plain = Chem.MolToSmiles(clean_mol, canonical=True, isomericSmiles=False)
        canonical.append(can)
        nonstereo.append(plain)
        clean_nonstereo.append(clean_plain)
        raw_groups[plain].append(idx)
        clean_groups[clean_plain].append(idx)
        changes.append({"id": int(ids[idx]), "clean_data_row_1based": idx+1,
                        "raw_excel_row": int(row["raw_excel_row"]),
                        "text_changed": smiles != row["clean_smiles"],
                        "canonical_isomeric_changed": can != clean_can,
                        "canonical_nonstereo_changed": plain != clean_plain,
                        "elemental_composition_changed": formula != clean_formula,
                        "raw_formula": json.dumps(dict(sorted(formula.items()))),
                        "clean_formula": json.dumps(dict(sorted(clean_formula.items()))),
                        "raw_radical_electrons": sum(a.GetNumRadicalElectrons() for a in mol.GetAtoms()),
                        "clean_radical_electrons": sum(a.GetNumRadicalElectrons() for a in clean_mol.GetAtoms())})
        rings = mol.GetRingInfo()
        for atom in mol.GetAtoms():
            for values, value in zip(atom_sets, reference.atom_values(atom, rings)):
                values.add(value)
        for bond in mol.GetBonds():
            for values, value in zip(bond_sets, reference.bond_values(bond, rings)):
                values.add(value)
        node_counts.append(mol.GetNumAtoms())
        edge_counts.append(mol.GetNumBonds() * 2)
        if (idx+1) % 5000 == 0:
            log("schema", completed=idx+1)
    atom_sets[1] = set(range(max(atom_sets[1])+1))
    atom_sets[2] = set(range(max(atom_sets[2])+1))
    atom_sets[6].add(0)
    bond_sets[2].add(0)
    atom_schema = {name: sorted(values) for name, values in zip(atom_names, atom_sets)}
    bond_schema = {name: sorted(values) for name, values in zip(bond_names, bond_sets)}
    vocab = {"PAD": 0}
    vocab.update({token: i+1 for i, token in enumerate(sorted({t for seq in tokens for t in seq}))})
    smiles_length = max(map(len, tokens))
    smiles_ids = np.zeros((len(ids), smiles_length), dtype=np.int16)
    for idx, seq in enumerate(tokens):
        smiles_ids[idx, :len(seq)] = [vocab[token] for token in seq]
    atom_maps = [{value: idx for idx, value in enumerate(atom_schema[name])} for name in atom_names]
    bond_maps = [{value: idx for idx, value in enumerate(bond_schema[name])} for name in bond_names]
    node_dim = sum(map(len, atom_maps)) + 2
    edge_dim = sum(map(len, bond_maps)) + 1
    node_offsets = np.r_[0, np.cumsum(node_counts)].astype(np.int64)
    edge_offsets = np.r_[0, np.cumsum(edge_counts)].astype(np.int64)
    graph_x = np.zeros((node_offsets[-1], node_dim), dtype=np.float32)
    edge_index = np.zeros((2, edge_offsets[-1]), dtype=np.int16)
    edge_attr = np.zeros((edge_offsets[-1], edge_dim), dtype=np.float32)
    for idx, mol in enumerate(mols):
        rings = mol.GetRingInfo()
        for atom in mol.GetAtoms():
            vec = []
            for j, (value, categories) in enumerate(zip(reference.atom_values(atom, rings), atom_maps)):
                if j == 6:
                    vec.append(float(atom.GetIsAromatic()))
                vec.extend(float(k == categories[value]) for k in range(len(categories)))
            vec.append(atom.GetMass()/100.)
            graph_x[node_offsets[idx]+atom.GetIdx()] = vec
        for bond in mol.GetBonds():
            vec = []
            for j, (value, categories) in enumerate(zip(reference.bond_values(bond, rings), bond_maps)):
                if j == 1:
                    vec.append(float(bond.GetIsConjugated()))
                vec.extend(float(k == categories[value]) for k in range(len(categories)))
            edge_pos = edge_offsets[idx]+2*bond.GetIdx()
            a, b = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
            edge_index[:, edge_pos:edge_pos+2] = [[a, b], [b, a]]
            edge_attr[edge_pos:edge_pos+2] = vec
    log("graphs", nodes=int(node_offsets[-1]), directed_edges=int(edge_offsets[-1]), node_dim=node_dim, edge_dim=edge_dim)
    max_atoms = max(sum(formula.values()) for formula in formulas)
    points = np.zeros((len(ids), max_atoms, 9), dtype=np.float32)
    point_lengths = np.zeros(len(ids), dtype=np.int16)
    xyz_hashes = []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for idx, (features, formula, digest) in enumerate(pool.map(load_xyz, ((row, xyz_dir) for row in mapping))):
            if formula != formulas[idx]:
                raise ValueError(f"Raw SMILES/XYZ formula differs: {ids[idx]}")
            if digest != prior_xyz[idx]["xyz_sha256"]:
                raise ValueError(f"Audited XYZ bytes changed: {ids[idx]}")
            point_lengths[idx] = len(features)
            points[idx, :len(features)] = features
            xyz_hashes.append(digest)
            if (idx+1) % 5000 == 0:
                log("xyz", completed=idx+1)
    # Conservative identity candidate groups: connected components sharing either
    # original or cleaned nonstereochemical canonical SMILES. Labels never enter grouping.
    parent = list(range(len(ids)))
    def find(value):
        while parent[value] != value:
            parent[value] = parent[parent[value]]
            value = parent[value]
        return value
    for groups in (raw_groups, clean_groups):
        for members in groups.values():
            for value in members[1:]:
                a, b = find(members[0]), find(value)
                parent[max(a, b)] = min(a, b)
    group_ids = np.asarray([ids[find(i)] for i in range(len(ids))], dtype=np.int64)
    grouped = defaultdict(list)
    for idx, group_id in enumerate(group_ids):
        grouped[int(group_id)].append(int(ids[idx]))
    duplicate_groups = [{"group_id": group_id, "ids": members} for group_id, members in grouped.items() if len(members)>1]
    (output / "duplicate_groups.json").write_text(json.dumps({"policy": "union of matching raw or clean canonical nonstereo SMILES; conservative candidate grouping; no labels used", "groups": duplicate_groups}, indent=2), encoding="utf-8")
    with (output / "structure_audit.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(changes[0]))
        writer.writeheader()
        writer.writerows(changes)
    manifest = output / "manifest.csv"
    with manifest.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["molecule_id", "cache_row_0based", "clean_data_row_1based", "raw_excel_row", "original_sdf_index_0based", "xyz_filename", "xyz_sha256", "point_atoms", "group_id", "raw_smiles", "clean_smiles", "raw_canonical_isomeric_smiles", "raw_canonical_nonstereo_smiles", *TARGETS])
        for idx, row in enumerate(mapping):
            writer.writerow([ids[idx], idx, idx+1, row["raw_excel_row"], ids[idx]-1, f"qm8_valid_{idx+1:05d}.xyz", xyz_hashes[idx], point_lengths[idx], group_ids[idx], row["raw_smiles"], row["clean_smiles"], canonical[idx], nonstereo[idx], *targets[idx]])
    arrays = dict(ids=ids, targets=targets, graph_x=graph_x, node_offsets=node_offsets, edge_offsets=edge_offsets,
                  edge_index=edge_index, edge_attr=edge_attr, smiles_ids=smiles_ids, points=points, point_lengths=point_lengths,
                  group_ids=group_ids)
    if any(not np.isfinite(array).all() for array in arrays.values()):
        raise ValueError("Nonfinite cached array")
    cache = output / "cache.npz"
    np.savez_compressed(cache, **arrays)
    with np.load(cache, allow_pickle=False) as archive:
        if set(archive.files) != set(arrays) or any(not np.array_equal(archive[key], value) for key, value in arrays.items()):
            raise ValueError("Saved cache differs from in-memory arrays")
    digest_lines = "".join(f"qm8_valid_{i+1:05d}.xyz {digest}\n" for i, digest in enumerate(xyz_hashes))
    structural_summary = {key: sum(bool(row[key]) for row in changes) for key in ("text_changed", "canonical_isomeric_changed", "canonical_nonstereo_changed", "elemental_composition_changed")}
    structural_summary.update(raw_rows_with_radicals=sum(row["raw_radical_electrons"]>0 for row in changes),
                              clean_rows_with_radicals=sum(row["clean_radical_electrons"]>0 for row in changes),
                              candidate_duplicate_groups=len(duplicate_groups),
                              rows_in_candidate_duplicate_groups=sum(len(group["ids"]) for group in duplicate_groups),
                              unique_group_count=len(grouped))
    metadata = {
        "preprocessing_version": VERSION, "dataset": "QM8", "sample_count": len(ids), "molecule_count": len(ids),
        "engineering_pilot_only": True, "source_table": str(raw_table), "source_table_sha256": sha256(raw_table),
        "clean_table": str(clean_table), "clean_table_sha256": sha256(clean_table), "xyz_source": str(xyz_dir),
        "xyz_directory_manifest_sha256": hashlib.sha256(digest_lines.encode()).hexdigest(),
        "cache_sha256": sha256(cache), "manifest_sha256": sha256(manifest), "prepare_script_sha256": sha256(__file__),
        "feature_reference": str(FEATURE_SOURCE), "feature_reference_sha256": sha256(FEATURE_SOURCE),
        "python": platform.python_version(), "numpy": np.__version__, "rdkit": rdBase.rdkitVersion,
        "dims": dict(node_dim=node_dim, edge_dim=edge_dim, vocab_size=len(vocab), smiles_length=smiles_length, point_dim=9, max_atoms=max_atoms),
        "atom_schema": atom_schema, "bond_schema": bond_schema, "token_to_idx": vocab,
        "target_names": TARGETS, "target_units": {name: ("hartree" if name.startswith("E") else "dimensionless") for name in TARGETS},
        "target_unit_basis": "Author repository README specifies excitation energies in hartree; workbook values retained without conversion",
        "target_unit_source": "https://github.com/raghurama123/ExcitedStatesQM8",
        "coordinate_source_method": "Author repository specifies B3LYP/6-31G(2df,p) geometries; local XYZ not independently compared to original SDF",
        "point_features": ["centered_x_source_units", "centered_y_source_units", "centered_z_source_units", "rho_source_units", *["element_"+e for e in ELEMENTS]],
        "point_coordinate_unit": "source XYZ units; expected angstrom but not independently verified against original SDF",
        "schema_policy": "Same QM9 reference feature functions and ordering; per-dataset categorical/token schema; no target statistics",
        "selection_policy": "All 21747 previously retained clean rows, same order; no new exclusions or deduplication",
        "identity_policy": "Stable integer id = original_sdf_index_0based + 1; NOT a verified official gdb9 ID; exact raw/clean labels and XYZ headers/hashes checked",
        "smiles_policy": "Use original-table raw SMILES for both sequence and graph; retain cleaned SMILES only in provenance",
        "alignment_policy": "Graph canonical atom order and XYZ atom order need not match: model pools each modality at molecule level. AddHs(raw graph) elemental composition equals XYZ for all rows.",
        "group_policy": "group_ids are conservative connected components of raw or clean canonical nonstereo SMILES; computed without labels; retain members together for leakage-aware splits",
        "structural_audit": structural_summary,
        "limitations": ["Original SDF bond block and explicit official molecule IDs not independently recovered.", "Composition and provenance do not prove graph/geometry bond-order consistency.", "Clean SMILES changes are discarded from features; raw-vs-clean structural differences are recorded.", "PBE0 method/basis attribution remains unresolved; this experiment selects E1-CC2 only.", "39 historical exclusions are preserved without asserting their removal reason."],
        "arrays": {key: {"shape": list(value.shape), "dtype": str(value.dtype)} for key, value in arrays.items()},
        "elapsed_s": time.monotonic()-started,
    }
    (output / "metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    log("complete", rows=len(ids), dims=metadata["dims"], structural_audit=structural_summary, cache_bytes=cache.stat().st_size, elapsed_s=metadata["elapsed_s"])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-table", type=Path, required=True)
    parser.add_argument("--clean-table", type=Path, required=True)
    parser.add_argument("--xyz-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--workers", type=int, default=12)
    parser.add_argument("--audit-dir", type=Path, required=True, help="Directory containing the three original audited mapping/table files")
    args = parser.parse_args()
    build(args.raw_table, args.clean_table, args.xyz_dir, args.output, args.workers, args.audit_dir)
