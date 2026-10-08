"""One explicit table and row-to-XYZ mapping for all benchmark modalities."""

import csv
import json
import os
from functools import lru_cache
from pathlib import Path


def get_config():
    value = os.environ.get("TRIMOGATE_BENCHMARK_CONFIG")
    if value is None:
        raise RuntimeError("Use python benchmark/run.py --help to configure this run.")
    return json.loads(value)


def validate_xyz_map(smiles, rows, xyz_dir):
    """Require a one-to-one, explicitly ordered mapping; never drop rows."""
    if len(rows) != len(smiles):
        raise ValueError("The XYZ map and label table have different row counts.")
    root = Path(xyz_dir).resolve()
    paths = []
    for index, (smi, row) in enumerate(zip(smiles, rows)):
        if int(row["row_index"]) != index or row["smiles"] != smi:
            raise ValueError(f"XYZ mapping does not match table row {index}.")
        relative = Path(row["xyz_file"])
        path = (root / relative).resolve()
        if relative.is_absolute() or not path.is_relative_to(root):
            raise ValueError(f"XYZ map row {index} must reference a file inside xyz-dir.")
        if not path.is_file():
            raise FileNotFoundError(f"Missing XYZ file for table row {index}: {relative}")
        paths.append(path)
    if len(set(paths)) != len(paths):
        raise ValueError("Each label row must map to a distinct XYZ file.")
    return paths


@lru_cache(maxsize=1)
def load_inputs():
    import numpy as np
    import pandas as pd
    from rdkit import Chem

    config = get_config()
    path = Path(config["table"])
    frame = pd.read_csv(path) if path.suffix.lower() == ".csv" else pd.read_excel(path)
    smiles = frame[config["smiles_column"]].tolist()
    target = pd.to_numeric(frame[config["target"]], errors="raise")
    if not len(frame) or not np.isfinite(target.to_numpy(dtype=float)).all():
        raise ValueError("The table must contain finite numeric targets and at least one row.")
    for index, smi in enumerate(smiles):
        if not isinstance(smi, str) or not smi or Chem.MolFromSmiles(smi) is None:
            raise ValueError(f"Invalid SMILES at row {index}; prepare a jointly filtered input table.")
    with open(config["xyz_map"], newline="", encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle))
    paths = validate_xyz_map(smiles, rows, config["xyz_dir"])
    return frame, smiles, target, paths


def validate_xyz_atoms(row_index, elements):
    """Check composition against the mapped record, including explicit hydrogens."""
    from collections import Counter
    from rdkit import Chem

    smiles = load_inputs()[1][row_index]
    molecule = Chem.AddHs(Chem.MolFromSmiles(smiles))
    expected = Counter(atom.GetSymbol() for atom in molecule.GetAtoms())
    if Counter(elements) != expected:
        raise ValueError(f"XYZ/SMILES atom composition mismatch at table row {row_index}.")
    if not set(elements).issubset({"C", "F", "H", "N", "O"}):
        raise ValueError(f"Unexpected QM8/QM9 element at table row {row_index}.")
