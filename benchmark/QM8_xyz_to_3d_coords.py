import os
import re
import torch
import numpy as np
from data_config import get_config, load_inputs, validate_xyz_atoms


# =========================
# QM8 3D point-cloud loader
# =========================
# Same logic as the QM9 xyz loader:
#   1. read atom symbols and xyz coordinates from per-molecule xyz files;
#   2. center coordinates by subtracting the molecular centroid;
#   3. concatenate centered xyz coordinates, radial distance rho_i, and atom-type one-hot vectors;
#   4. pad all molecules to the maximum atom count and expose:
#        d_input, point_cloud, point_cloud_lengths
#
# Expected xyz filename format:
#   qm8_valid_00001.xyz
#   qm8_valid_00002.xyz
#   ...
#
# The first xyz file should correspond to the first row of QM8_clean.xlsx.

if get_config()["dataset"] != "qm8":
    raise ValueError("QM8 XYZ loader requires --dataset qm8.")

print("QM8 3D loader: xyz coordinates + rho + xyz atom one-hot; RDKit is not used.")


def fortran_float(value):
    """Convert Fortran-style floats, such as 2.1997*^-6, to Python floats."""
    return re.sub(r"(\d+\.?\d*)\*\^([+-]?\d+)", r"\1e\2", value)


def parse_qm8_xyz_and_center(file_path):
    """Read one QM8 xyz file and return centered coordinates in xyz atom order."""
    with open(file_path, "r") as f:
        lines = f.readlines()

    if len(lines) < 3:
        raise ValueError(f"Invalid xyz file with fewer than 3 lines: {file_path}")

    n_atom = int(lines[0].strip())
    if n_atom <= 0:
        raise ValueError(f"Nonpositive XYZ atom count in {file_path}")
    elements = []
    raw_coords = []

    atom_lines = lines[2: 2 + n_atom]
    if len(atom_lines) != n_atom:
        raise ValueError(f"Expected {n_atom} atom lines but found {len(atom_lines)} in {file_path}")

    for line in atom_lines:
        parts = line.split()
        if len(parts) < 4:
            raise ValueError(f"Invalid atom line in {file_path}: {line.rstrip()}")
        elements.append(parts[0])
        raw_coords.append([
            float(fortran_float(parts[1])),
            float(fortran_float(parts[2])),
            float(fortran_float(parts[3])),
        ])

    raw_coords = np.asarray(raw_coords, dtype=np.float32)
    if not np.isfinite(raw_coords).all():
        raise ValueError(f"Nonfinite coordinates in {file_path}")
    centroid = np.mean(raw_coords, axis=0, keepdims=True)
    centered_coords = raw_coords - centroid
    return centered_coords, elements


df, smiles_list, _target, xyz_paths = load_inputs()
total_files = len(smiles_list)

print(f"QM8 rows in label file: {total_files}")

# Build atom vocabulary directly from xyz element symbols.
# This does not use RDKit and keeps the one-hot atom type aligned with xyz atom order.
all_elements = []
for idx, filepath in enumerate(xyz_paths):
    _, xyz_elements = parse_qm8_xyz_and_center(filepath)
    validate_xyz_atoms(idx, xyz_elements)
    all_elements.extend(xyz_elements)


element_list = sorted(set(all_elements))
atom_type_to_idx = {atom: idx for idx, atom in enumerate(element_list)}
print(f"element_list: {element_list}")

point_clouds = []
point_cloud_lengths = []
qm8_xyz_files = []

for idx, filepath in enumerate(xyz_paths, start=1):
    coords, xyz_elements = parse_qm8_xyz_and_center(filepath)

    n_atoms = len(xyz_elements)

    atom_type_onehot = np.zeros((n_atoms, len(element_list)), dtype=np.float32)
    for atom_idx, symbol in enumerate(xyz_elements):
        type_idx = atom_type_to_idx.get(symbol)
        if type_idx is None:
            raise ValueError(f"Unknown atom type {symbol} in {filepath}")
        atom_type_onehot[atom_idx, type_idx] = 1.0

    # Since coordinates are already centered, rho_i = ||r_i^c||_2.
    rho = np.linalg.norm(coords, axis=1, keepdims=True).astype(np.float32)

    # Same feature logic as QM9: [centered_x, centered_y, centered_z, rho, atom_type_onehot].
    point_cloud_feature = np.concatenate(
        [coords.astype(np.float32), rho, atom_type_onehot],
        axis=1,
    )

    point_clouds.append(point_cloud_feature)
    point_cloud_lengths.append(n_atoms)
    qm8_xyz_files.append(os.path.basename(filepath))

    if idx % 5000 == 0:
        print(f"Processed {idx} files...")

if len(point_clouds) != total_files:
    raise ValueError(
        f"Number of processed point clouds ({len(point_clouds)}) does not match label rows ({total_files})."
    )

max_atoms = max(point_cloud_lengths)
d_input = point_clouds[0].shape[1]

print(f"d_input = {d_input}, type: {type(d_input)}")
print(f"Maximum atoms in dataset: {max_atoms}")
print(f"Feature dimension per atom: {d_input}")
print(f"Total molecules processed: {len(point_clouds)}")

padded_point_clouds = []
for pc, length in zip(point_clouds, point_cloud_lengths):
    if length < max_atoms:
        pad_size = max_atoms - length
        padding = np.zeros((pad_size, pc.shape[1]), dtype=np.float32)
        padded_pc = np.vstack([pc, padding])
    else:
        padded_pc = pc
    padded_point_clouds.append(padded_pc)

point_cloud = [torch.tensor(pc, dtype=torch.float32) for pc in padded_point_clouds]
point_cloud_lengths = torch.tensor(point_cloud_lengths, dtype=torch.long)
qm8_element_list = element_list
