"""Label-free source and cache checks extracted unchanged."""
import hashlib
from pathlib import Path
from collections import Counter
import numpy as np
from rdkit import Chem
import feature_reference as reference
ELEMENTS = reference.ELEMENTS

def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for data in iter(lambda: handle.read(1 << 20), b""):
            digest.update(data)
    return digest.hexdigest()


def canonical_identity(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return {"parse_ok": False, "input_smiles": smiles}
    return {
        "parse_ok": True, "input_smiles": smiles,
        "canonical_isomeric_smiles": Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True),
        "canonical_nonstereo_smiles": Chem.MolToSmiles(mol, canonical=True, isomericSmiles=False),
        "element_counts_with_h": dict(sorted(Counter(a.GetSymbol() for a in Chem.AddHs(mol).GetAtoms()).items())),
        "formal_charge": sum(a.GetFormalCharge() for a in mol.GetAtoms()),
        "radical_electrons": sum(a.GetNumRadicalElectrons() for a in mol.GetAtoms()),
        "fragments": len(Chem.GetMolFrags(mol)),
        "chiral_centers": len(Chem.FindMolChiralCenters(mol, includeUnassigned=True)),
    }


def compare_identity(first, second):
    okay = first["parse_ok"] and second["parse_ok"]
    return {
        "both_parse": okay,
        "canonical_isomeric_equal": okay and first["canonical_isomeric_smiles"] == second["canonical_isomeric_smiles"],
        "canonical_nonstereo_equal": okay and first["canonical_nonstereo_smiles"] == second["canonical_nonstereo_smiles"],
        "element_counts_with_h_equal": okay and first["element_counts_with_h"] == second["element_counts_with_h"],
    }


def point_features(elements, coordinates):
    xyz = np.asarray(coordinates, dtype=np.float32)
    xyz -= xyz.mean(axis=0)
    one_hot = np.eye(5, dtype=np.float32)[[ELEMENTS.index(element) for element in elements]]
    return np.concatenate([xyz, np.linalg.norm(xyz, axis=1, keepdims=True), one_hot], axis=1)


def graph_matches_cache(mol, index, arrays, metadata):
    atom_maps = [{value: offset for offset, value in enumerate(values)} for values in metadata["atom_schema"].values()]
    bond_maps = [{value: offset for offset, value in enumerate(values)} for values in metadata["bond_schema"].values()]
    rings = mol.GetRingInfo()
    graph_x = []
    for atom in mol.GetAtoms():
        vector = []
        for j, (value, mapping) in enumerate(zip(reference.atom_values(atom, rings), atom_maps)):
            if j == 6:
                vector.append(float(atom.GetIsAromatic()))
            vector.extend(float(k == mapping[value]) for k in range(len(mapping)))
        vector.append(atom.GetMass()/100.0)
        graph_x.append(vector)
    edge_attr, edge_index = [], []
    for bond in mol.GetBonds():
        vector = []
        for j, (value, mapping) in enumerate(zip(reference.bond_values(bond, rings), bond_maps)):
            if j == 1:
                vector.append(float(bond.GetIsConjugated()))
            vector.extend(float(k == mapping[value]) for k in range(len(mapping)))
        edge_attr.extend([vector, vector])
        a, b = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        edge_index.extend([[a, b], [b, a]])
    n0, n1 = arrays["node_offsets"][index:index+2]
    e0, e1 = arrays["edge_offsets"][index:index+2]
    expected_x = np.asarray(graph_x, dtype=np.float32)
    expected_attr = np.asarray(edge_attr, dtype=np.float32).reshape(-1, metadata["dims"]["edge_dim"])
    expected_index = np.asarray(edge_index, dtype=np.int16).reshape(-1, 2).T
    return {
        "node_features_exact": np.array_equal(expected_x, arrays["graph_x"][n0:n1]),
        "edge_features_exact": np.array_equal(expected_attr, arrays["edge_attr"][e0:e1]),
        "edge_indices_exact": np.array_equal(expected_index, arrays["edge_index"][:, e0:e1]),
        "heavy_graph_node_count": mol.GetNumAtoms(), "graph_undirected_bond_count": mol.GetNumBonds(),
    }


def qm8_selected_xyz(source, selected_indices):
    selected_indices = set(selected_indices)
    found = {}
    record_index = 0
    with source.open(encoding="utf-8") as handle:
        while True:
            line = handle.readline()
            if not line:
                break
            if not line.strip():
                continue
            count = int(line.strip())
            header = handle.readline().strip()
            atom_rows = [handle.readline().split() for _ in range(count)]
            if record_index in selected_indices:
                fields = header.split(maxsplit=3)
                if len(fields) != 4 or fields[0] != "Molecule":
                    raise ValueError(f"Bad QM8 source header at record {record_index}")
                found[record_index] = {
                    "official_gdb9_id": int(fields[1]), "header_smiles": fields[3],
                    "elements": [row[0] for row in atom_rows],
                    "coordinates": [[float(x) for x in row[1:4]] for row in atom_rows],
                }
            record_index += 1
    if set(found) != selected_indices:
        raise ValueError("Selected QM8 source record missing")
    return found


def qm9_xyz(source, row, molecule_id):
    raw = source.read_bytes()
    if hashlib.sha256(raw).hexdigest() != row["xyz_sha256"]:
        raise ValueError(f"QM9 XYZ hash mismatch for {molecule_id}")
    lines = raw.decode("utf-8").splitlines()
    count = int(lines[0])
    header = lines[1].split()
    if header[:2] != ["gdb", str(molecule_id)]:
        raise ValueError(f"QM9 XYZ ID mismatch for {molecule_id}")
    atom_rows = [line.split() for line in lines[2:count+2]]
    smiles = lines[count+3].split()
    if smiles != [row["canonical_SMILES"], row["isomeric_SMILES"]]:
        raise ValueError(f"QM9 manifest vs XYZ SMILES mismatch for {molecule_id}")
    return {
        "elements": [atom[0] for atom in atom_rows],
        "coordinates": [[float(value.replace("*^", "e")) for value in atom[1:4]] for atom in atom_rows],
        "xyz_smiles_1": smiles[0], "xyz_smiles_2": smiles[1], "xyz_sha256": hashlib.sha256(raw).hexdigest(),
    }
