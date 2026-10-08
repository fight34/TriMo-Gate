import torch
import numpy as np
from rdkit import Chem
from collections import Counter
from torch_geometric.data import Data
from data_config import load_inputs


df, smiles_list, y, _xyz_paths = load_inputs()
print(y.name)

# atom feature lists
all_elements = []
all_num_heavy_atom = []
all_num_h = []
all_formal_charges = []
all_hybridizations = []
all_chiralities = []
all_atom_ring_sizes = []
all_bond_ring_sizes = []

# bond feature lists
all_bond_types = []
all_bond_stereos = []

invalid_smiles = []
valid_smiles = []
valid_labels = []


def get_chirality_type(x):
    chiral_tag = x.GetChiralTag()
    return {
        Chem.ChiralType.CHI_UNSPECIFIED: 'CHI_UNSPECIFIED',
        Chem.ChiralType.CHI_TETRAHEDRAL_CW: 'CHI_TETRAHEDRAL_CW',
        Chem.ChiralType.CHI_TETRAHEDRAL_CCW: 'CHI_TETRAHEDRAL_CCW',
        Chem.ChiralType.CHI_OTHER: 'CHI_OTHER'
    }.get(chiral_tag, 'CHI_UNSPECIFIED')


def get_min_ring_sizes(mols):
    """
    返回：
    - min_atom_ring_sizes: 每个原子的最小环大小；不在环中为 0
    - min_bond_ring_sizes: 每条键的最小环大小；不在环中为 0
    """
    ring_info = mols.GetRingInfo()

    min_atom_ring_size = []
    for atom_idx in range(mols.GetNumAtoms()):
        if ring_info.NumAtomRings(atom_idx) > 0:
            min_atom_ring_size.append(ring_info.MinAtomRingSize(atom_idx))
        else:
            min_atom_ring_size.append(0)

    min_bond_ring_size = []
    for bond_idx in range(mols.GetNumBonds()):
        if ring_info.NumBondRings(bond_idx) > 0:
            min_bond_ring_size.append(ring_info.MinBondRingSize(bond_idx))
        else:
            min_bond_ring_size.append(0)

    return min_atom_ring_size, min_bond_ring_size


for idx, (smiles, label) in enumerate(zip(smiles_list, y)):

    mol = Chem.MolFromSmiles(smiles)

    if mol is None:
        print(f"Invalid SMILES: {smiles}")
        invalid_smiles.append({
            "data_index": idx,
            "excel_row": idx + 2,
            "smiles": smiles,
            "label": label,
            "reason": "RDKit MolFromSmiles/RemoveHs sanitize failed"
        })
        raise ValueError(f"Invalid SMILES at table row {idx}; no modality may silently drop rows.")
    valid_smiles.append(smiles)
    valid_labels.append(label)

    min_atom_ring_sizes, min_bond_ring_sizes = get_min_ring_sizes(mol)

    # Retrieve atom Information
    for atom in mol.GetAtoms():
        all_elements.append(atom.GetSymbol())
        all_num_heavy_atom.append(atom.GetDegree())
        all_num_h.append(atom.GetTotalNumHs())
        all_formal_charges.append(atom.GetFormalCharge())
        all_hybridizations.append(str(atom.GetHybridization()))
        all_chiralities.append(get_chirality_type(atom))
        all_atom_ring_sizes.append(min_atom_ring_sizes[atom.GetIdx()])

    # Retrieve bond Information
    for bond in mol.GetBonds():
        all_bond_types.append(str(bond.GetBondType()))
        all_bond_stereos.append(str(bond.GetStereo()))
        all_bond_ring_sizes.append(min_bond_ring_sizes[bond.GetIdx()])



# atom type
element_list = sorted(set(all_elements))
print(f"element_list: {element_list}")

# heavy atom
degree_counts = Counter(all_num_heavy_atom)
max_degree = max(all_num_heavy_atom)
print(f"Max number of heavy atom bonded: {max_degree}")
print("number of heavy atom bonded distribution:", dict(degree_counts))
num_heavy_atom_list = list(range(0, max_degree + 1))
print(f"num_heavy_atom_list: {num_heavy_atom_list}")

# hydrogen
num_h_counts = Counter(all_num_h)
max_num_h = max(all_num_h)
print(f"Max number of H bonded: {max_num_h}")
print("number of H bonded distribution:", dict(num_h_counts))
num_h_list = list(range(0, max_num_h + 1))
print(f"num_h_list: {num_h_list}")

# formal charge
formal_charge_counts = Counter(all_formal_charges)
print("Formal charge distribution:", dict(formal_charge_counts))
formal_charge_list = sorted(set(all_formal_charges))
print(f"formal_charge_list: {formal_charge_list}")

# hybridization
hybridization_counts = Counter(all_hybridizations)
print("Hybridization distribution:", dict(hybridization_counts))
hybridization_list = sorted(set(all_hybridizations))
print(f"hybridization_list: {hybridization_list}")

# chirality
chirality_counts = Counter(all_chiralities)
print("Chirality distribution:", dict(chirality_counts))
chirality_list = sorted(set(all_chiralities))
print(f"chirality_list: {chirality_list}")

# atom ring sizes
atom_ring_size_counts = Counter(all_atom_ring_sizes)
print("Atom ring size distribution:", dict(atom_ring_size_counts))
atom_ring_size_list = sorted(set(all_atom_ring_sizes + [0]))
print(f"atom_ring_size_list: {atom_ring_size_list}")

# bond ring sizes
bond_ring_size_counts = Counter(all_bond_ring_sizes)
print("Bond ring size distribution:", dict(bond_ring_size_counts))
bond_ring_size_list = sorted(set(all_bond_ring_sizes + [0]))
print(f"bond_ring_size_list: {bond_ring_size_list}")

# bond types
bond_type_counts = Counter(all_bond_types)
print("Bond type distribution:", dict(bond_type_counts))
bond_type_list = sorted(set(all_bond_types))
print(f"bond_type_list: {bond_type_list}")

# bond stereochemistry
bond_stereo_counts = Counter(all_bond_stereos)
print("Bond stereo distribution:", dict(bond_stereo_counts))
bond_stereo_list = sorted(set(all_bond_stereos))
print(f"bond_stereo_list: {bond_stereo_list}")

# atomic features mapped to dictionaries
atom_type_map = {element: idx for idx, element in enumerate(element_list)}
num_heavy_atom_map = {degree: idx for idx, degree in enumerate(num_heavy_atom_list)}
num_h_map = {num: idx for idx, num in enumerate(num_h_list)}
formal_charge_map = {charge: idx for idx, charge in enumerate(formal_charge_list)}
hybridization_map = {hybrid: idx for idx, hybrid in enumerate(hybridization_list)}
chirality_map = {chiral: idx for idx, chiral in enumerate(chirality_list)}
atom_ring_size_map = {size: idx for idx, size in enumerate(atom_ring_size_list)}
bond_ring_size_map = {size: idx for idx, size in enumerate(bond_ring_size_list)}

# bond features mapped to dictionaries
bond_stereo_map = {bs: idx for idx, bs in enumerate(bond_stereo_list)}
bond_type_map = {bt: idx for idx, bt in enumerate(bond_type_list)}


def get_one_hot(value, value_map, num_classes):
    idx = value_map.get(value, -1)
    if idx == -1:
        print(f"Warning: Unknown feature value '{value}'！")
        return [0] * num_classes

    one_hot = [0] * num_classes
    one_hot[idx] = 1
    return one_hot


_node_dim = (
    len(element_list)
    + len(num_heavy_atom_list)
    + len(num_h_list)
    + len(formal_charge_list)
    + len(hybridization_list)
    + len(chirality_list)
    + 1
    + len(atom_ring_size_list)
    + 1
)

_edge_dim = (
    len(bond_type_list)
    + 1
    + len(bond_stereo_list)
    + len(bond_ring_size_list)
)

print(f"Precomputed node_dim = {_node_dim}, edge_dim = {_edge_dim}")


# SMILES to graph
def smiles_to_graph(x):

    # SMILES to canonical mol
    mols = Chem.MolFromSmiles(x)
    if mols is None:
        return None

    canonical_smiles = Chem.MolToSmiles(mols, isomericSmiles=True, canonical=True)
    mols = Chem.MolFromSmiles(canonical_smiles)

    min_ring_sizes, min_ring_sizes_bond = get_min_ring_sizes(mols)

    # get atomic features
    node_features = []
    for atom_idx, atom in enumerate(mols.GetAtoms()):

        # atom type
        atom_type = atom.GetSymbol()
        atom_type_one_hot = get_one_hot(atom_type, atom_type_map, len(element_list))

        # Number of heavy_atom forming bonds
        num_heavy_atom = atom.GetDegree()
        num_heavy_atom_one_hot = get_one_hot(num_heavy_atom, num_heavy_atom_map, len(num_heavy_atom_list))

        # Number of hydrogen atoms forming bonds
        num_h = atom.GetTotalNumHs()
        num_h_one_hot = get_one_hot(num_h, num_h_map, len(num_h_list))

        # Formal charge
        formal_charge = atom.GetFormalCharge()
        formal_charge_one_hot = get_one_hot(formal_charge, formal_charge_map, len(formal_charge_list))

        # Hybridization
        hybridization = str(atom.GetHybridization())
        hybridization_one_hot = get_one_hot(hybridization, hybridization_map, len(hybridization_list))

        # chirality
        chirality_type = get_chirality_type(atom)
        chirality_one_hot = get_one_hot(chirality_type, chirality_map, len(chirality_list))

        # aromaticity
        is_aromatic = [1] if atom.GetIsAromatic() else [0]

        # Ring size
        atom_ring_size = min_ring_sizes[atom_idx]
        atom_ring_size_one_hot = get_one_hot(atom_ring_size, atom_ring_size_map, len(atom_ring_size_list))

        # atomic mass
        atomic_weight = atom.GetMass()
        scaled_weight = atomic_weight / 100.0

        # concatenation
        atom_features = np.concatenate([
            atom_type_one_hot,
            num_heavy_atom_one_hot,
            num_h_one_hot,
            formal_charge_one_hot,
            hybridization_one_hot,
            chirality_one_hot,
            is_aromatic,
            atom_ring_size_one_hot,
            [scaled_weight]
        ])

        node_features.append(atom_features)

    # Get bond features
    edges_index = []
    edge_features = []

    for bond in mols.GetBonds():
        i = bond.GetBeginAtomIdx()  # the initial atom
        j = bond.GetEndAtomIdx()  # the initial atom
        edges_index.append((i, j))
        edges_index.append((j, i))

        # bond type
        bond_type = str(bond.GetBondType())
        bond_type_one_hot = get_one_hot(bond_type, bond_type_map, len(bond_type_list))

        # conjugation
        is_conjugated = [1] if bond.GetIsConjugated() else [0]

        # stereochemistry
        stereo_str = str(bond.GetStereo())
        stereo_one_hot = get_one_hot(stereo_str, bond_stereo_map, len(bond_stereo_list))

        bond_idx = bond.GetIdx()
        bond_ring_size = min_ring_sizes_bond[bond_idx]

        bond_ring_size_one_hot = get_one_hot(bond_ring_size, bond_ring_size_map, len(bond_ring_size_list))

        # concatenation
        bond_features = np.concatenate([
            bond_type_one_hot,
            is_conjugated,
            stereo_one_hot,
            bond_ring_size_one_hot
        ])

        # bidirectional edges in the same features
        edge_features.append(bond_features)
        edge_features.append(bond_features)

    # to Tensor object
    node_feature = torch.tensor(np.array(node_features), dtype=torch.float)

    # 单原子分子
    if len(edges_index) == 0:
        edges_index = torch.empty((2, 0), dtype=torch.long)
        edge_feature = torch.empty((0, _edge_dim), dtype=torch.float)
    else:
        edges_index = torch.tensor(edges_index, dtype=torch.long).t().contiguous()
        edge_feature = torch.tensor(np.array(edge_features), dtype=torch.float)

    return Data(x=node_feature, edge_index=edges_index,edge_attr=edge_feature)


graphs = [smiles_to_graph(smi) for smi in valid_smiles]
node_dim, edge_dim = _node_dim, _edge_dim
valid_smiles = valid_smiles

print("\n检查图数据中的边特征异常...")
problem_idx = []

for i, (g, smi) in enumerate(zip(graphs, valid_smiles)):
    edge_attr = g.edge_attr
    edge_index = g.edge_index

    if edge_index.ndim != 2 or edge_index.shape[0] != 2:
        print(f"❌ 分子索引 {i}: SMILES = {smi}")
        print(f"   edge_index 形状异常: {edge_index.shape}，期望为 [2, num_edges]")
        problem_idx.append(i)

    if edge_attr.ndim != 2 or edge_attr.shape[1] != edge_dim:
        print(f"❌ 分子索引 {i}: SMILES = {smi}")
        print(f"   edge_attr 形状异常: {edge_attr.shape}，期望为 [num_edges, {edge_dim}]")
        problem_idx.append(i)

    if edge_attr.shape[0] != edge_index.shape[1]:
        print(f"❌ 分子索引 {i}: SMILES = {smi}")
        print(f"   边数量不一致: edge_attr 有 {edge_attr.shape[0]} 条，edge_index 有 {edge_index.shape[1]} 条")
        problem_idx.append(i)

    if edge_attr.shape[0] == 0:
        print(f"⚡ 分子索引 {i}: SMILES = {smi} 是无边分子")


# # test
# graph = smiles_to_graph('CC(=O)NCCCOc1cccc(CN2CCCCC2)c1')
#
# # 将特征转换为 DataFrame 并输出 Excel
# # # 定义原子特征列名
# atom_col_names = (
#     [f'elem_{e}' for e in element_list] +
#     [f'num_heavy_atom_{d}' for d in num_heavy_atom_list] +
#     [f'numH_{h}' for h in num_h_list] +
#     [f'formal_charge_{c}' for c in formal_charge_list] +
#     [f'hybridization_{h}' for h in hybridization_list] +
#     [f'chiral_{c}' for c in chirality_list] +
#     ['is_aromatic'] +
#     [f'atom_ring_size_{s}' for s in atom_ring_size_list] +
#     ['mass_div_100']
# )
#
# df_nodes = pd.DataFrame(graph.x.numpy(), columns=atom_col_names)
# df_nodes.insert(0, 'atom_idx', range(len(df_nodes)))
#
# # 定义边特征列名
# edge_col_names = (
#     [f'bond_{bt}' for bt in bond_type_list] +
#     ['is_conjugated'] +
#     [f'bond_stereo_{bs}' for bs in bond_stereo_list] +
#     [f'bond_ring_size_{s}' for s in bond_ring_size_list]
# )
#
# df_edges = pd.DataFrame(graph.edge_attr.numpy(), columns=edge_col_names)
# # 边索引信息
# edge_src = graph.edge_index[0].tolist()
# edge_dst = graph.edge_index[1].tolist()
# df_edges.insert(0, 'src_atom', edge_src)
# df_edges.insert(1, 'dst_atom', edge_dst)
#
# # 保存到 Excel 文件
# with pd.ExcelWriter('test_features.xlsx') as writer:
#     df_nodes.to_excel(writer, sheet_name='Atom features', index=False)
#     df_edges.to_excel(writer, sheet_name='Bond features', index=False)
#
# print("特征已保存至 test_features.xlsx")
