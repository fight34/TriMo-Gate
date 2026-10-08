"""Feature checks from the original cache builder."""
from rdkit import Chem
ELEMENTS = ["C", "F", "H", "N", "O"]

def atom_values(atom, rings):
    idx = atom.GetIdx()
    return [atom.GetSymbol(), atom.GetDegree(), atom.GetTotalNumHs(),
            atom.GetFormalCharge(), str(atom.GetHybridization()),
            str(atom.GetChiralTag()),
            rings.MinAtomRingSize(idx) if rings.NumAtomRings(idx) else 0]


def bond_values(bond, rings):
    idx = bond.GetIdx()
    return [str(bond.GetBondType()), str(bond.GetStereo()),
            rings.MinBondRingSize(idx) if rings.NumBondRings(idx) else 0]


def canonical_mol(smiles, molecule_id):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f'ID {molecule_id}: invalid SMILES {smiles!r}')
    mol = Chem.MolFromSmiles(Chem.MolToSmiles(mol, isomericSmiles=True, canonical=True))
    if mol is None or not mol.GetNumAtoms():
        raise ValueError(f'ID {molecule_id}: canonical round-trip failed')
    if any(atom.GetSymbol() not in ELEMENTS for atom in mol.GetAtoms()):
        raise ValueError(f'ID {molecule_id}: unsupported element')
    return mol
