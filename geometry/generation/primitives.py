"""Unchanged geometry primitives extracted from the frozen pilot."""
import hashlib
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem
ELEMENTS = ["C", "F", "H", "N", "O"]
EMBED_SEED_BASE = 20260928
MAX_OPT_ITERS = 300

def canonical(mol):
    copy = Chem.Mol(mol)
    Chem.SanitizeMol(copy)
    return Chem.MolToSmiles(Chem.RemoveHs(copy), canonical=True, isomericSmiles=True)


def seed_for(dataset, molecule_id):
    text = f"geometry-pilot-v1|{EMBED_SEED_BASE}|{dataset}|{molecule_id}"
    return int.from_bytes(hashlib.sha256(text.encode()).digest()[:4], "big") % 2147483647


def embed_parameters(seed):
    params = AllChem.ETKDGv3()
    params.randomSeed = seed
    params.numThreads = 1
    params.enforceChirality = True
    params.useRandomCoords = False
    params.maxIterations = 1000
    params.trackFailures = True
    return params


def point_features(mol, max_atoms):
    """Match the frozen cache semantics: float32 xyz/rho/C,F,H,N,O, zero pad."""
    n = mol.GetNumAtoms()
    if n > max_atoms or n < 1:
        raise ValueError(f"Atom count {n} outside frozen padding capacity {max_atoms}")
    xyz = np.asarray(mol.GetConformer().GetPositions(), dtype=np.float32)
    if xyz.shape != (n, 3) or not np.isfinite(xyz).all():
        raise ValueError("Invalid coordinates")
    xyz -= xyz.mean(axis=0)
    symbols = [atom.GetSymbol() for atom in mol.GetAtoms()]
    onehot = np.eye(5, dtype=np.float32)[[ELEMENTS.index(symbol) for symbol in symbols]]
    features = np.concatenate((xyz, np.linalg.norm(xyz, axis=1, keepdims=True), onehot), axis=1)
    points = np.zeros((max_atoms, 9), dtype=np.float32)
    points[:n] = features
    assert points.dtype == np.float32 and np.isfinite(points).all()
    assert np.max(np.abs(points[:n, :3].mean(axis=0))) < 2e-6
    assert np.array_equal(points[:n, 3], np.sqrt(np.sum(points[:n, :3] ** 2, axis=1)))
    assert np.all(points[:n, 4:].sum(axis=1) == 1) and not points[n:].any()
    return points


def warmup():
    """Exclude RDKit initialization cost from measured molecule processing."""
    mol = Chem.AddHs(Chem.MolFromSmiles("CCO"))
    if AllChem.EmbedMolecule(mol, embed_parameters(12345)) == 0:
        AllChem.MMFFOptimizeMolecule(mol, mmffVariant="MMFF94", maxIters=MAX_OPT_ITERS)
