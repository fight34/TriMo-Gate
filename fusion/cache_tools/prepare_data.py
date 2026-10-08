"""Build an auditable, portable QM9 engineering-pilot cache; never edit source data.

The dataset-level schema reproduces the original loaders' feature semantics.
This is an engineering pilot, not the final publication data protocol.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import platform
import time
import xml.etree.ElementTree as ET
from collections import Counter, deque
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from zipfile import ZipFile

import numpy as np
from rdkit import Chem, rdBase
from SmilesPE.pretokenizer import atomwise_tokenizer

VERSION = 'qm9-pilot-v1'
ELEMENTS = ['C', 'F', 'H', 'N', 'O']


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b''):
            digest.update(chunk)
    return digest.hexdigest()


def read_xlsx(path):
    ns = '{http://schemas.openxmlformats.org/spreadsheetml/2006/main}'
    with ZipFile(path) as archive:
        shared = []
        if 'xl/sharedStrings.xml' in archive.namelist():
            with archive.open('xl/sharedStrings.xml') as handle:
                for _, element in ET.iterparse(handle, events=('end',)):
                    if element.tag == ns + 'si':
                        shared.append(''.join(t.text or '' for t in element.iter(ns + 't')))
                        element.clear()
        with archive.open('xl/worksheets/sheet1.xml') as handle:
            for _, element in ET.iterparse(handle, events=('end',)):
                if element.tag != ns + 'row':
                    continue
                values = {}
                for cell in element.findall(ns + 'c'):
                    letters = ''.join(c for c in cell.get('r') if c.isalpha())
                    col = 0
                    for letter in letters:
                        col = col * 26 + ord(letter) - 64
                    value = cell.find(ns + 'v')
                    value = value.text if value is not None else None
                    if cell.get('t') == 's':
                        value = shared[int(value)]
                    elif cell.get('t') == 'inlineStr':
                        value = ''.join(t.text or '' for t in cell.iter(ns + 't'))
                    values[col - 1] = value
                yield [values.get(i) for i in range(max(values) + 1)]
                element.clear()


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


def read_xyz(args):
    molecule_id, xyz_dir, row = args
    path = xyz_dir / f'dsgdb9nsd_{molecule_id:06d}.xyz'
    raw = path.read_bytes()
    lines = raw.decode('utf-8').splitlines()
    n_atoms = int(lines[0])
    props = lines[1].split()
    if props[0] != 'gdb' or int(props[1]) != molecule_id or len(props) != 17:
        raise ValueError(f'ID {molecule_id}: XYZ identity/property header mismatch')
    vals = np.asarray([float(value.replace('*^', 'e')) for value in props[2:]], dtype=np.float64)
    expected = np.asarray([float(value) for value in row[2:17]], dtype=np.float64)
    if not np.isfinite(vals).all() or not np.array_equal(vals, expected):
        raise ValueError(f'ID {molecule_id}: XYZ/table properties mismatch')
    if lines[n_atoms + 3].split() != row[:2]:
        raise ValueError(f'ID {molecule_id}: XYZ/table SMILES mismatch')
    xyz_lines = [line.split() for line in lines[2:n_atoms + 2]]
    elements = [parts[0] for parts in xyz_lines]
    coords = np.asarray([[float(value.replace('*^', 'e')) for value in parts[1:4]]
                         for parts in xyz_lines], dtype=np.float32)
    if coords.shape != (n_atoms, 3) or not np.isfinite(coords).all():
        raise ValueError(f'ID {molecule_id}: bad/nonfinite coordinates')
    if set(elements) - set(ELEMENTS):
        raise ValueError(f'ID {molecule_id}: unsupported XYZ element')
    coords -= coords.mean(axis=0)
    one_hot = np.eye(len(ELEMENTS), dtype=np.float32)[[ELEMENTS.index(e) for e in elements]]
    features = np.concatenate([coords, np.linalg.norm(coords, axis=1, keepdims=True), one_hot], axis=1)
    return molecule_id, features, Counter(elements), hashlib.sha256(raw).hexdigest()


def build(table, xyz_dir, output, workers=16):
    started = time.monotonic()
    output.mkdir(parents=True, exist_ok=True)
    rows = iter(read_xlsx(table))
    names = next(rows)
    rows = list(rows)
    if len(rows) != 133885 or any(len(row) != len(names) for row in rows):
        raise ValueError('Expected the audited full 133885-row QM9 table')
    if names[:2] != ['canonical_SMILES', 'isomeric_SMILES'] or 'dipole_moment' not in names:
        raise ValueError(f'Unexpected explicit column names: {names}')
    print(json.dumps({'phase': 'table_loaded', 'rows': len(rows), 'elapsed_s': time.monotonic()-started}), flush=True)
    ids = np.arange(1, len(rows)+1, dtype=np.int64)
    target_names = names[2:]
    targets = np.asarray([[float(value) for value in row[2:]] for row in rows], dtype=np.float64)
    if not np.isfinite(targets).all():
        raise ValueError('Nonfinite target values are not allowed')
    atom_names = ['element','degree','total_h','formal_charge','hybridization','chirality','atom_ring_size']
    bond_names = ['bond_type','stereo','bond_ring_size']
    atom_sets = [set() for _ in atom_names]
    bond_sets = [set() for _ in bond_names]
    tokens = []
    node_counts = []
    edge_counts = []
    for molecule_id, row in zip(ids, rows):
        smiles = row[0]
        if smiles != smiles.replace('\xa0','').replace(' ',''):
            raise ValueError(f'ID {molecule_id}: whitespace would alter SMILES')
        seq = atomwise_tokenizer(smiles)
        if ''.join(seq) != smiles:
            raise ValueError(f'ID {molecule_id}: tokenizer lost text')
        tokens.append(seq)
        mol = canonical_mol(smiles, molecule_id)
        rings = mol.GetRingInfo()
        for atom in mol.GetAtoms():
            for observed, value in zip(atom_sets, atom_values(atom, rings)):
                observed.add(value)
        for bond in mol.GetBonds():
            for observed, value in zip(bond_sets, bond_values(bond, rings)):
                observed.add(value)
        node_counts.append(mol.GetNumAtoms())
        edge_counts.append(mol.GetNumBonds()*2)
        if molecule_id % 25000 == 0:
            print(json.dumps({'phase':'schema','completed':int(molecule_id)}), flush=True)
    atom_sets[1] = set(range(max(atom_sets[1])+1))
    atom_sets[2] = set(range(max(atom_sets[2])+1))
    atom_sets[6].add(0)
    bond_sets[2].add(0)
    atom_schema = {name: sorted(values) for name, values in zip(atom_names, atom_sets)}
    bond_schema = {name: sorted(values) for name, values in zip(bond_names, bond_sets)}
    vocab = {'PAD':0}
    vocab.update({token:i+1 for i,token in enumerate(sorted({t for seq in tokens for t in seq}))})
    max_length = max(map(len, tokens))
    smiles_ids = np.zeros((len(rows), max_length), dtype=np.int16)
    for idx, seq in enumerate(tokens):
        smiles_ids[idx,:len(seq)] = [vocab[t] for t in seq]
    del tokens
    atom_maps = [{value: idx for idx,value in enumerate(atom_schema[name])} for name in atom_names]
    bond_maps = [{value: idx for idx,value in enumerate(bond_schema[name])} for name in bond_names]
    node_dim = sum(map(len,atom_maps))+2
    edge_dim = sum(map(len,bond_maps))+1
    node_offsets = np.r_[0,np.cumsum(node_counts)].astype(np.int64)
    edge_offsets = np.r_[0,np.cumsum(edge_counts)].astype(np.int64)
    graph_x = np.zeros((node_offsets[-1],node_dim),dtype=np.float32)
    edge_attr = np.zeros((edge_offsets[-1],edge_dim),dtype=np.float32)
    edge_index = np.zeros((2,edge_offsets[-1]),dtype=np.int16)
    # Preserve original feature ordering: aromaticity before atom ring and mass;
    # conjugation after bond type, before stereo and bond ring.
    for idx,(molecule_id,row) in enumerate(zip(ids,rows)):
        mol = canonical_mol(row[0],molecule_id)
        rings = mol.GetRingInfo()
        for atom in mol.GetAtoms():
            values = atom_values(atom,rings)
            vec = []
            for j,(value,mapping) in enumerate(zip(values,atom_maps)):
                if j == 6:
                    vec.append(float(atom.GetIsAromatic()))
                vec.extend(float(k == mapping[value]) for k in range(len(mapping)))
            vec.append(atom.GetMass()/100.)
            graph_x[node_offsets[idx]+atom.GetIdx()] = vec
        for bond in mol.GetBonds():
            values = bond_values(bond,rings)
            vec = []
            for j,(value,mapping) in enumerate(zip(values,bond_maps)):
                if j == 1:
                    vec.append(float(bond.GetIsConjugated()))
                vec.extend(float(k == mapping[value]) for k in range(len(mapping)))
            start = edge_offsets[idx] + 2*bond.GetIdx()
            a,b = bond.GetBeginAtomIdx(),bond.GetEndAtomIdx()
            edge_index[:,start:start+2] = [[a,b],[b,a]]
            edge_attr[start:start+2] = vec
        if molecule_id % 25000 == 0:
            print(json.dumps({'phase':'graphs','completed':int(molecule_id)}), flush=True)
    point_lengths = np.zeros(len(rows),dtype=np.int16)
    points = np.zeros((len(rows),29,9),dtype=np.float32)
    manifest_path = output / 'manifest.csv'
    with manifest_path.open('w',encoding='utf-8',newline='') as handle, ThreadPoolExecutor(max_workers=workers) as pool:
        writer = csv.writer(handle)
        writer.writerow(['molecule_id','xyz_filename','xyz_sha256','point_atoms',*names])
        pending = deque()
        def receive(item):
            molecule_id,features,counts,digest = item
            idx = molecule_id-1
            if len(features) > 29:
                raise ValueError(f'ID {molecule_id}: more than 29 atoms')
            for element in ELEMENTS:
                if counts[element] != int(float(rows[idx][names.index('n_'+element)])):
                    raise ValueError(f'ID {molecule_id}: elemental composition mismatch')
            point_lengths[idx] = len(features)
            points[idx,:len(features)] = features
            writer.writerow([molecule_id,f'dsgdb9nsd_{molecule_id:06d}.xyz',digest,len(features),*rows[idx]])
            if molecule_id % 25000 == 0:
                print(json.dumps({'phase':'xyz','completed':int(molecule_id)}),flush=True)
        for molecule_id,row in zip(ids,rows):
            pending.append(pool.submit(read_xyz,(int(molecule_id),xyz_dir,row)))
            if len(pending)>=256:
                receive(pending.popleft().result())
        while pending:
            receive(pending.popleft().result())
    cache_path = output / 'cache.npz'
    arrays = dict(ids=ids,smiles_ids=smiles_ids,graph_x=graph_x,edge_index=edge_index,
                  edge_attr=edge_attr,node_offsets=node_offsets,edge_offsets=edge_offsets,
                  points=points,point_lengths=point_lengths,targets=targets)
    if any(not np.isfinite(array).all() for array in arrays.values()):
        raise ValueError('Nonfinite cached data')
    print(json.dumps({'phase':'compressing','elapsed_s':time.monotonic()-started}),flush=True)
    np.savez_compressed(cache_path,**arrays)
    metadata = {
        'preprocessing_version':VERSION,'engineering_pilot_only':True,'molecule_count':len(ids),
        'source_table':str(table),'source_table_sha256':sha256(table),'xyz_source':str(xyz_dir),
        'manifest_sha256':sha256(manifest_path),'cache_sha256':sha256(cache_path),
        'prepare_script_sha256':sha256(__file__), 'python':platform.python_version(),
        'numpy':np.__version__,'rdkit':rdBase.rdkitVersion,
        'dims':dict(node_dim=node_dim,edge_dim=edge_dim,vocab_size=len(vocab),
                    smiles_length=max_length,point_dim=9,max_atoms=29),
        'atom_schema':atom_schema,'bond_schema':bond_schema,'token_to_idx':vocab,
        'target_names':target_names,'target_units':{'dipole_moment':'Debye'},
        'point_features':['centered_x_angstrom','centered_y_angstrom','centered_z_angstrom',
                          'rho_angstrom',*['element_'+e for e in ELEMENTS]],
        'schema_policy':'dataset-level categorical/token schema, as original; no target statistics',
        'selection_policy':'all 133885 locally audited rows; official geometry exclusion list not yet applied',
        'identity_policy':'1-based stable gdb ID matched to filename, XYZ internal ID, both SMILES and all 15 shared properties',
        'elapsed_s':time.monotonic()-started}
    (output/'metadata.json').write_text(json.dumps(metadata,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps({'phase':'complete','rows':len(ids),'dims':metadata['dims'],
                      'cache_bytes':cache_path.stat().st_size,'elapsed_s':metadata['elapsed_s']}),flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--table',type=Path,required=True)
    parser.add_argument('--xyz-dir',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--workers',type=int,default=16)
    args = parser.parse_args()
    build(args.table,args.xyz_dir,args.output,args.workers)
