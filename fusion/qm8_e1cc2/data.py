"""Portable QM8 cache loader. Importing this module never accesses raw data."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset
from torch_geometric.data import Batch, Data


DATASET = 'QM8'
TARGET = 'E1-CC2'
UNIT = 'hartree'
EXPECTED_SAMPLE_COUNT = 21747


class QM8Dataset(Dataset):
    def __init__(self, data_dir, target=TARGET, indices=None, verify_hash=True):
        directory = Path(data_dir)
        self.metadata = json.loads((directory/'metadata.json').read_text(encoding='utf-8'))
        if self.metadata.get('dataset') != DATASET:
            raise ValueError(f'Expected explicit dataset={DATASET!r} in cache metadata')
        sample_count = self.metadata.get('sample_count')
        if isinstance(sample_count, bool) or not isinstance(sample_count, int) or sample_count < 1:
            raise ValueError('Cache metadata sample_count must be a positive integer')
        self.dims = self.metadata['dims']
        self.target = target
        names = self.metadata['target_names']
        if not isinstance(names, list) or len(set(names)) != len(names) or target not in names:
            raise ValueError(f'Unknown explicit target {target!r}; choices: {names}')
        if target != TARGET or self.metadata.get('target_units', {}).get(target) != UNIT:
            raise ValueError(f'This experiment requires {TARGET} in {UNIT}')
        cache_path = directory/'cache.npz'
        if verify_hash:
            digest = hashlib.sha256()
            with cache_path.open('rb') as handle:
                for chunk in iter(lambda: handle.read(1<<20),b''):
                    digest.update(chunk)
            if digest.hexdigest() != self.metadata['cache_sha256']:
                raise ValueError('Cache SHA256 mismatch')
        with np.load(cache_path,allow_pickle=False) as archive:
            self.arrays = {key:archive[key] for key in archive.files}
        ids = self.arrays['ids']
        if ids.ndim != 1 or len(ids) != sample_count:
            raise ValueError('Cache molecule IDs disagree with metadata sample_count')
        if self.arrays['targets'].shape != (sample_count, len(names)):
            raise ValueError('Cache target dimensions disagree with metadata')
        if len(np.unique(ids)) != len(ids):
            raise ValueError('Duplicate molecule IDs in cache')
        group_ids = self.arrays.get('group_ids')
        if group_ids is None or group_ids.shape != ids.shape or group_ids.dtype.kind not in 'iu':
            raise ValueError('Cache requires one integer structural group_id per molecule')
        unique_groups, inverse = np.unique(group_ids, return_inverse=True)
        minima = np.full(len(unique_groups), np.iinfo(np.int64).max, dtype=np.int64)
        np.minimum.at(minima, inverse, ids)
        if not np.array_equal(unique_groups, minima):
            raise ValueError('Each structural group_id must equal its minimum stable molecule ID')
        selected = np.arange(len(ids)) if indices is None else np.asarray(indices,dtype=np.int64)
        if selected.ndim != 1 or (selected<0).any() or (selected>=len(ids)).any():
            raise ValueError('indices must be valid zero-based cache row indices')
        if len(np.unique(selected)) != len(selected):
            raise ValueError('Repeated indices are not allowed')
        self.indices = selected
        self.ids = ids[selected]
        self.group_ids = group_ids[selected].astype(np.int64)
        self.targets = self.arrays['targets'][selected,names.index(target)].astype(np.float64)
        if not np.isfinite(self.targets).all():
            raise ValueError('Nonfinite targets')
        self._all_targets = self.arrays['targets'][:,names.index(target)].astype(np.float64)
        # Materialize shared tensor views once; indexing samples does not parse molecules.
        self._x = torch.from_numpy(self.arrays['graph_x'])
        self._edge_attr = torch.from_numpy(self.arrays['edge_attr'])
        self._edge_index = torch.from_numpy(self.arrays['edge_index'].astype(np.int64))
        self._smiles = torch.from_numpy(self.arrays['smiles_ids'].astype(np.int64))
        self._points = torch.from_numpy(self.arrays['points'])

    def __len__(self):
        return len(self.indices)

    def __getitem__(self,index):
        row = int(self.indices[index])
        n0,n1 = self.arrays['node_offsets'][row:row+2]
        e0,e1 = self.arrays['edge_offsets'][row:row+2]
        graph = Data(x=self._x[n0:n1],edge_index=self._edge_index[:,e0:e1],
                     edge_attr=self._edge_attr[e0:e1])
        return {'id':int(self.arrays['ids'][row]),'smiles_ids':self._smiles[row],
                'graph':graph,'points':self._points[row],
                'point_length':int(self.arrays['point_lengths'][row]),
                'y':torch.tensor([self._all_targets[row]],dtype=torch.float64)}


def collate_qm8(samples):
    if not samples:
        raise ValueError('Cannot collate an empty batch')
    return {'id':torch.tensor([sample['id'] for sample in samples],dtype=torch.long),
            'smiles_ids':torch.stack([sample['smiles_ids'] for sample in samples]),
            'graph':Batch.from_data_list([sample['graph'] for sample in samples]),
            'points':torch.stack([sample['points'] for sample in samples]),
            'point_lengths':torch.tensor([sample['point_length'] for sample in samples],dtype=torch.long),
            'y':torch.stack([sample['y'] for sample in samples])}
