"""Portable QM9 cache loader. Importing this module never accesses raw data."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset
from torch_geometric.data import Batch, Data


class QM9Dataset(Dataset):
    def __init__(self, data_dir, target='dipole_moment', indices=None, verify_hash=True):
        directory = Path(data_dir)
        self.metadata = json.loads((directory/'metadata.json').read_text(encoding='utf-8'))
        self.dims = self.metadata['dims']
        self.target = target
        names = self.metadata['target_names']
        if target not in names:
            raise ValueError(f'Unknown explicit target {target!r}; choices: {names}')
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
        if len(np.unique(ids)) != len(ids):
            raise ValueError('Duplicate molecule IDs in cache')
        selected = np.arange(len(ids)) if indices is None else np.asarray(indices,dtype=np.int64)
        if selected.ndim != 1 or (selected<0).any() or (selected>=len(ids)).any():
            raise ValueError('indices must be valid zero-based cache row indices')
        if len(np.unique(selected)) != len(selected):
            raise ValueError('Repeated indices are not allowed')
        self.indices = selected
        self.ids = ids[selected]
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


def collate_qm9(samples):
    if not samples:
        raise ValueError('Cannot collate an empty batch')
    return {'id':torch.tensor([sample['id'] for sample in samples],dtype=torch.long),
            'smiles_ids':torch.stack([sample['smiles_ids'] for sample in samples]),
            'graph':Batch.from_data_list([sample['graph'] for sample in samples]),
            'points':torch.stack([sample['points'] for sample in samples]),
            'point_lengths':torch.tensor([sample['point_length'] for sample in samples],dtype=torch.long),
            'y':torch.stack([sample['y'] for sample in samples])}
