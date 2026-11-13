"""Lazy, read-only HDF5 access for the published frozen input tensors."""
from pathlib import Path
import h5py
import numpy as np


class H5Input:
    def __init__(self, path, variant='standardized'):
        if variant != 'standardized':
            raise ValueError('The compact Hub release supports standardized x only. sx is not an inverse-normalized noisy input.')
        self.path = Path(path)
        self._file = h5py.File(self.path, 'r')
        self._data = self._file['x']
        self.shape, self.dtype, self.ndim = self._data.shape, self._data.dtype, self._data.ndim

    def __len__(self):
        return self.shape[0]

    def __getitem__(self, index):
        if isinstance(index, (int, np.integer, slice)):
            return self._data[index]
        indices = np.asarray(index, dtype=np.int64)
        # h5py requires increasing, unique fancy indices; restore shuffled batch order.
        unique, inverse = np.unique(indices, return_inverse=True)
        return self._data[unique.tolist()][inverse]

    def close(self):
        self._file.close()


def open_input(path, variant='standardized'):
    path = Path(path)
    return H5Input(path, variant) if path.suffix == '.h5' else np.load(path, mmap_mode='r', allow_pickle=False)


def close_input(value):
    if isinstance(value, H5Input):
        value.close()
    else:
        mmap = getattr(value, '_mmap', None)
        if mmap is not None:
            mmap.close()
