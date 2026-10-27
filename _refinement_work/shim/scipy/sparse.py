import numpy as np


class coo_matrix:
    """Accumulates duplicate entries like scipy; tocsr() returns a dense array."""

    def __init__(self, arg, shape):
        data, (rows, cols) = arg
        self._dense = np.zeros(shape, dtype=np.float64)
        np.add.at(self._dense, (np.asarray(rows), np.asarray(cols)), np.asarray(data, dtype=np.float64))
        self.shape = shape

    def tocsr(self):
        return self._dense
