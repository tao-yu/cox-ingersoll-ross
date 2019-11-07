import numpy as np
import os

class MatlabRandn:

    def __init__(self):
        script_dir = os.path.dirname(os.path.realpath(__file__))
        self._randn = np.load(os.path.join(script_dir, "../randn/randn.npy"))
        self._index = 0

    def __call__(self, sz1, sz2=None):
        if sz2 is None:
            start = self._index
            end = self._index + sz1
            self._index = end
            return self._randn[start:end]
        else:
            start = self._index
            end = self._index + (sz1 * sz2)
            self._index = end
            return self._randn[start:end].reshape(sz2, sz1).T

    def reset(self):
        self._index = 0