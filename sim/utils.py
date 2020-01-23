import numpy as np
import os
import scipy.stats
from numba import jit


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


def mc_mean(a, ci_width=0.95, axis=None):
    if axis is None:
        a = a.flatten()
    mc_mean = np.mean(a, axis=axis)
    quantile = scipy.stats.norm.ppf(1-(1-ci_width)/2)
    upper = mc_mean + quantile*np.std(a, axis=axis)/np.sqrt(a.shape[axis])
    lower = mc_mean - quantile*np.std(a, axis=axis)/np.sqrt(a.shape[axis])
    return mc_mean, lower, upper


def brownian_paths(T, N, M):
    dt = T/N
    dW = np.random.normal(0, np.sqrt(dt), size=(M, N))
    dW = np.hstack([np.zeros((M, 1)), dW])
    W = np.cumsum(dW, axis=1)        
    t = np.arange(0, T+dt, dt)
    return t, W


@jit(nopython=True)
def brownian_paths_jit(T, N, M):
    dt = T/N
    dW = np.zeros((M, N+1))
    dW[:,1:] = np.random.normal(0, np.sqrt(dt), size=(M, N))
    W = np.zeros(dW.shape)
    for i in range(0, W.shape[0]):
        W[i,:] = np.cumsum(dW[i,:])
    t = np.arange(0, T+dt, dt)
    return t, W


def estimate_order(scheme, T, n, M):
    t, W = brownian_paths(T, 2*n, M)
    t_n, X_n = scheme(t[::2], W[:,::2])
    t_2n, X_2n = scheme(t, W)
    S_n = np.array(np.mean(np.amax(np.abs(X_n - X_2n[:,::2]), axis=1), axis=0))

    t_10, W_10 = brownian_paths(T, 20*n, M)
    t_10n, X_10n = scheme(t_10[::2], W_10[:,::2])
    t_20n, X_20n = scheme(t_10, W_10)
    S_10n = np.array(np.mean(np.amax(np.abs(X_10n - X_20n[:,::2]), axis=1), axis=0))
    return np.log10(S_n) - np.log10(S_10n)