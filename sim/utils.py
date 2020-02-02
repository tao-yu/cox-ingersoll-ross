import numpy as np
import os
import scipy.stats
from numba import jit, njit
from scipy.stats import gaussian_kde
import matplotlib.cm as cm
import matplotlib.pyplot as plt
from solvers import direct_simulation, implicit_scheme


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
    dW = np.zeros((N+1, M))
    dW[1:,:] = np.random.normal(0, np.sqrt(dt), size=(N, M))
    W = np.cumsum(dW, axis=0)        
    t = np.linspace(0, T, N+1)
    return t, W


def estimate_order(k, lamda, theta, X_0, scheme, T, n, M):
    t, W = brownian_paths(T, 2*n, M)
    t_n, X_n = scheme(k=k, lamda=lamda, theta=theta, X_0=X_0, t=t[::2], W=W[::2,:])
    t_2n, X_2n = scheme(k=k, lamda=lamda, theta=theta, X_0=X_0, t=t, W=W)
    S_n = np.mean(np.amax(np.abs(X_n - X_2n[::2,:]), axis=0))

    t_10, W_10 = brownian_paths(T, 20*n, M)
    t_10n, X_10n = scheme(k=k, lamda=lamda, theta=theta, X_0=X_0, t=t_10[::2], W=W_10[::2,:])
    t_20n, X_20n = scheme(k=k, lamda=lamda, theta=theta, X_0=X_0, t=t_10, W=W_10)
    S_10n = np.mean(np.amax(np.abs(X_10n - X_20n[::2,:]), axis=0))
    return np.log10(S_n) - np.log10(S_10n), np.log10(S_n), np.log10(S_10n)


def plot_distribution(k, lamda, theta, X_0, scheme, T, N_set, M, legend=False):
    colors = cm.rainbow(np.linspace(0, 1, len(N_set)))[::-1]
    for i, n in enumerate(N_set):
        t, W = brownian_paths(T, int(n), M)
        t, X_sim = scheme(k, lamda, theta, X_0, t, W)

        x = np.linspace(0, 5, 100)
        kde_sim = gaussian_kde(X_sim[-1,:])
        plt.plot(x, kde_sim(x), color=colors[i], label=str(n))

    X_T = np.zeros(M)
    for i in range(M):
        _, X_dist = direct_simulation(k, lamda, theta, X_0, T, 1)
        X_T[i] = X_dist[1]
        
    kde_dist = gaussian_kde(X_T)

    plt.plot(x, kde_dist(x), "--", label="True", color="black", linewidth=2)
    if legend:
        plt.legend()


def cir_bond_price(k, lamda, theta, X_t, T):
    t = 0
    h = np.sqrt(k**2 + 2*theta**2)
    A = ((2*h*np.exp((k+h)*(T-t)/2))/(2*h+(k+h)*(np.exp((T-t)*h)-1)))**(2*k*lamda/theta**2)
    B = (2*(np.exp((T-t)*h)-1))/(2*h+(k+h)*(np.exp((T-t)*h)-1))
    return A*np.exp(-B*X_t)


def binary_option(k, lamda, theta, X_0, T, N, M, payoff):
    t, W = brownian_paths(T, N, M)
    _, X = implicit_scheme(k, lamda, theta, X_0, t, W)
    X_int = np.sum(X, axis=0)*(T/N)
    val = np.exp(-X_int) * payoff(X[-1])
    mc_mean = np.mean(val)
    sd = np.std(val)
    return mc_mean, mc_mean - 1.96*sd/np.sqrt(M),  mc_mean + 1.96*sd/np.sqrt(M)