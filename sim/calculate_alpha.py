import numpy as np
from utils import *
from solvers import *
from tqdm import tqdm
import scipy.stats as stats
import time


def timestamp():
    return time.strftime("%m-%d_%H-%M")


def convergence(scheme, k, lamda, T, X_0, n, M, f_range, num_alphas):
    alphas = np.zeros((num_alphas, f_range.shape[0]))
    t1s = np.zeros((num_alphas, f_range.shape[0]))
    t2s = np.zeros((num_alphas, f_range.shape[0]))

    for j in tqdm(range(num_alphas)):
        for i, f in enumerate(tqdm(f_range)):
            theta = np.sqrt(2*f)
            alpha, t1, t2 = estimate_order(k, lamda, theta, X_0, scheme, T, n, M)
            alphas[j, i] = alpha
            t1s[j, i] = t1
            t2s[j, i] = t2
    return alphas, t1s, t2s


def record_convergence(scheme, k, lamda, T, X_0, n, M, f_range, num_alphas, name):
    alphas, t1s, t2s = convergence(
        k = k,
        lamda = lamda,
        T = T,
        X_0 = X_0,
        n = n,
        M = M,
        f_range = f_range,
        scheme=scheme,
        num_alphas=num_alphas
    )
    alphas = np.vstack([f_range, alphas])
    np.savetxt(f"alphas/{name}_{timestamp()}_{k:.1f}_{lamda:.1f}_{T}_{X_0}_{n}_{M}.csv", 
               alphas,
               delimiter=",")

    t1s = np.vstack([f_range, t1s])
    np.savetxt(f"alphas/t1s/{name}_{timestamp()}_{k:.1f}_{lamda:.1f}_{T}_{X_0}_{n}_{M}.csv", 
               t1s,
               delimiter=",")

    t2s = np.vstack([f_range, t2s])
    np.savetxt(f"alphas/t2s/{name}_{timestamp()}_{k:.1f}_{lamda:.1f}_{T}_{X_0}_{n}_{M}.csv", 
               t2s,
               delimiter=",")

start = time.time()
# Explicit method            
record_convergence(
    k = 1,
    lamda = 1,
    T = 1,
    X_0 = 1,
    n = 200,
    M = 3000,
    f_range = np.concatenate([np.array([0.001, 0.01, 0.1]), np.arange(0.2, 3.4, 0.2)]),
    scheme=explicit_scheme,
    num_alphas=30,
    name="Ezero"
)

# Implicit Method
record_convergence(
    k = 1,
    lamda = 1,
    T = 1,
    X_0 = 1,
    n = 200,
    M = 3000,
    f_range = np.concatenate([np.array([0.001, 0.01, 0.1]), np.arange(0.2, 3.4, 0.2)]),
    scheme=implicit_scheme,
    num_alphas=30,
    name="driftimpsqrt"
)

# Diop
record_convergence(
    k = 1,
    lamda = 1,
    T = 1,
    X_0 = 1,
    n = 200,
    M = 3000,
    f_range = np.concatenate([np.array([0.001, 0.01, 0.1]), np.arange(0.2, 3.4, 0.2)]),
    scheme=diop,
    num_alphas=30,
    name="diop"
)

# D-D
record_convergence(
    k = 1,
    lamda = 1,
    T = 1,
    X_0 = 1,
    n = 200,
    M = 3000,
    f_range = np.concatenate([np.array([0.001, 0.01, 0.1]), np.arange(0.2, 3.4, 0.2)]),
    scheme=deelstra_delbaen,
    num_alphas=30,
    name="deelstradelbaen"
)

end = time.time() - start
print(end)