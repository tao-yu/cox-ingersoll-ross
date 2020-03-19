from heston import heston_final_time, single_heston_log, fs_heston_final_time, \
    heston_log, single_heston_milstein
from solvers import implicit_scheme, full_truncation
import numpy as np
import pandas as pd
from utils import show_probplot, perform_kstest
from tqdm import tqdm
from numba import njit


def timestamp():
    return time.strftime("%m-%d_%H-%M")


K = 100
k = 6.21
lamda = 0.019
theta = 0.61
#theta = 0.01
X_0 = 0.010201
T = 1
rho = 1/2**6

r = 1
S_0 = 100
rf = 0.0319
cor = -0.70
true_price = 6.8061
num_paths = 10000

h_max_arr = T/np.array([100, 400, 1600])


@njit
def value_call(S_T,strike, rf, T):
    payoff = np.exp(-rf*T) * np.maximum(S_T-strike, 0)
    m = np.mean(payoff)
    sd = np.std(payoff)
    #m - se, m + se
    return m, sd


schemes = {"Implicit Scheme":implicit_scheme, 
           "Full Truncation":full_truncation}


results_df = pd.DataFrame()
sd_df = pd.DataFrame()

for h_max in tqdm(h_max_arr):
    h_min = h_max*rho
    S_heston, X_heston, h_mean = heston_final_time(k, lamda, theta, X_0, T, h_max, h_min, r, S_0, rf, cor, num_paths)
    h_mean_str = str(int(1/h_mean)+1)

    m, sd = value_call(S_heston[0], K, rf, T) 
    results_df.at["Adaptive (log)", h_mean_str] = m - true_price
    sd_df.at["Adaptive (log)", h_mean_str] = sd

    m, sd = value_call(S_heston[1], K, rf, T) 
    results_df.at["Adaptive (Milstein)", h_mean_str] = m - true_price
    sd_df.at["Adaptive (Milstein)", h_mean_str] = sd

    for name, scheme in tqdm(schemes.items()):
        S_heston = fs_heston_final_time(scheme, heston_log, k, lamda, theta, X_0, T, S_0, rf, cor, int(1/h_mean)+1, 1000, num_paths//1000)
        m, sd = value_call(S_heston, K, rf, T) 
        results_df.at[name, h_mean_str] = m - true_price
        sd_df.at[name, h_mean_str] = sd

print(results_df)
results_df.to_csv(f"heston_bias {num_paths}_{timestamp()}.csv")
sd_df.to_csv(f"heston_sd {num_paths}_{timestamp()}.csv")