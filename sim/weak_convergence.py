import numpy as np
from solvers import diop
from utils import brownian_paths
import time
from tqdm import tqdm
from pathlib import Path


def timestamp():
    return time.strftime("%m-%d_%H-%M")
    

k = 2
lamda = 2
theta = 1
X_0 = lamda
T = 1
M = 1000000

num_batches = 100
batch_size = M//num_batches

max_N_power = 15
min_N_power = 5
num_powers = max_N_power - min_N_power + 1

N_vals = 2**np.arange(max_N_power, min_N_power-1, -1)

start = time.time()
for batch in tqdm(range(num_batches)):
    t, W = brownian_paths(T, 2**max_N_power, batch_size)
    batch_output = np.zeros((num_powers, batch_size))

    for i in tqdm(range(num_powers)):
        _, X = diop(k, lamda, theta, X_0, t[::2**i], W[::2**i])
        batch_output[i] = X[-1]

    folder_name = f"weak/{k}_{lamda}_{theta}_{X_0}_{T}_{M}_{num_batches}" 
    Path(folder_name).mkdir(parents=True, exist_ok=True)
    np.save(f"{folder_name}/{max_N_power}_{min_N_power}_{batch}_{timestamp()}.npy", batch_output)

np.save(f"{folder_name}/N_vals.npy", N_vals)

end = time.time() - start 
print(f"Finished in {end} seconds")
