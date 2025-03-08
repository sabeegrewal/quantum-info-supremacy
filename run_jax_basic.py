import time
from optimize.optimize_jax import *

n = 12
depth = 86
noisy = True

target_state = np.random.normal(size=([2] * n)) + 1j * np.random.normal(size=([2] * n))
target_state = target_state / np.linalg.norm(target_state)

start = time.time()
opt = optimize(target_state, depth, noisy=noisy)
print(f"Optimization time: {time.time() - start}")
