import time
from optimize.optimize_qujax import *

n = 12
depth = 84
optimizer = AnsatzOptimizer(n)
noisy = True

target_state = np.random.normal(size=([2] * n)) + 1j * np.random.normal(size=([2] * n))
target_state = target_state / np.linalg.norm(target_state)

start = time.time()
opt = optimizer.optimize(target_state, depth, noisy=noisy)
print(f"Optimization time: {time.time() - start}")
