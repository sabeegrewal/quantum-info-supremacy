from classical_bounds import *

import numpy as np
import matplotlib
import matplotlib.pyplot as plt

# Plot the noiseless lower bound as a function of n

def max_classical_hm_fidelity(m, n):
    return 2 * (m + 1) / np.sqrt(2**n - 1)

def min_communication_hm(fidelity, n):
    return fidelity * np.sqrt(2**n - 1) / 2 - 1

myrange = range(4, 17)

plt.rcParams['font.family'] = 'Arial'
plt.rcParams['font.size'] = 12

haar_comm = [min_communication(1.0, n, "haar") for n in myrange]
cliff_comm = [min_communication(1.0, n, "cliff") for n in myrange]
hm_comm = [min_communication_hm(1.0, n) for n in myrange]
plt.plot(myrange, haar_comm, color="#30A08E", linestyle=":",
         linewidth=1.5, marker="o", label="DXHOG, Haar")
plt.plot(myrange, cliff_comm, color="#30A08E", linestyle=":",
         linewidth=1.5, marker="s", label="DXHOG, Clifford")
plt.plot(myrange, hm_comm, color="#8064A2", linestyle=":",
         linewidth=1.5, marker="o", label="HM")
plt.xlabel("Number of noiseless qubits, $n$")
plt.ylabel("Lower bound on bits of classical communication to spoof, $m$")
plt.yscale("log", base=2)
plt.legend()
plt.grid(True, which="both", linestyle="--", linewidth=0.5, alpha=0.7)

plt.savefig("noiseless_bound.pdf")
plt.show()
