from classical_bounds import *

import numpy as np
import matplotlib
import matplotlib.pyplot as plt

# Plot the n=12 bound along with the XEB observed in experiment

# First compute mu and mu - 5*sigma

filename = "../data/shots_H1-1_12_86_Thu_Jun_12_17-06-28_2025.csv"
xeb_list = []
n_shots = 10000
with open(filename, "r") as file:
    # First line is a header
    file.readline()
    for i in range(n_shots):
        xeb = float(file.readline().strip().split(",")[-1])
        xeb_list.append(xeb)

# mu at ~77.36 bits
# mu - 5 sigma at ~61.42 bits
mu = np.average(xeb_list)
print(f"mu: {mu}")
sigma = np.std(xeb_list) / np.sqrt(n_shots)
print(f"sigma: {sigma}")

mu_comm = min_communication(mu, 12, "cliff")
mu_5sigma_comm = min_communication(mu - 5*sigma, 12, "cliff")

myrange = np.arange(0, 250, 1)

plt.rcParams['font.family'] = 'Arial'
plt.rcParams['font.size'] = 12

xebs = [max_classical_xeb_jit(x, 12, "cliff") for x in myrange]
plt.plot(xebs, myrange, color="#30A08E", linewidth=2, label="Classical bound")
plt.ylabel("Bits of classical communication, $m$")
plt.xlabel("Linear cross-entropy benchmarking fidelity, $\mathcal{F}_{\mathsf{XEB}}$")
plt.ylim((0,250))
plt.xlim((0,1))

# 0.427040981894732
plt.vlines(mu, 0, 250, colors='#8064A2', linestyles='-.', linewidth=1.5, label=r"$\widehat{\mathcal{F}}_{\mathsf{XEB}}$")
# 0.361714588004784
plt.vlines(mu - 5 * sigma, 0, 250, colors='#E75D72', linestyles='--', linewidth=1.5, label=r"$\widehat{\mathcal{F}}_{\mathsf{XEB}} - 5\sigma$")

plt.annotate(f"$m$ = {mu_comm:.1f}", xy=(mu, mu_comm), xytext=(mu+0.15, mu_comm),
            arrowprops=dict(arrowstyle='->'), ha='center', va='center')
plt.annotate(f"$m$ = {mu_5sigma_comm:.1f}", xy=(mu - 5 * sigma, mu_5sigma_comm), xytext=(mu - 5*sigma - 0.15, mu_5sigma_comm),
            arrowprops=dict(arrowstyle='->'), ha='center', va='center')

plt.legend()

# 'tight' removes whitespace
plt.savefig("achieved_xeb.pdf", bbox_inches='tight')
plt.show()
