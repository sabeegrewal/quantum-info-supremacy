from classical_bounds import *

import numpy as np
import matplotlib
import matplotlib.pyplot as plt

import math

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

min_comm_mean = min_communication(mu, 12, "cliff")
min_comm_5sigma = min_communication(mu - 5*sigma, 12, "cliff")
max_comm_mean = max_communication(mu, 12)
max_comm_5sigma = max_communication(mu + 5*sigma, 12)

myrange = list(range(401))

plt.rcParams['font.family'] = 'Arial'
plt.rcParams['font.size'] = 12

xebs = [max_classical_xeb_jit(x, 12, "cliff") for x in myrange]
lb_xebs = [achievable_classical_xeb(x, 12) for x in myrange]

plt.ylabel("Bits of classical communication, $m$")
plt.xlabel("Linear cross-entropy benchmarking fidelity, $\mathcal{F}_{\mathsf{XEB}}$")
plt.ylim((0,400))
plt.xlim((0,1))

# 0.427040981894732
plt.vlines(mu, 0, 400, colors='gray', linestyles='-.', linewidth=1.5,
           label=r"$\widehat{\mathcal{F}}_{\mathsf{XEB}}$")
# 0.361714588004784 to 0.492367375784679
plt.fill_betweenx(np.arange(0,400,1.0), mu-5*sigma, mu+5*sigma, color='gray',
                  alpha = 0.2, linewidth=1.5, label=r"$\widehat{\mathcal{F}}_{\mathsf{XEB}} \pm 5\sigma$")

plt.fill_between(lb_xebs, myrange, 400, alpha=0.2, linewidth=1.5, color='#30A08E', label="Feasible")
plt.fill_between(xebs, myrange, 0, alpha=0.2, linewidth=1.5, color='#8064A2', label="Infeasible")

plt.plot(mu, min_comm_mean, marker="o", color="black", markersize=4)
plt.plot(mu-5*sigma, min_comm_5sigma, marker="o", color="black", markersize=4)
plt.plot(mu, max_comm_mean, marker="o", color="black", markersize=4)
plt.plot(mu + 5 * sigma, max_comm_5sigma, marker="o", color="black", markersize=4)

plt.annotate(f"$m$ = {math.ceil(min_comm_mean)}", xy=(mu, min_comm_mean),
             xytext=(mu-0.075, min_comm_mean+15), ha='center', va='center')
plt.annotate(f"$m$ = {math.ceil(min_comm_5sigma)}", xy=(mu-5*sigma, min_comm_5sigma),
             xytext=(mu-5*sigma-0.09, min_comm_5sigma), ha='center', va='center')
plt.annotate(f"$m$ = {math.ceil(max_comm_mean)}", xy=(mu, max_comm_mean),
             xytext=(mu+0.09, max_comm_mean), ha='center', va='center')
plt.annotate(f"$m$ = {math.ceil(max_comm_5sigma)}", xy=(mu+5*sigma, max_comm_5sigma),
             xytext=(mu+5*sigma+0.09, max_comm_5sigma), ha='center', va='center')

plt.legend()

# 'tight' removes whitespace
plt.savefig("achieved_xeb.pdf", bbox_inches='tight')
plt.show()
