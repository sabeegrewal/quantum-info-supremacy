from classical_bounds import *

import numpy as np
import matplotlib
import matplotlib.pyplot as plt

# Plot the n=12 bound along with the XEB observed in experiment

myrange = np.arange(0, 250, 1)

# mu at ~77.29 bits
# mu - 5 sigma at ~61.37 bits
mu = 0.4267625296473500
sigma = 1.3058928853745300 / np.sqrt(10000)

plt.rcParams['font.family'] = 'Arial'
plt.rcParams['font.size'] = 12

xebs = [max_classical_xeb_jit(x, 12, "cliff") for x in myrange]
plt.plot(xebs, myrange, color="#30A08E", linewidth=2, label="Classical bound")
plt.ylabel("Bits of classical communication, $m$")
plt.xlabel("Maximum achievable $\mathcal{F}_{\mathrm{XEB}}$")
plt.ylim((0,250))
plt.xlim((0,1))

# 0.426762529
plt.vlines(mu, 0, 250, colors='#8064A2', linestyles='-.', linewidth=1.5, label=r"Observed $\mu$")
# 0.361467885
plt.vlines(mu - 5 * sigma, 0, 250, colors='#E75D72', linestyles='--', linewidth=1.5, label=r"Observed $\mu - 5\sigma$")

plt.annotate("$m$ = 77.3", xy=(mu, 77.29), xytext=(mu+0.15, 77.29),
            arrowprops=dict(arrowstyle='->'), ha='center')
plt.annotate("$m$ = 61.4", xy=(mu - 5 * sigma, 61.37), xytext=(mu - 5*sigma - 0.15, 61.37),
            arrowprops=dict(arrowstyle='->'), ha='center')

plt.legend()

# 'tight' removes whitespace
plt.savefig("achieved_xeb.pdf", bbox_inches='tight')
plt.show()
