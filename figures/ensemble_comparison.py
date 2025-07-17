from classical_bounds import *

import numpy as np
import matplotlib
import matplotlib.pyplot as plt

# Plot the n=12 bound for different unitary ensembles

myrange = np.arange(0, 400, 1)


plt.rcParams['font.family'] = 'Arial'
plt.rcParams['font.size'] = 12

haar_xebs = [max_classical_xeb_jit(x, 12, "haar") for x in myrange]
design_xebs = [max_classical_xeb_jit(x, 12, "design") for x in myrange]
cliff_xebs = [max_classical_xeb_jit(x, 12, "cliff") for x in myrange]

#prod_xebs = [max_classical_xeb_jit(x, 12, "cliff_prod") for x in myrange[:11]]
plt.plot(myrange, haar_xebs, color="#E75D72", linewidth=1.5, label="Haar")
plt.plot(myrange, design_xebs, color="#8064A2", linewidth=1.5, label="$10$-design")
plt.plot(myrange, cliff_xebs, color="#30A08E", linewidth=1.5, label="Clifford")
#plt.plot(myrange[:11], prod_xebs, color="#FF9A56", linewidth=2, label="Product Clifford")
plt.xlabel("Bits of classical communication, $m$")
plt.ylabel("Maximum achievable XEB, $\epsilon$")
plt.xlim((0,400))
plt.ylim((0,1))
plt.legend()

# Ensure proper axis ordering
handles, labels = plt.gca().get_legend_handles_labels()
order = [2, 1, 0]
plt.legend([handles[i] for i in order], [labels[i] for i in order])
plt.grid(True, which="both", linestyle="--", linewidth=0.5, alpha=0.7)

plt.savefig("ensemble_comparison.pdf")
plt.show()
