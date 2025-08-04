from classical_bounds import *

import numpy as np
import matplotlib
import matplotlib.pyplot as plt

# Plot the n=12 bound for different unitary ensembles

myrange = list(range(401))


plt.rcParams['font.family'] = 'Arial'
plt.rcParams['font.size'] = 12

haar_xebs = [max_classical_xeb_jit(x, 12, "haar") for x in myrange]
design_xebs = [max_classical_xeb_jit(x, 12, "design") for x in myrange]
cliff_xebs = [max_classical_xeb_jit(x, 12, "cliff") for x in myrange]
lb_xebs = [achievable_classical_xeb(x, 12) for x in myrange]

#prod_xebs = [max_classical_xeb_jit(x, 12, "cliff_prod") for x in myrange[:11]]
plt.plot(haar_xebs, myrange, color="gray", linewidth=1.5, linestyle=(5, (10, 3)), label="Haar")
plt.plot(design_xebs, myrange, color="#E75D72", linewidth=1.5, linestyle="-.", label="$10$-design")
plt.plot(cliff_xebs, myrange, color="#8064A2", linewidth=1.5, linestyle="--", label="Clifford")
plt.plot(lb_xebs, myrange, color="#30A08E", linewidth=1.5, linestyle = "-", label="Classical protocol")
#plt.plot(myrange[:11], prod_xebs, color="#FF9A56", linewidth=2, label="Product Clifford")
plt.ylabel("Bits of classical communication, $m$")
plt.xlabel("Bound on $\mathcal{F}_{\mathsf{XEB}}$, $\epsilon$")
plt.ylim((0,400))
plt.xlim((0,1))
plt.legend()

# Ensure proper axis ordering
handles, labels = plt.gca().get_legend_handles_labels()
order = [3, 0, 1, 2]
plt.legend([handles[i] for i in order], [labels[i] for i in order])
plt.grid(True, which="both", linestyle="--", linewidth=0.5, alpha=0.7)

# 'tight' removes whitespace
plt.savefig("ensemble_comparison.pdf",bbox_inches='tight')
plt.show()
