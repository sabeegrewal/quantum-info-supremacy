import numpy as np
from matplotlib import pyplot as plt
import matplotlib.ticker as ticker

# Contains a numpy array of all of the two-qubit gate angles
angles = np.load("../data/angles_H1-1_12_86_Tue_Aug_26_12-56-25_2025.npy", mmap_mode="r")

def normalize(theta):
    # In qujax, the maximally entangling ZZ gate has theta = 0.5;
    # See https://cqcl.github.io/qujax/gates.html

    # First normalize theta to [0, 1), which is equivalent to applying Z gates after if necessary
    theta = np.mod(theta, 1)
    # Then normalize to [0, 1/2), which is equivalent to conjugating one qubit by X if necessary
    theta = 1 / 2 - np.abs(theta - 1 / 2)

    return theta

angles_n = normalize(angles)

plt.rcParams['font.family'] = 'Arial'
plt.rcParams['font.size'] = 12

plt.grid(True, which="both", linestyle="--", linewidth=0.5, alpha=0.7, zorder=1)
plt.hist(angles_n, bins=40, color="#69D3BE", density=True, zorder=2)
plt.xlim((0,0.5))
plt.xlabel("Magnitude of angles $|\\theta_i|$, units of pi")
plt.ylabel("Probability density")
plt.gca().yaxis.set_major_locator(ticker.MaxNLocator(integer=True))

plt.savefig("angle_distribution.pdf", bbox_inches='tight')
plt.show()
