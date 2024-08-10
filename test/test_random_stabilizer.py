# Let's sanity check random stabilizer generation with qiskit

from clifford_utils.random_stabilizer import *

from qiskit import QuantumCircuit
from qiskit.quantum_info import StabilizerState

import time

n = 3
gate_list = stabilizer_gate_list_ag(n)
stab_counts = {}  # Will record counts of stabilizer tableaux

start = time.time()

iters = 100000
for i in range(iters):
    toggles = random_stabilizer_toggles_ag(n)
    qc = QuantumCircuit(n)
    for j in range(len(gate_list)):
        active = toggles[j]
        if active:
            gate_name, qubits = gate_list[j]
            if gate_name == "x":
                qc.x(qubits[0])
            elif gate_name == "h":
                qc.h(qubits[0])
            elif gate_name == "s":
                qc.s(qubits[0])
            elif gate_name == "cz":
                qc.cz(qubits[0], qubits[1])
            elif gate_name == "cx":
                qc.cx(qubits[0], qubits[1])
    stab = StabilizerState(qc)
    # Extract the unsigned stabilizer tableau
    tableau_unsigned = tuple(tuple(row) for row in stab.clifford.stab[:, : 2 * n])
    if tableau_unsigned in stab_counts:
        stab_counts[tableau_unsigned] += 1
    else:
        stab_counts[tableau_unsigned] = 1

print(time.time() - start)

# Should equal the number of n-qubit stabilizer states
print(len(stab_counts) * 2**n)
# Should be close to 1
print(sum(x**2 for x in stab_counts.values()) * len(stab_counts) / (iters**2))
