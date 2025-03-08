from optimize.optimize_jax import *
import qiskit
from pytket.extensions.quantinuum import QuantinuumBackend, QuantinuumAPIOffline

n = 12

import matplotlib.pyplot as plt
import time

print(time.time())
for depth in range(80, 81, 1):
    all_two_qubit_params = []
    for i in range(10):
        init_params = random_init_params(n, depth)

        target_state = np.random.normal(size=([2] * n)) + 1j * np.random.normal(
            size=([2] * n)
        )
        target_state = target_state / np.linalg.norm(target_state)

        opt = optimize(target_state, depth, noisy=True)
        print((depth, opt.fun))
        all_two_qubit_params.append(zzphase_params(n, opt.x))
    par = jnp.array(all_two_qubit_params).flatten()
    par = jnp.mod(par, 1)
    par = 1 / 2 - jnp.abs(par - 1 / 2)
    plt.hist(par, bins=50)
    plt.show()

    file = open(f"n_{n}_depth_{depth}_two_qubit_params.txt", "w")
    for p in par:
        file.write(str(p) + "\n")
    file.close()
print(time.time())

x = opt.x
qc = make_circuit(n, x, "qiskit")
import qiskit.quantum_info as qi

qstate = qi.Statevector.from_instruction(qc).data.reshape([2] * n)
ostate = output_state(n, x)
# Need to transpose because qiskit's convention is to order qubit indices backwards
print(jnp.vdot(qstate.T, ostate))

pqc = make_circuit(n, x, "pytket")

from pytket.extensions.qiskit import (
    AerStateBackend,
    AerBackend,
    AerUnitaryBackend,
    IBMQBackend,
    IBMQEmulatorBackend,
    qiskit_to_tk,
)

aer_state_b = AerStateBackend()
state_handle = aer_state_b.process_circuit(pqc)
pstate = aer_state_b.get_result(state_handle).get_state().reshape([2] * n)
print(jnp.vdot(pstate, ostate))


api_offline = QuantinuumAPIOffline()
backend = QuantinuumBackend(device_name="H1-1LE", api_handler=api_offline)
backend.default_compilation_pass().apply(pqc)
# print(pqc)
state_handle = aer_state_b.process_circuit(pqc)
pstate = aer_state_b.get_result(state_handle).get_state().reshape([2] * n)
print(jnp.vdot(pstate, ostate))

clifford = qiskit.quantum_info.random_clifford(n)
clifford_circ = qiskit.synthesis.synth_clifford_full(clifford)
clifford_circ_tk = qiskit_to_tk(clifford_circ)
backend.default_compilation_pass().apply(clifford_circ_tk)

pqc.append(clifford_circ_tk)

# Serialize to JSON
print(pqc.to_dict())
# Can be read using Circuit.from_dict()
