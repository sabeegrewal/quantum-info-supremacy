from optimize.optimize_qujax import *
from clifford_utils.random_stabilizer import *
from pytket import Circuit
from pytket.extensions.quantinuum import QuantinuumBackend, QuantinuumAPIOffline
from pytket.extensions.quantinuum.backends.api_wrappers import QuantinuumAPI
from pytket.extensions.quantinuum.backends.credential_storage import (
    QuantinuumConfigCredentialStorage,
)

from qujax.statetensor import apply_gate
from qujax.gates import *

import time


n = 4
depth = 4
start = time.time()
online = True
noisy = True

optimizer = AnsatzOptimizer(n)
target_state = np.random.normal(size=([2] * n)) + 1j * np.random.normal(size=([2] * n))
target_state = target_state / np.linalg.norm(target_state)

opt = optimizer.optimize(target_state, depth, noisy=noisy)
output_state = optimizer.output_state(opt.x)
noiseless_fidelity = -optimizer.loss(opt.x, target_state)
fidelity_from_noise = optimizer.fidelity_from_noise(opt.x)
print(f"Noiseless fidelity: {noiseless_fidelity}")
print(f"Estimated fidelity due to noise: {fidelity_from_noise}")
print(f"Estimated overall fidelity: {fidelity_from_noise * noiseless_fidelity}")
print(f"Optimization time: {time.time() - start}")
print("")

opt_params = opt.x
state_prep_circ = optimizer.pytket_circuit(opt_params)
# Add a barrier between state preparation and Clifford measurement
state_prep_circ.add_barrier(qubits=list(range(n)))

if online:
    backend = QuantinuumBackend(
        device_name="H1-1E",
        api_handler=QuantinuumAPI(token_store=QuantinuumConfigCredentialStorage()),
    )
else:
    api_offline = QuantinuumAPIOffline()
    backend = QuantinuumBackend(device_name="H1-1LE", api_handler=api_offline)
backend.default_compilation_pass().apply(state_prep_circ)

stab_gates = stabilizer_gate_list_ag(n)
stab_gates.reverse()

n_cliffords = 1
shots_per_clifford = 10000
observations = []
for i in range(n_cliffords):
    start = time.time()
    toggles = random_stabilizer_toggles_ag(n)
    toggles.reverse()

    cliff_circ = Circuit(n)
    scoring_state = target_state
    cliff_output_state = output_state
    for i in range(len(stab_gates)):
        if toggles[i]:
            gate_name, qubits = stab_gates[i]
            if gate_name == "x":
                cliff_circ.X(*qubits)
                scoring_state = apply_gate(scoring_state, X, qubits)
                cliff_output_state = apply_gate(cliff_output_state, X, qubits)
            elif gate_name == "h":
                cliff_circ.H(*qubits)
                scoring_state = apply_gate(scoring_state, H, qubits)
                cliff_output_state = apply_gate(cliff_output_state, H, qubits)
            elif gate_name == "s":
                cliff_circ.Sdg(*qubits)
                scoring_state = apply_gate(scoring_state, Sdg, qubits)
                cliff_output_state = apply_gate(cliff_output_state, Sdg, qubits)
            elif gate_name == "cz":
                cliff_circ.CZ(*qubits)
                scoring_state = apply_gate(scoring_state, CZ, qubits)
                cliff_output_state = apply_gate(cliff_output_state, CZ, qubits)
            else:
                raise Exception("invalid gate name: " + gate_name)
    backend.default_compilation_pass().apply(cliff_circ)

    overall_circ = state_prep_circ.copy()
    overall_circ.append(cliff_circ)
    overall_circ.measure_all()

    result = backend.run_circuit(overall_circ, n_shots=shots_per_clifford)
    xeb_scores = list(
        abs(scoring_state[tuple(shot)]) ** 2 * 2**n - 1 for shot in result.get_shots()
    )
    observed_xeb = sum(xeb_scores) / len(xeb_scores)
    basis_xeb = sum(abs((scoring_state * cliff_output_state).flatten() ** 2)) * 2**n - 1
    print(f"Observed XEB: {observed_xeb}")
    print(f"Noiseless basis XEB: {basis_xeb}")
    print(f"Est. noisy basis XEB: {basis_xeb * fidelity_from_noise}")
    print(f"Sampling time: {time.time() - start}")
    print("")
    observations.append(observed_xeb)

print(f"Average XEB: {sum(observations) / n_cliffords}")


##pytket_circ.measure_all()
##shots = 75000
##
##start = time.time()
##result = backend.run_circuit(pytket_circ, n_shots=shots)
##observed_xeb = (sum(abs(target_state[tuple(shot)])**2 for shot in result.get_shots()) / shots) * 2**n - 1
##print(f"Observed XEB: {observed_xeb}")
##print(f"Sampling time: {time.time() - start}")
##
### Should equal XEB in the computational basis
##perfect_xeb = (sum(abs((output_state * target_state).flatten() ** 2)) * 2**n - 1)
##print(f"Perfect XEB: {perfect_xeb}")
