# TODO: organize imports
from optimize.optimize_qujax import *
from clifford_utils.random_stabilizer import *
from pytket import Circuit
from pytket.extensions.quantinuum import QuantinuumBackend, QuantinuumAPIOffline
from pytket.extensions.quantinuum.backends.api_wrappers import QuantinuumAPI
from pytket.extensions.quantinuum.backends.credential_storage import (
    QuantinuumConfigCredentialStorage,
)
from pytket.extensions.quantinuum.backends.leakage_gadget import get_leakage_gadget_circuit
from pytket import Qubit, Bit

from qujax.statetensor import apply_gate
from qujax.gates import *

import time
import json

n = 8
depth = 10
start = time.time()
online = True
noisy = True
detect_leakage = False
submit_job = False

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


if online:
    backend = QuantinuumBackend(
        device_name="H1-1E",
        api_handler=QuantinuumAPI(token_store=QuantinuumConfigCredentialStorage()),
    )
else:
    api_offline = QuantinuumAPIOffline()
    backend = QuantinuumBackend(device_name="H1-1LE", api_handler=api_offline)

stab_gates = stabilizer_gate_list_ag(n)
stab_gates.reverse()

# TODO this code isn't used right now
n_cliffords = 1
shots_per_clifford = 1000
for i in range(n_cliffords):
    start = time.time()
    toggles = random_stabilizer_toggles_ag(n)
    toggles.reverse()
    num_stab_gates = len(toggles)

    cliff_circ = Circuit(n)
    max_register_size = 32
    
    scoring_state = target_state
    cliff_output_state = output_state
    for i in range(num_stab_gates):
        control_bit = Bit("clifford" + str(i // max_register_size), i % max_register_size)
        cliff_circ.add_bit(control_bit)
        cliff_circ.add_c_setbits([toggles[i]], [control_bit])
        
        gate_name, qubits = stab_gates[i]
        if gate_name == "x":
            cliff_circ.X(*qubits, condition=control_bit)
            qujax_gate = X
        elif gate_name == "h":
            cliff_circ.H(*qubits, condition=control_bit)
            qujax_gate = H
        elif gate_name == "s":
            cliff_circ.Sdg(*qubits, condition=control_bit)
            qujax_gate = Sdg
        elif gate_name == "cz":
            cliff_circ.CZ(*qubits, condition=control_bit)
            qujax_gate = CZ
        else:
            raise Exception("invalid gate name: " + gate_name)

        if toggles[i]:
            scoring_state = apply_gate(scoring_state, qujax_gate, qubits)
            cliff_output_state = apply_gate(cliff_output_state, qujax_gate, qubits)

    overall_circ = Circuit(n)
    for i in range(num_stab_gates // max_register_size + bool(num_stab_gates % max_register_size)):
        overall_circ.add_c_register("clifford" + str(i), max_register_size)
    overall_circ.append(state_prep_circ)

    if detect_leakage:
        for i in range(n):
            leakage_gadget = get_leakage_gadget_circuit(
                Qubit("q", i),
                Qubit("q", n+1),
                Bit("leakage_detection_bit", i),
            )
            overall_circ.append(leakage_gadget)
    
    # Add a barrier between state preparation and Clifford measurement
    overall_circ.add_barrier(overall_circ.qubits + overall_circ.bits)
    overall_circ.append(cliff_circ)
    overall_circ.measure_all()
    backend.default_compilation_pass().apply(overall_circ)

    basis_xeb = sum(abs((scoring_state * cliff_output_state).flatten() ** 2)) * 2**n - 1
    print(f"Noiseless basis XEB: {basis_xeb}")
    print(f"Est. noisy basis XEB: {basis_xeb * fidelity_from_noise}")

    if submit_job:
        result_handle = backend.process_circuit(overall_circ, n_shots=shots_per_clifford)

        time_str = time.asctime().replace("/","_").replace(":","-").replace(" ","_")
        file = open(f"job_handles/{n}_{depth}_{time_str}.txt", "w")
        qc_str = json.dumps(overall_circ.to_dict())
        file.write(str(n) + "\n")
        file.write(str(depth) + "\n")
        file.write(str(online) + "\n")
        file.write(str(noisy) + "\n")
        file.write(str(detect_leakage) + "\n")
        file.write(str(toggles) + "\n")
        file.write(str(target_state.tolist()) + "\n")
        file.write(str(scoring_state.tolist()) + "\n")
        file.write(str(cliff_output_state.tolist()) + "\n")
        file.write(qc_str + "\n")
        file.write(str(result_handle) + "\n")
        file.close()

        for trial in range(100):
            time.sleep(10)
            result, job_status = backend.get_partial_result(result_handle)
            if job_status.status.name == 'COMPLETED':
                print("Done!")
                break
            elif job_status.status.name == 'ERROR':
                print("ERROR")
                break
            elif job_status.status.name == 'CANCELLED':
                print("CANCELLED")
                break
            else:
                print("Waiting...")

        measured_bits = [Bit("c", i) for i in range(n)]
        if detect_leakage:
            measured_bits = measured_bits + [Bit("leakage_detection_bit", i) for i in range(n)]
        all_shots_with_leakage_results = result.get_shots(cbits=measured_bits)
        all_shots = [shot[:n] for shot in all_shots_with_leakage_results]
        pruned_shots = [shot[:n] for shot in all_shots_with_leakage_results if not any(shot[n:])]
        
        xeb_scores = [abs(scoring_state[tuple(shot)]) ** 2 * 2**n - 1 for shot in all_shots]
        pruned_xeb_scores = [abs(scoring_state[tuple(shot)]) ** 2 * 2**n - 1 for shot in pruned_shots]
        
        observed_xeb = sum(xeb_scores) / len(xeb_scores)
        pruned_xeb = sum(pruned_xeb_scores) / len(pruned_xeb_scores)
        
        print(f"Observed XEB: {observed_xeb}")
        print(f"Observed pruned XEB: {pruned_xeb}")
        print(f"Sampling time: {time.time() - start}")
