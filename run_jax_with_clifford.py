from optimize.optimize_jax import *
from clifford_utils.random_stabilizer import *

from pytket import Circuit, Qubit, Bit
from pytket.extensions.quantinuum import QuantinuumBackend, QuantinuumAPIOffline
from pytket.extensions.quantinuum.backends.api_wrappers import QuantinuumAPI
from pytket.extensions.quantinuum.backends.credential_storage import (
    QuantinuumConfigCredentialStorage,
)
from pytket.extensions.quantinuum.backends.leakage_gadget import get_leakage_gadget_circuit
from pytket.passes import SequencePass

from qujax.statetensor import apply_gate
from qujax.gates import *

import numpy as np
import time
import json
import random

def max_register_size():
    """The maximum classical register size that can be used on
    quantinuum devices.

    Returns
    -------
    int
        Currently defaults to 32; see
        https://docs.quantinuum.com/tket/extensions/pytket-quantinuum/changelog.html#id4
    """
    return 32

def make_and_apply_cliff(target_state, output_state):
    """Generate a random Clifford circuit and apply it to the target and output states.

    Parameters
    ----------
    target_state : array
        Complex array of shape `[2] * n` for some `n`.
    output_state : array
        Complex array of shape `[2] * n` for some `n`.

    Returns
    -------
    (list[bool], Circuit, array, array)
        A list of which Clifford gates to toggle using Aaronson-Gottesman compilation of the measurement,
        a pytket Circuit implementing the Clifford measurement, and the states obtained by applying the
        Clifford to the target state and output state, respectively.
    """
    
    # Infer the number of qubits from the dimension
    n = len(target_state.shape)

    # Generate toggles for preparing a random stabilizer state
    toggles = random_stabilizer_toggles_ag(n)
    # Reverse to turn into a measurement
    toggles.reverse()
    num_stab_gates = len(toggles)
    # Also get the corresponding gates
    stab_gates = stabilizer_gate_list_ag(n)
    stab_gates.reverse()

    # Now make the circuit
    cliff_circ = Circuit(n)
    scoring_state = target_state
    cliff_output_state = output_state
    for i in range(num_stab_gates):
        # Add a classically-controlled gate for each toggle
        # Divide the bits into registers of size max_register_size()
        control_bit = Bit("clifford" + str(i // max_register_size()), i % max_register_size())
        cliff_circ.add_bit(control_bit)
        # Set the classical control bit according to the toggle
        cliff_circ.add_c_setbits([toggles[i]], [control_bit])

        # Now add the relevant controlled gate to the circuit
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

        # Apply the gate to the two states
        if toggles[i]:
            scoring_state = apply_gate(scoring_state, qujax_gate, qubits)
            cliff_output_state = apply_gate(cliff_output_state, qujax_gate, qubits)

    return (toggles, cliff_circ, scoring_state, cliff_output_state)

def make_overall_circ(state_prep_circ, cliff_circ, backend, detect_leakage, num_leakage_qubits=1):
    """Make the overall pytket Circuit that includes both state preparation and measurement.

    Parameters
    ----------
    state_prep_circ : Circuit
        pytket Circuit implementing state preparation. Should have only a quantum register.
    cliff_circ : Circuit
        pytket Circuit implementing Clifford measurement.
        Should have the same number of qubits as `state_prep_circ` and classical registers
        for the classically-controlled Clifford gates.
    backend : Backend
        Backend to use for compilation.
    detect_leakage : bool
        Whether to use the leakage detection gadget after state preparation.
    num_leakage_qubits : int
        Optional number of qubits to be used for leakage detection. Defaults to 1.

    Returns
    -------
    Circuit
        A combined and compiled pytket Circuit. 
    """
    
    # Make an empty circuit with n qubits
    n = state_prep_circ.n_qubits
    assert cliff_circ.n_qubits == n
    overall_circ = Circuit(n)

    # Add classically-controlled registers for the Clifford gates
    num_stab_gates = num_stab_gates_ag(n)
    # Divide the stabilizer gate controls into registers of size max_register_size()
    for i in range(num_stab_gates // max_register_size() + bool(num_stab_gates % max_register_size())):
        overall_circ.add_c_register("clifford" + str(i), max_register_size())

    # State preparation
    overall_circ.append(state_prep_circ)

    # Leakage detection
    if detect_leakage:
        for i in range(n):
            leakage_gadget = get_leakage_gadget_circuit(
                Qubit("q", i),
                Qubit("q", n + (i % num_leakage_qubits)),
                Bit("leakage_detection_bit", i),
            )
            overall_circ.append(leakage_gadget)
    
    # Barrier between state preparation and Clifford measurement
    overall_circ.add_barrier(overall_circ.qubits + overall_circ.bits)

    # Clifford
    overall_circ.append(cliff_circ)

    # Measurement
    overall_circ.measure_all()
    
    # Custom compilation pass:
    # Level 2 optimisation doesn't get the classically-controlled CZ gates right,
    # but level 3 optimisation does a couple of operations that we don't want.
    # So, we take the level 3 passes and remove the unneeded ones.
    default_pass_list = backend.default_compilation_pass(optimisation_level=3).get_sequence()
    my_pass_list = [
        pas for pas in default_pass_list
        if pas.to_dict()["StandardPass"]["name"] not in ["RemoveBarriers", "GreedyPauliSimp"]
        ]
    compilation_pass = SequencePass(my_pass_list)
    compilation_pass.apply(overall_circ)

    return overall_circ

def save_result_handle(n, depth, online, noisy, detect_leakage, toggles, target_state, scoring_state,
                       cliff_output_state, overall_circ, result_handle, filename=None):
    """Save the data for this job submission locally so that it can be recovered later.

    Parameters
    ----------
    n : int
        Number of qubits.
    depth : int
        Depth of the state preparation circuit, as measured by 2-qubit gate layers.
    online : bool
        Whether the job was run online or locally.
    noisy : bool
        Whether the circuit was optimized using the noisy or noiseless loss function.
    detect_leakage : bool
        Whether the circuit uses the leakage detection gadget.
    toggles : list[bool]
        A list of which Clifford gates are toggled using Aaronson-Gottesman compilation of the measurement.
    target_state : array
        Complex array of shape `[2] * n`.
    scoring_state : array
        Complex array of shape `[2] * n`
    cliff_output_state : array
        Complex array of shape `[2] * n`
    overall_circ : Circuit
        The circuit submitted to the backend.
    result_handle : ResultHandle
        Handle used to access the job result.
    filename : str
        Optional filename to which the data should be saved. Defaults to a file location in the `job_handles`
        folder with a name that is a concatenation of the number of qubits, depth, and current time.
    """
    if filename is None:
        time_str = time.asctime().replace("/","_").replace(":","-").replace(" ","_")
        filename = f"job_handles/{n}_{depth}_{time_str}.txt"
    
    file = open(filename, "w")
    file.write(str(n) + "\n")
    file.write(str(depth) + "\n")
    file.write(str(online) + "\n")
    file.write(str(noisy) + "\n")
    file.write(str(detect_leakage) + "\n")
    file.write(str(toggles) + "\n")
    file.write(str(target_state.tolist()) + "\n")
    file.write(str(scoring_state.tolist()) + "\n")
    file.write(str(cliff_output_state.tolist()) + "\n")
    file.write(json.dumps(overall_circ.to_dict()) + "\n")
    file.write(str(result_handle) + "\n")
    file.close()

def load_result_handle(filename):
    """Load the data that was saved using `save_result_handle`.

    Parameters
    ----------
    filename : str
        File to load from.

    Returns
    -------
    (int, int, bool, bool, bool, list[bool], array, array, array, Circuit, ResultHandle)
        The data saved using `save_result_handle`.
    """
    
    file = open(filename, "r")

    n = int(file.readline()[:-1])
    depth = int(file.readline()[:-1])
    online = bool(file.readline()[:-1])
    noisy = bool(file.readline()[:-1])
    detect_leakage = bool(file.readline()[:-1])
    # TODO ideally something safer than running eval()...
    toggles = eval(file.readline()[:-1])
    target_state = np.array(eval(file.readline()[:-1]))
    scoring_state = np.array(eval(file.readline()[:-1]))
    cliff_output_state = np.array(eval(file.readline()[:-1]))
    overall_circ = Circuit.from_dict(json.loads(file.readline()[:-1]))
    result_handle = ResultHandle.from_str(file.readline()[:-1])

    return (n, depth, online, noisy, detect_leakage, toggles, target_state, scoring_state, cliff_output_state, overall_circ, result_handle)

def await_job(backend, result_handle, sleep=20, trials=1000):
    """Load the data that was saved using `save_result_handle`.

    Parameters
    ----------
    backend : Backend
        Backend to submit the job to.
    result_handle : ResultHandle
        Result handle for the job.
    sleep : real
        Optional time in seconds to wait between intervals of checking whether the job is finished.
        Defaults to 20.
    trials : int
        Number of trials to attempt getting the results.
        Defaults to 1000.

    Returns
    -------
    BackendResult
        The BackendResult obtained either when the job is completed,
        errored, cancelled, or the number of trials runs out.
    """
    
    for trial in range(1000):
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
        time.sleep(sleep)
    return result

def print_results(scoring_state, detect_leakage, result):
    """Compute and print XEB scores from the results on the given scoring state.

    Parameters
    ----------
    scoring_state : array
        Complex array of shape `[2] * n' for some `n`.
    detect_leakage : bool
        Whether to the leakage detection gadget was used.
    result : BackendResult
        Result containing samples from the output distribution.
    """

    n = len(scoring_state.shape)

    # The bits that were measured in each shot
    measured_bits = [Bit("c", i) for i in range(n)]
    if detect_leakage:
        measured_bits = measured_bits + [Bit("leakage_detection_bit", i) for i in range(n)]
    all_shots_with_leakage_results = result.get_shots(cbits=measured_bits)
    all_shots = [shot[:n] for shot in all_shots_with_leakage_results]
    # Prunes all shots detected as leaky
    pruned_shots = [shot[:n] for shot in all_shots_with_leakage_results if not any(shot[n:])]

    xeb_scores = [abs(scoring_state[tuple(shot)]) ** 2 * 2**n - 1 for shot in all_shots]
    pruned_xeb_scores = [abs(scoring_state[tuple(shot)]) ** 2 * 2**n - 1 for shot in pruned_shots]
    
    observed_xeb = sum(xeb_scores) / len(xeb_scores)
    pruned_xeb = sum(pruned_xeb_scores) / len(pruned_xeb_scores)
    
    print(f"Observed XEB: {observed_xeb}")
    print(f"Observed pruned XEB: {pruned_xeb}")

n = 12
depth = 86
online = True
noisy = True
detect_leakage = False
submit_job = True

for seed in range(1):
    print(f"seed {seed}")
    np.random.seed(seed)
    random.seed(seed)

    start = time.time()

    target_state = np.random.normal(size=([2] * n)) + 1j * np.random.normal(size=([2] * n))
    target_state = target_state / np.linalg.norm(target_state)
    # TODO technically we don't need to normalize

    opt = optimize(target_state, depth, noisy=noisy)
    output_state = output_state(n, opt.x)
    noiseless_fidelity = -loss(opt.x, target_state)
    fidelity_from_noise = fidelity_from_noise(n, opt.x)
    print(f"Noiseless fidelity: {noiseless_fidelity}")
    print(f"Estimated fidelity due to noise: {fidelity_from_noise}")
    print(f"Estimated overall fidelity: {fidelity_from_noise * noiseless_fidelity}")
    print(f"Optimization time: {time.time() - start}")
    print("")

    opt_params = opt.x
    state_prep_circ = make_circuit(n, opt_params, "pytket")

    if online:
        backend = QuantinuumBackend(
            device_name="H1-1E",
            api_handler=QuantinuumAPI(token_store=QuantinuumConfigCredentialStorage()),
        )
    else:
        api_offline = QuantinuumAPIOffline()
        backend = QuantinuumBackend(device_name="H1-1LE", api_handler=api_offline)

    toggles, cliff_circ, scoring_state, cliff_output_state = make_and_apply_cliff(target_state, output_state)
    overall_circ = make_overall_circ(state_prep_circ, cliff_circ, backend, detect_leakage)            

    basis_xeb = sum(abs((scoring_state * cliff_output_state).flatten() ** 2)) * 2**n - 1
    print(f"Noiseless basis XEB: {basis_xeb}")
    print(f"Est. noisy basis XEB: {basis_xeb * fidelity_from_noise}")

    if submit_job:
        result_handle = backend.process_circuit(overall_circ, n_shots=10000)
        save_result_handle(n, depth, online, noisy, detect_leakage, toggles, target_state, scoring_state,
                           cliff_output_state, overall_circ, result_handle)
        result = await_job(backend, result_handle)
        print_results(scoring_state, detect_leakage, result)
    
