from optimize.optimize_jax import *
from clifford_utils.random_stabilizer import *

from randomness_utils import rand

from pytket import Circuit, Qubit, Bit
from pytket.extensions.quantinuum import QuantinuumBackend, QuantinuumAPIOffline
from pytket.extensions.quantinuum.backends.api_wrappers import QuantinuumAPI
from pytket.extensions.quantinuum.backends.credential_storage import (
    QuantinuumConfigCredentialStorage,
)
from pytket.extensions.quantinuum.backends.leakage_gadget import (
    get_leakage_gadget_circuit,
)
from pytket.passes import SequencePass
from pytket.backends.resulthandle import ResultHandle

from qujax.statetensor import apply_gate
from qujax.gates import *

import numpy as np
import time
import json


def make_clifford_circuit(n, reversed_ag_toggles, backend):
    """Make a Clifford measurement circuit for the given list of toggles.

    Parameters
    ----------
    n : int
        Number of qubits.
    reversed_ag_toggles : list[bool]
        List of toggles corresponding to a reversed Aaronson-Gottesman state preparation circuit.
        Should have length equal to `num_stab_gates_ag(n)`.
    backend : Backend
        Backend to use for compilation.
        Needed to ensure the classical registers have appropriate width.

    Returns
    -------
    Circuit
        A pytket Circuit implementing the Clifford measurement.
    """
    
    cliff_circ = Circuit(n)
    
    # Add classical registers for the Clifford gate controls
    num_stab_gates = num_stab_gates_ag(n)
    # Maximum classical register width
    max_register_width = backend.backend_info.get_misc("max_classical_register_width")
    # Divide the stabilizer gate controls into registers of size max_register_width
    for i in range(
        num_stab_gates // max_register_width
        + bool(num_stab_gates % max_register_width)
    ):
        cliff_circ.add_c_register(f"clifford{i}", max_register_width)


    # List of all control bits, in order
    bits = [
        Bit(
            f"clifford{i // max_register_width}", i % max_register_width
        ) for i in range(num_stab_gates)
    ]
    # Set all of the classical toggles before doing any gates
    cliff_circ.add_c_setbits(reversed_ag_toggles, bits)

    # Get the corresponding gates
    stab_gates = list(reversed(stabilizer_gate_list_ag(n)))

    # Now apply the gates
    for i in range(num_stab_gates):
        # Add a classically-controlled gate for each toggle
        # Divide the bits into registers of size max_register_size()
        control_bit = Bit(f"clifford{i // max_register_width}", i % max_register_width)

        # Add the relevant controlled gate to the circuit
        gate_name, qubits = stab_gates[i]
        if gate_name == "x":
            cliff_circ.X(*qubits, condition=control_bit)
        elif gate_name == "h":
            cliff_circ.H(*qubits, condition=control_bit)
        elif gate_name == "s":
            cliff_circ.Sdg(*qubits, condition=control_bit)
        elif gate_name == "cz":
            cliff_circ.CZ(*qubits, condition=control_bit)
        else:
            raise Exception("invalid gate name: " + gate_name)

    return cliff_circ


def apply_clifford(state, reversed_ag_toggles):
    """Apply a Clifford measurement circuit for the given list of toggles to the given state.

    Parameters
    ----------
    state : array
        Complex array of shape `[2] * n` for some `n`.
    reversed_ag_toggles : list[bool]
        List of toggles corresponding to a reversed Aaronson-Gottesman state preparation circuit.
        Should have length equal to `num_stab_gates_ag(n)`.

    Returns
    -------
    array, array
        The state obtained by applying the Clifford measurement to the input state.
    """

    # Infer the number of qubits from the dimension
    n = len(state.shape)
    num_stab_gates = num_stab_gates_ag(n)

    # Get the corresponding gates
    stab_gates = list(reversed(stabilizer_gate_list_ag(n)))

    for i in range(num_stab_gates):
        if reversed_ag_toggles[i]:
            # Compute and apply the relevant gate
            gate_name, qubits = stab_gates[i]
            if gate_name == "x":
                qujax_gate = X
            elif gate_name == "h":
                qujax_gate = H
            elif gate_name == "s":
                qujax_gate = Sdg
            elif gate_name == "cz":
                qujax_gate = CZ
            else:
                raise Exception("invalid gate name: " + gate_name)
                
            state = apply_gate(state, qujax_gate, qubits)

    return state


def stitch_circuits(
    state_prep_circs, cliff_circs, backend, detect_leakage, num_leakage_qubits=1
):
    """Make the overall pytket Circuit that includes both state preparation and measurement
    for each of the given circuits.

    Parameters
    ----------
    state_prep_circ : list[Circuit]
        List of pytket Circuits implementing state preparation.
        Each circuit should have only a quantum register, each with the
        same number of qubits.
    cliff_circ : list[Circuit]
        List of pytket Circuits implementing Clifford measurement.
        Should have the same length as `state_prep_circs`.
        Each circuit should have the same number of qubits as hose in `state_prep_circs`,
        and classical registers for the classically-controlled Clifford gates.
    backend : Backend
        Backend to use for compilation.
    detect_leakage : bool
        Whether to use the leakage detection gadget after state preparation.
    num_leakage_qubits : int
        Optional number of qubits to be used for leakage detection. Defaults to 1.

    Returns
    -------
    Circuit
        A combined and compiled pytket Circuit that stitches all of the runs together.
    """
    
    n = state_prep_circs[0].n_qubits
    num_circs = len(state_prep_circs)

    # Check that the input is valid
    assert len(cliff_circs) == num_circs
    for circ_idx in range(num_circs):
        assert state_prep_circs[circ_idx].n_qubits == n
        assert cliff_circs[circ_idx].n_qubits == n

    # Make an empty circuit with n qubits
    overall_circ = Circuit(n)

    # Add classical registers for leakage detection, if necessary
    if detect_leakage:
        for circ_idx in range(num_circs):
            # Register width is number of qubits
            overall_circ.add_c_register(f"leakage_detection{circ_idx}", n)

    # Add classical registers for the Clifford gate controls
    # These registers get reused for each of the stitched sub-circuits
    num_stab_gates = num_stab_gates_ag(n)
    # Maximum classical register width
    max_register_width = backend.backend_info.get_misc("max_classical_register_width")
    # Divide the stabilizer gate controls into registers of size max_register_width
    for i in range(
        num_stab_gates // max_register_width
        + bool(num_stab_gates % max_register_width)
    ):
        overall_circ.add_c_register(f"clifford{i}", max_register_width)

    # Add classical registers for the measurement results
    for circ_idx in range(num_circs):
        # Register width is number of qubits
        overall_circ.add_c_register(f"measurement{circ_idx}", n)

    # Stitch the circuits together sequentially
    for circ_idx in range(num_circs):
        if circ_idx > 0:
            # After the first iteration, add a barrier and reset all qubits to 0
            overall_circ.add_barrier(overall_circ.qubits + overall_circ.bits)
            for i in range(n):
                overall_circ.Reset(i)

        # State preparation
        overall_circ.append(state_prep_circs[circ_idx])

        # Leakage detection
        if detect_leakage:
            for i in range(n):
                leakage_gadget = get_leakage_gadget_circuit(
                    Qubit(i), # Circuit qubit
                    Qubit(n + (i % num_leakage_qubits)), # Postselection qubit
                    Bit(f"leakage_detection{circ_idx}", i), # Store result here
                )
                overall_circ.append(leakage_gadget)

        # Barrier between state preparation and Clifford measurement
        overall_circ.add_barrier(overall_circ.qubits + overall_circ.bits)

        # Clifford
        overall_circ.append(cliff_circs[circ_idx])

        # Measurement
        for i in range(n):
            overall_circ.Measure(Qubit(i), Bit(f"measurement{circ_idx}", i))

    # Custom compilation pass:
    # Level 2 optimisation doesn't get the classically-controlled CZ gates right,
    # but level 3 optimisation does a couple of operations that we don't want.
    # So, we take the level 3 passes and remove the unneeded ones.
    default_pass_list = backend.default_compilation_pass(
        optimisation_level=3
    ).get_sequence()
    my_pass_list = [
        pas
        for pas in default_pass_list
        if pas.to_dict()["StandardPass"]["name"]
        not in ["RemoveBarriers", "GreedyPauliSimp"]
    ]
    compilation_pass = SequencePass(my_pass_list)
    compilation_pass.apply(overall_circ)

    # TODO add some more sanity checks before returning
    # See https://docs.quantinuum.com/systems/trainings/knowledge_articles/circuit_stitching.html

    return overall_circ


class JobData:
    """Class for storing data associated with a job submission.

    Parameters
    ----------
    n : int
        Number of qubits.
    depth : int
        Depth of the state preparation circuit, as measured by 2-qubit gate layers.
    noisy : bool
        Whether the circuits were optimized using the noisy or noiseless loss function.
    device_name : str
        Device name, e.g. "H1-1", "H1-1E", "H1-1LE".
    detect_leakage : bool
        Whether the circuit uses the leakage detection gadget.
    seeds : list[int]
        List of seeds used for randomness in each of the trials.
    target_states : list[array]
        List of complex arrays of shape `[2] * n`.
    reversed_ag_toggle_lists : list[list[bool]]
        A list of which Clifford gates are toggled using Aaronson-Gottesman compilation of the measurement,
        one for each trial.
    opt_param_lists : list[array]
        List of real arrays containing the optimized circuit parameters for each state.
    overall_circ : Circuit
        The circuit submitted to the backend.
    result_handle : ResultHandle
        Handle used to access the job result.
    """
    def __init__(
        self,
        n,
        depth,
        noisy,
        device_name,
        detect_leakage,
        seeds,
        target_states,
        reversed_ag_toggle_lists,
        opt_param_lists,
        overall_circ,
        result_handle
    ):
        self.n = n
        self.depth = depth
        self.noisy = noisy
        self.device_name = device_name
        self.detect_leakage = detect_leakage
        self.seeds = seeds
        self.target_states = np.array(target_states)
        self.reversed_ag_toggle_lists = reversed_ag_toggle_lists
        self.opt_param_lists = np.array(opt_param_lists)
        self.overall_circ = overall_circ
        self.result_handle = result_handle

    def save(self, filename=None):
        """Save the data for this job submission locally so that it can be recovered later.

        Parameters
        ----------
        filename : str
            Optional filename to which the data should be saved. Defaults to a file location in the `job_handles`
            folder with a name that is a concatenation of the number of qubits, depth, and current time.
        """
        if filename is None:
            time_str = time.asctime().replace("/","_").replace(":","-").replace(" ","_")
            filename = f"job_handles/{n}_{depth}_{time_str}.txt"

        file = open(filename, "w")
        file.write(str(self.n) + "\n")
        file.write(str(self.depth) + "\n")
        file.write(str(self.noisy) + "\n")
        file.write(self.device_name + "\n")
        file.write(str(self.detect_leakage) + "\n")
        file.write(str(self.seeds) + "\n")
        file.write(str(self.target_states.tolist()) + "\n")
        file.write(str(self.reversed_ag_toggle_lists) + "\n")
        file.write(str(self.opt_param_lists.tolist()) + "\n")
        file.write(json.dumps(self.overall_circ.to_dict()) + "\n")
        file.write(str(self.result_handle) + "\n")
        file.close()

    @staticmethod
    def load(filename):
        file = open(filename, "r")

        n = int(file.readline()[:-1])
        depth = int(file.readline()[:-1])
        noisy = file.readline()[:-1] == "True"
        device_name = file.readline()[:-1]
        detect_leakage = file.readline()[:-1] == "True"
        # TODO ideally something safer than running eval()...
        seeds = eval(file.readline()[:-1])
        target_states = np.array(eval(file.readline()[:-1]))
        reversed_ag_toggle_lists = eval(file.readline()[:-1])
        opt_param_lists = np.array(eval(file.readline()[:-1]))
        overall_circ = Circuit.from_dict(json.loads(file.readline()[:-1]))
        result_handle = ResultHandle.from_str(file.readline()[:-1])

        return JobData(
            n,
            depth,
            noisy,
            device_name,
            detect_leakage,
            seeds,
            target_states,
            reversed_ag_toggle_lists,
            opt_param_lists,
            overall_circ,
            result_handle,
        )


def await_job(backend, result_handle, sleep=20, trials=1000):
    """Retrieve a job from the server, waiting until completed.

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
        if job_status.status.name == "COMPLETED":
            print("Done!")
            break
        elif job_status.status.name == "ERROR":
            print("ERROR")
            break
        elif job_status.status.name == "CANCELLED":
            print("CANCELLED")
            break
        else:
            print("Waiting...")
        time.sleep(sleep)
    return result


def print_results(scoring_states, detect_leakage, result):
    """Compute and print XEB scores from the results on the given scoring states.

    Parameters
    ----------
    scoring_states : array
        List of complex arrays of shape `[2] * n' for some `n`.
    detect_leakage : bool
        Whether to the leakage detection gadget was used.
    result : BackendResult
        Result containing samples from the output distribution.
    """

    n = len(scoring_states[0].shape)

    for circ_idx in range(len(scoring_states)):
        # The bits that were measured in each shot
        measured_bits = [Bit(f"measurement{circ_idx}", i) for i in range(n)]
        if detect_leakage:
            measured_bits = measured_bits + [
                Bit(f"leakage_detection{circ_idx}", i) for i in range(n)
            ]
        all_shots_with_leakage_results = result.get_shots(cbits=measured_bits)
        all_shots = [shot[:n] for shot in all_shots_with_leakage_results]
        # Prunes all shots detected as leaky
        pruned_shots = [
            shot[:n] for shot in all_shots_with_leakage_results if not any(shot[n:])
        ]

        xeb_scores = [abs(scoring_states[circ_idx][tuple(shot)]) ** 2 * 2**n - 1 for shot in all_shots]
        pruned_xeb_scores = [
            abs(scoring_states[circ_idx][tuple(shot)]) ** 2 * 2**n - 1 for shot in pruned_shots
        ]

        observed_xeb = sum(xeb_scores) / len(xeb_scores)
        pruned_xeb = sum(pruned_xeb_scores) / len(pruned_xeb_scores)

        print(f"Observed XEB: {observed_xeb}")
        print(f"Observed pruned XEB: {pruned_xeb}")


n = 12
depth = 86
noisy = True
device_name = "H1-1SE"
detect_leakage = False

submit_job = True
n_stitches = 5
n_shots = 10000 // n_stitches

for seed in range(1):
    print(f"seed {seed}")
    random_bits = rand.read_chunk(seed)
    rand_gen = rand.TrueRandom(random_bits)

    # First do all of the randomness generation
    target_state_r = rand_gen.normal(size=([2] * n))
    target_state_i = rand_gen.normal(size=([2] * n))
    target_state = (target_state_r + 1j * target_state_i) / 2**((n + 1) / 2)

    ag_toggles = random_stabilizer_toggles_ag(n, rand_gen)
    reversed_ag_toggles = list(reversed(ag_toggles))

    # Optimize
    start = time.time()
    # Ensure initial parameters are chosen consistently pseudorandomly
    np.random.seed(seed)
    opt = optimize(target_state, depth, noisy=noisy)
    opt_params = opt.x

    output_state = output_state(n, opt_params)
    noiseless_fidelity = -loss(opt_params, target_state)
    fidelity_from_noise = fidelity_from_noise(n, opt_params)
    print(f"Noiseless fidelity: {noiseless_fidelity}")
    print(f"Estimated fidelity due to noise: {fidelity_from_noise}")
    print(f"Estimated overall fidelity: {fidelity_from_noise * noiseless_fidelity}")
    print(f"Optimization time: {time.time() - start}")
    print("")

    if device_name == "H1-1LE":
        api_offline = QuantinuumAPIOffline()
        backend = QuantinuumBackend(device_name=device_name, api_handler=api_offline)
    else:
        backend = QuantinuumBackend(
            device_name=device_name,
            api_handler=QuantinuumAPI(token_store=QuantinuumConfigCredentialStorage()),
        )

    state_prep_circ = make_ansatz_circuit(n, opt_params, method="pytket")
    cliff_circ = make_clifford_circuit(n, reversed_ag_toggles, backend)
    scoring_state = apply_clifford(target_state, reversed_ag_toggles)
    cliff_output_state = apply_clifford(output_state, reversed_ag_toggles)
    
    overall_circ = stitch_circuits(
        [state_prep_circ] * n_stitches,
        [cliff_circ] * n_stitches,
        backend,
        detect_leakage
    )

    basis_xeb = sum(abs((scoring_state * cliff_output_state).flatten() ** 2)) * 2**n - 1
    print(f"Noiseless basis XEB: {basis_xeb}")
    print(f"Est. noisy basis XEB: {basis_xeb * fidelity_from_noise}")

    if submit_job:
        if device_name == "H1-1LE":
            result = backend.run_circuit(overall_circ, n_shots=n_shots)
        else:
            if device_name == "H1-1":
                input("Submitting job to the real machine! Press enter to continue.")
            result_handle = backend.process_circuit(overall_circ, n_shots=n_shots)
            job_data = JobData(
                n,
                depth,
                noisy,
                device_name,
                detect_leakage,
                [seed] * n_stitches,
                [target_state] * n_stitches,
                [reversed_ag_toggles] * n_stitches,
                [opt_params] * n_stitches,
                overall_circ,
                result_handle
            )
            job_data.save()
            result = await_job(backend, result_handle)
            
        print_results([scoring_state] * n_stitches, detect_leakage, result)
