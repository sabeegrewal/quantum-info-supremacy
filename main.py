from optimize.optimize_jax import *

from ansatz.ansatz_jax import *

from utils import rand
from utils.process_io import await_job, print_results
from utils.circuit import apply_clifford, make_clifford_circuit, stitch_circuits
from utils.random_stabilizer import random_stabilizer_toggles_ag


from pytket.extensions.quantinuum import QuantinuumBackend, QuantinuumAPIOffline
from pytket.extensions.quantinuum.backends.api_wrappers import QuantinuumAPI
from pytket.extensions.quantinuum.backends.credential_storage import (
    QuantinuumConfigCredentialStorage,
)

import numpy as np
import time


n = 4
depth = 4
noisy = True
device_name = "H1-1LE"
detect_leakage = False

submit_job = True
n_stitches = 5
n_shots = 1
start_seed = 0
n_jobs = 10

print("-" * 30)
print(f"n               : {n}")
print(f"depth           : {depth}")
print(f"noisy           : {noisy}")
print(f"device_name     : {device_name}")
print(f"detect_leakage  : {detect_leakage}")
print(f"submit_job      : {submit_job}")
print(f"n_stitches      : {n_stitches}")
print(f"n_shots         : {n_shots}")
print("")
print(f"start_seed      : {start_seed}")
print(f"n_jobs          : {n_jobs}")
print(f"n_submissions   : {n_jobs//n_stitches}")
print("-" * 30)

# Prompt user
while True:
    user_input = input("Enter 'Y' to continue or 'N' to exit: ").strip().lower()
    if user_input == "y":
        print("Continuing...")
        break
    elif user_input == "n":
        print("Exiting...")
        exit()
    else:
        print("Invalid input. Please enter 'Y' or 'N'.")


for seed in range(1):
    print(f"seed {seed}")
    random_bits = rand.read_chunk(seed)
    rand_gen = rand.TrueRandom(random_bits)

    # First do all of the randomness generation
    target_state_r = rand_gen.normal(size=([2] * n))
    target_state_i = rand_gen.normal(size=([2] * n))
    target_state = (target_state_r + 1j * target_state_i) / 2 ** ((n + 1) / 2)

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
        detect_leakage,
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
            job_data = job_data.JobData(
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
                result_handle,
            )
            job_data.save()
            result = await_job(backend, result_handle)

        print_results([scoring_state] * n_stitches, detect_leakage, result)
