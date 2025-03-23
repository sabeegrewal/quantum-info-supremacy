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
from concurrent.futures import ProcessPoolExecutor


def run_optimization(job):
    i = job["i"]
    s = job["seed"]
    target_state = job["target_state"]
    n = job["n"]
    depth = job["depth"]
    noisy = job["noisy"]

    # Ensure initial parameters are chosen consistently pseudorandomly
    np.random.seed(s)

    start = time.time()

    opt = optimize(target_state, depth, noisy=noisy)
    opt_params = opt.x

    out_state = output_state(n, opt_params)
    noiseless_fidelity = -loss(opt_params, target_state)
    noisy_fidelity = fidelity_from_noise(n, opt_params)

    return {
        "i": i,
        "opt_params": opt_params,
        "output_state": out_state,
        "noiseless_fidelity": noiseless_fidelity,
        "noisy_fidelity": noisy_fidelity,
        "time": time.time() - start,
    }


if __name__ == "__main__":

    n = 4
    depth = 4
    noisy = True
    device_name = "H1-1LE"
    detect_leakage = False

    submit_job = True
    n_stitches = 5
    n_shots = 1
    start_seed = 0
    n_jobs = 3

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
    print(f"n_submissions   : {int(np.ceil(n_jobs/n_stitches))}")
    print("-" * 30)

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

    # Double check...
    if device_name == "H1-1":
        print(f"Are you sure? You are submitting to the real machine!")
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

    for seed in range(start_seed, start_seed + n_jobs, n_stitches):
        batch = list(range(seed, min(seed + n_stitches, start_seed + n_jobs)))
        print(f"seeds: {batch}")
        random_bits_list = [rand.read_chunk(s) for s in batch]
        rand_gens = [rand.TrueRandom(random_bits) for random_bits in random_bits_list]

        # First do all of the randomness generation
        target_states_r = [rand_gen.normal(size=([2] * n)) for rand_gen in rand_gens]
        target_states_i = [rand_gen.normal(size=([2] * n)) for rand_gen in rand_gens]
        target_states = [
            (target_state_r + 1j * target_state_i) / 2 ** ((n + 1) / 2)
            for target_state_r, target_state_i in zip(target_states_r, target_states_i)
        ]

        ag_toggles_list = [
            random_stabilizer_toggles_ag(n, rand_gen) for rand_gen in rand_gens
        ]
        reversed_ag_toggles_list = [
            list(reversed(ag_toggles)) for ag_toggles in ag_toggles_list
        ]

        # Optimize
        jobs = []
        for i, s in enumerate(batch):
            jobs.append(
                {
                    "i": i,
                    "seed": s,
                    "target_state": target_states[i],
                    "n": n,
                    "depth": depth,
                    "noisy": noisy,
                }
            )
        with ProcessPoolExecutor() as executor:
            optimization_results = list(executor.map(run_optimization, jobs))
        # Make sure we preserve order (this shouldn't be needed, but doing to be extra safe)
        optimization_results.sort(key=lambda r: r["i"])

        output_states = [r["output_state"] for r in optimization_results]
        opt_params_list = [r["opt_params"] for r in optimization_results]
        noisy_fidelities = [r["noisy_fidelity"] for r in optimization_results]

        for r in optimization_results:
            print(f"Noiseless fidelity: {r['noiseless_fidelity']}")
            print(f"Estimated fidelity due to noise: {r['noisy_fidelity']}")
            print(
                f"Estimated overall fidelity: {r['noisy_fidelity'] * r['noiseless_fidelity']}"
            )
            print(f"Optimization time: {r['time']}")
            print("")

        if device_name == "H1-1LE":
            api_offline = QuantinuumAPIOffline()
            backend = QuantinuumBackend(
                device_name=device_name, api_handler=api_offline
            )
        else:
            backend = QuantinuumBackend(
                device_name=device_name,
                api_handler=QuantinuumAPI(
                    token_store=QuantinuumConfigCredentialStorage()
                ),
            )

        state_prep_circs = [
            make_ansatz_circuit(n, opt_params, method="pytket")
            for opt_params in opt_params_list
        ]
        cliff_circs = [
            make_clifford_circuit(n, reversed_ag_toggles, backend)
            for reversed_ag_toggles in reversed_ag_toggles_list
        ]
        scoring_states = [
            apply_clifford(target_state, reversed_ag_toggles)
            for target_state, reversed_ag_toggles in zip(
                target_states, reversed_ag_toggles_list
            )
        ]
        cliff_output_states = [
            apply_clifford(output_state, reversed_ag_toggles)
            for output_state, reversed_ag_toggles in zip(
                output_states, reversed_ag_toggles_list
            )
        ]

        basis_xebs = [
            sum(abs((scoring_state * cliff_output_state).flatten() ** 2)) * 2**n - 1
            for scoring_state, cliff_output_state in zip(
                scoring_states, cliff_output_states
            )
        ]
        print(f"Noiseless basis XEB: {[float(xeb) for xeb in basis_xebs]}")
        print(
            f"Est. noisy basis XEB: {[float(basis_xeb * nf) for basis_xeb, nf in zip(basis_xebs, noisy_fidelities)]}"
        )
        print("")

        overall_circ = stitch_circuits(
            state_prep_circs,
            cliff_circs,
            backend,
            detect_leakage,
        )

        if submit_job:
            if device_name == "H1-1LE":
                result = backend.run_circuit(overall_circ, n_shots=n_shots)
            else:
                result_handle = backend.process_circuit(overall_circ, n_shots=n_shots)
                job_data = job_data.JobData(
                    n,
                    depth,
                    noisy,
                    device_name,
                    detect_leakage,
                    batch,
                    target_states,
                    reversed_ag_toggles_list,
                    opt_params_list,
                    overall_circ,
                    result_handle,
                )
                job_data.save()
                result = await_job(backend, result_handle)

            print_results(scoring_states, detect_leakage, result)
