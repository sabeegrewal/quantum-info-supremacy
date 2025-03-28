from optimize.optimize_jax import *

from ansatz.ansatz_jax import *

from utils import rand
from utils.job_data import JobData
from utils.circuit import apply_clifford, make_clifford_circuit, stitch_circuits
from utils.random_stabilizer import random_stabilizer_toggles_ag


from pytket.extensions.quantinuum import QuantinuumBackend, QuantinuumAPIOffline
from pytket.extensions.quantinuum.backends.api_wrappers import QuantinuumAPI
from pytket.extensions.quantinuum.backends.credential_storage import (
    QuantinuumConfigCredentialStorage,
)

import numpy as np
import time
from datetime import datetime
import pathlib
import logging
from concurrent.futures import ProcessPoolExecutor

# This is necssary for platforms that run os.fork() by default in the ProcessPoolExectuor
# It is not necssasry on macOS, but it is on certain Linux platforms.
# import multiprocessing


def run_optimization(job):
    i = job["i"]
    s = job["seed"]
    target_state = job["target_state"]
    n = job["n"]
    depth = job["depth"]
    noisy = job["noisy"]

    # Ensure initial parameters are chosen consistently pseudorandomly
    rng = np.random.default_rng(s)

    start = time.time()

    opt = optimize(target_state, depth, noisy=noisy, rng=rng)
    opt_params = opt.x

    out_state = output_state(n, opt_params)
    noiseless_fidelity = -loss(opt_params, target_state)
    noise_fidelity = fidelity_from_noise(n, opt_params)

    return {
        "i": i,
        "opt_params": opt_params,
        "output_state": out_state,
        "noiseless_fidelity": noiseless_fidelity,
        "noise_fidelity": noise_fidelity,
        "time": time.time() - start,
    }


# I believe this is necessary on macs for ProcessPoolExecutor
if __name__ == "__main__":

    # This is necssary for platforms that run os.fork() by default in the ProcessPoolExectuor
    # It is not necssasry on macOS, but it is on certain Linux platforms.
    # multiprocessing.set_start_method("spawn")

    n = 12
    depth = 86
    noisy = True
    device_name = "H1-1"
    detect_leakage = False

    submit_job = True
    n_stitches = 5
    n_shots = 1
    start_seed = 0
    n_seeds = 5

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_path = f"logs/{device_name}/n_{n}_depth_{depth}"
    pathlib.Path(log_path).mkdir(parents=True, exist_ok=True)
    log_filename = (
        f"{log_path}/seeds_{start_seed}-{start_seed+n_seeds-1}_{timestamp}.txt"
    )
    logging.basicConfig(
        filename=log_filename,
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    logging.info("-" * 30)
    logging.info(f"n               : {n}")
    logging.info(f"depth           : {depth}")
    logging.info(f"noisy           : {noisy}")
    logging.info(f"device_name     : {device_name}")
    logging.info(f"detect_leakage  : {detect_leakage}")
    logging.info(f"submit_job      : {submit_job}")
    logging.info(f"n_stitches      : {n_stitches}")
    logging.info(f"n_shots         : {n_shots}")
    logging.info("")
    logging.info(f"start_seed      : {start_seed}")
    logging.info(f"n_seeds         : {n_seeds}")
    logging.info(
        f"n_submissions   : {n_seeds // n_stitches + bool(n_seeds % n_stitches)}"
    )
    logging.info("-" * 30)

    for seed in range(start_seed, start_seed + n_seeds, n_stitches):
        batch = list(range(seed, min(seed + n_stitches, start_seed + n_seeds)))
        logging.info(f"seeds: {batch}")
        random_bits_list = [rand.read_chunk(s) for s in batch]
        rand_gens = [rand.TrueRandom(random_bits) for random_bits in random_bits_list]

        # First do all of the randomness generation
        target_states_r = [rand_gen.normal(size=([2] * n)) for rand_gen in rand_gens]
        target_states_i = [rand_gen.normal(size=([2] * n)) for rand_gen in rand_gens]
        target_states = [
            (target_state_r + 1j * target_state_i) / 2 ** ((n + 1) / 2)
            for target_state_r, target_state_i in zip(target_states_r, target_states_i)
        ]

        ag_toggle_lists = [
            random_stabilizer_toggles_ag(n, rand_gen) for rand_gen in rand_gens
        ]
        reversed_ag_toggle_lists = [list(reversed(lst)) for lst in ag_toggle_lists]

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

        overall_start = time.time()
        with ProcessPoolExecutor() as executor:
            optimization_results = list(executor.map(run_optimization, jobs))
        total_time = time.time() - overall_start
        logging.info(f"🔁 Total parallel optimization time: {total_time:.2f} seconds")
        logging.info("")

        # Make sure we preserve order (this shouldn't be needed, but doing to be extra safe)
        optimization_results.sort(key=lambda r: r["i"])

        output_states = [r["output_state"] for r in optimization_results]
        opt_param_lists = [r["opt_params"] for r in optimization_results]
        noisy_fidelities = [r["noise_fidelity"] for r in optimization_results]

        for r in optimization_results:
            logging.info(f"Noiseless fidelity: {r['noiseless_fidelity']}")
            logging.info(f"Estimated fidelity due to noise: {r['noise_fidelity']}")
            logging.info(
                f"Estimated overall fidelity: {r['noise_fidelity'] * r['noiseless_fidelity']}"
            )
            logging.info(f"Optimization time: {r['time']}")
            logging.info("")

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
            for opt_params in opt_param_lists
        ]
        cliff_circs = [
            make_clifford_circuit(n, lst, backend) for lst in reversed_ag_toggle_lists
        ]
        scoring_states = [
            apply_clifford(target_state, lst)
            for target_state, lst in zip(target_states, reversed_ag_toggle_lists)
        ]
        cliff_output_states = [
            apply_clifford(output_state, lst)
            for output_state, lst in zip(output_states, reversed_ag_toggle_lists)
        ]

        basis_xebs = [
            sum(abs((scoring_state * cliff_output_state).flatten() ** 2)) * 2**n - 1
            for scoring_state, cliff_output_state in zip(
                scoring_states, cliff_output_states
            )
        ]
        logging.info(f"Noiseless basis XEB: {[float(xeb) for xeb in basis_xebs]}")
        logging.info(
            f"Est. noisy basis XEB: {[float(basis_xeb * nf) for basis_xeb, nf in zip(basis_xebs, noisy_fidelities)]}"
        )
        logging.info("")

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
                job_data = JobData(
                    n,
                    depth,
                    noisy,
                    device_name,
                    detect_leakage,
                    batch,
                    target_states,
                    reversed_ag_toggle_lists,
                    opt_param_lists,
                    overall_circ,
                    result_handle,
                )
                save_path = f"job_handles/{device_name}/n_{n}_depth_{depth}"
                pathlib.Path(save_path).mkdir(parents=True, exist_ok=True)
                job_data.save(filename=f"{save_path}/seeds_{batch[0]}-{batch[-1]}.txt")
