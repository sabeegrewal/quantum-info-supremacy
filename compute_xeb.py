from utils.job_data import JobData
from utils.process_io import await_job, get_xeb_scores
from utils.circuit import apply_clifford

from pytket.extensions.quantinuum import QuantinuumBackend, QuantinuumAPIOffline
from pytket.extensions.quantinuum.backends.api_wrappers import QuantinuumAPI
from pytket.extensions.quantinuum.backends.credential_storage import (
    QuantinuumConfigCredentialStorage,
)

import numpy as np

n = 12
depth = 86
device_name = "H1-1"
n_stitches = 5

start_seed = 0
n_seeds = 1600

print("-" * 30)
print(f"n               : {n}")
print(f"depth           : {depth}")
print(f"device_name     : {device_name}")
print(f"n_stitches      : {n_stitches}")
print("")
print(f"start_seed      : {start_seed}")
print(f"n_seeds         : {n_seeds}")
print(f"n_submissions   : {n_seeds // n_stitches + bool(n_seeds % n_stitches)}")
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
save_path = f"job_handles/{device_name}/n_{n}_depth_{depth}"

all_scores = []
for seed in range(start_seed, start_seed + n_seeds, n_stitches):
    batch = list(range(seed, min(seed + n_stitches, start_seed + n_seeds)))
    job_data = JobData.load(f"{save_path}/seeds_{batch[0]}-{batch[-1]}.txt")

    scoring_states = [
        apply_clifford(target_state, lst)
        for target_state, lst in zip(
            job_data.target_states, job_data.reversed_ag_toggle_lists
        )
    ]

    result, job_status = backend.get_partial_result(job_data.result_handle)
    scores = get_xeb_scores(scoring_states, job_data.detect_leakage, result)
    all_scores.extend([x for lst in scores for x in lst])

print(f"Number of shots: {len(all_scores)}")
print(f"Average XEB: {np.mean(all_scores)}")
print(f"XEB std: {np.std(all_scores)}")
