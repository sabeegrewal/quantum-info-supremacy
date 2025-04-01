from utils.job_data import JobData

from pytket.extensions.quantinuum import QuantinuumBackend, QuantinuumAPIOffline
from pytket.extensions.quantinuum.backends.api_wrappers import QuantinuumAPI
from pytket.extensions.quantinuum.backends.credential_storage import (
    QuantinuumConfigCredentialStorage,
)


import os

n = 12
depth = 86
device_name = "H1-1"

print("-" * 30)
print(f"n               : {n}")
print(f"depth           : {depth}")
print(f"device_name     : {device_name}")
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
save_path = f"job_handles/{device_name}/n_{n}_depth_{depth}/"
filenames = os.listdir(save_path)
txt_filenames = [fn for fn in filenames if fn[-4:] == ".txt"]
job_datas = [JobData.load(save_path + fn) for fn in txt_filenames]
job_datas.sort(key=lambda jd: jd.seeds)

for jd in job_datas:
    if backend.circuit_status(jd.result_handle).status.name == "QUEUED":
        print(jd.seeds)
        backend.cancel(jd.result_handle)
