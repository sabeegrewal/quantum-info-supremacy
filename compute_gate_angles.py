from utils.job_data import JobData
from utils.process_io import await_job, get_shots_and_xeb_scores
from utils.circuit import apply_clifford

from optimize.optimize_jax import zzphase_params

import time
import os
import numpy as np

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

save_path = f"job_handles/{device_name}/n_{n}_depth_{depth}/"
filenames = os.listdir(save_path)
txt_filenames = [fn for fn in filenames if fn[-4:] == ".txt"]
job_datas = [JobData.load(save_path + fn) for fn in txt_filenames]
job_datas.sort(key=lambda jd: jd.seeds)

print("Loaded")

# Now save the data
time_str = (
    time.asctime().replace("/", "_").replace(":", "-").replace(" ", "_")
)
filename = f"data/angles_{device_name}_{n}_{depth}_{time_str}.csv"
with open(filename, "w") as file:
    file.write("Seed,Index,Angle\n")

    all_zz_params = []
    for job_data in job_datas:
        for seed, all_params in zip(job_data.seeds, job_data.opt_param_lists):
            zz_params = zzphase_params(job_data.n, all_params)
            all_zz_params.extend(zz_params)
            for i in range(len(zz_params)):
                file.write(f"{seed},{i},{zz_params[i]}\n")

from matplotlib import pyplot as plt
plt.hist(all_zz_params)
plt.show()
