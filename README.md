# Quantum Information Supremacy  

This repository contains the code accompanying the paper  
**“Demonstrating an unconditional separation between quantum and classical information resources.”**

The data used for this project can be found in [quantum-info-supremacy-data](https://github.com/sabeegrewal/quantum-info-supremacy-data/).

## Project Organization

------------
    ├── ansatz        <- Code for variationally training Alice's state preparation circuit.  
    ├── figures       <- Scripts to generate the figures appearing in the manuscript.  
    ├── optimize      <- Code to optimize the parameters in Alice's parameterized state preparation circuit.  
    └── utils         <- Utility functions (e.g., generating random stabilizer states, preparing Bob's Clifford 
                         measurement, processing job data).  
    main.py           <- Implements the main execution loop of the program. Specifically, it: (i) Reads in a Haar-random
                         target state; (ii) Trains a parameterized circuit to approximate this state. (iii) Generates a 
                         random Clifford measurement. (iv) Executes the circuit and records the results.
------------

## Installation

This project requires **Python 3**. To install the necessary dependencies, run:

```bash
pip install -r requirements.txt
```

Installation should complete within a few minutes.


## Demo 

It is possible to run a toy example of our experiment on a local emulator. The expected runtime of the demo is between 1 and 10 minutes, depending on your computer's processor. This demo runs in around 80 seconds on a 2022 M2 Macbook Air. The demo has been tested on Python 3.12.3.

To try it out, follow these steps:

1. **Clone this repository.**
   Follow the [installation instructions](#installation) to set up the environment. 

3. **Download randomness data.**  
   Obtain [this file](https://github.com/sabeegrewal/quantum-info-supremacy-data/blob/main/randomness/ANU_13Oct2017_100MB_1), which contains 100 MB of true randomness used to generate Alice’s input state and Bob’s Clifford measurement. Place the file in `./randomness/`.

4. **Check settings in `main.py`.**
   In `main.py`, ensure that lines 67–78 match the following:
   ```python
    n = 12
    depth = 86
    noisy = True
    device_name = "H1-1LE" # Change to H1-1 for actual experiment
    detect_leakage = False

    submit_job = True
    n_stitches = 2
    n_parallel = 8
    n_shots = 1
    start_seed = 0
    n_seeds = 2 # Change to 10000 for actual experiment
    ```

6. **Run the demo.**  
   Execute:
   ```bash
   python main.py
   ```

This script will:

* Load the randomness from Step 2 and use it to generate Alice’s state and Bob’s measurement.

* Train a parameterized circuit to approximate Alice’s Haar-random state using variational optimization. This step can take between 1 and 10 minutes to complete. 

* Construct the overall circuit by combining Alice’s state preparation with Bob’s measurement.

* Repeat the above and stitch the circuits together into a single job (see [Circuit Stitching](https://docs.quantinuum.com/systems/trainings/knowledge_articles/circuit_stitching.html)).

* Run the resulting circuit on Quantinuum’s local emulator, and compute the XEB scores from the two instances. 

* Log the experimental data in the `/logs/` directory. The expected output of this demo can be found (here)[https://github.com/sabeegrewal/quantum-info-supremacy-data/blob/main/logs/H1-1LE/n_12_depth_86/seeds_0-1_2025-09-15_13-18-51.txt].

The demo is a lightweight illustration of our full protocol. It does not require access to Quantinuum hardware, but faithfully reproduces the logic of the experiment. For those interested in larger-scale reproductions, all of the true randomness used in our experiments is available [here](https://github.com/sabeegrewal/quantum-info-supremacy-data/blob/main/randomness/).

## Generating Figures

The code used to generate the figures in our manuscript is located in the `/figures/` directory.  
- The scripts `classical_bounds.py`, `ensemble_comparison.py`, and `noiseless_bound.py` can be run as standalone programs, provided the dependencies in `requirements.txt` have been installed.  
- The remaining scripts require external data files available in the companion repository [quantum-info-supremacy-data](https://github.com/sabeegrewal/quantum-info-supremacy-data/).  

In particular:  
- To run `achieved_xeb.py`, first download [this CSV file](https://github.com/sabeegrewal/quantum-info-supremacy-data/blob/main/data/shots_H1-1_12_86_Thu_Jun_12_17-06-28_2025.csv).  
- To run `angle_distribution.py`, first download [this NumPy file](https://github.com/sabeegrewal/quantum-info-supremacy-data/blob/main/data/angles_H1-1_12_86_Tue_Aug_26_12-56-25_2025.npy).  
