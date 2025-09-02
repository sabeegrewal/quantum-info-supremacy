# Quantum Information Supremacy  

This repository contains the code accompanying the paper  
**“Demonstrating an unconditional separation between quantum and classical information resources.”**

The data used for this project can be found in [quantum-info-supremacy-data](https://github.com/sabeegrewal/quantum-info-supremacy-data/).

Install requirements:
`pip install -r requirements.txt`

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

