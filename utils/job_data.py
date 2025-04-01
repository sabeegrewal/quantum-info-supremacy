from pytket import Circuit

from pytket.backends.resulthandle import ResultHandle

import time
import json
import numpy as np
import ast


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
        result_handle,
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
            time_str = (
                time.asctime().replace("/", "_").replace(":", "-").replace(" ", "_")
            )
            filename = f"job_handles/{self.n}_{self.depth}_{time_str}.txt"

        with open(filename, "w") as file:
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

    @staticmethod
    def load(filename):
        with open(filename, "r") as file:
            n = int(file.readline().strip())
            depth = int(file.readline().strip())
            noisy = file.readline().strip() == "True"
            device_name = file.readline().strip()
            detect_leakage = file.readline().strip() == "True"
            seeds = ast.literal_eval(file.readline().strip())
            target_states = np.array(ast.literal_eval(file.readline().strip()))
            # In some previous versions of the code, numpy and python bools printed differently
            # E.g. True would be printed as "np.True_"
            # The purpose of this code is to allow parsing of either format for the toggles
            untrimmed_lit = file.readline().strip()
            trimmed_lit = untrimmed_lit.replace("np.", "").replace("_", "")
            reversed_ag_toggle_lists = ast.literal_eval(trimmed_lit)
            opt_param_lists = np.array(ast.literal_eval(file.readline().strip()))
            overall_circ = Circuit.from_dict(json.loads(file.readline().strip()))
            result_handle = ResultHandle.from_str(file.readline().strip())

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
