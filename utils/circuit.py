from pytket import Circuit, Bit, Qubit
from pytket.extensions.quantinuum.backends.leakage_gadget import (
    get_leakage_gadget_circuit,
)
from pytket.passes import SequencePass

from qujax.statetensor import apply_gate
from qujax.gates import *

from utils.random_stabilizer import num_stab_gates_ag, stabilizer_gate_list_ag


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
        num_stab_gates // max_register_width + bool(num_stab_gates % max_register_width)
    ):
        cliff_circ.add_c_register(f"clifford{i}", max_register_width)

    # List of all control bits, in order
    bits = [
        Bit(f"clifford{i // max_register_width}", i % max_register_width)
        for i in range(num_stab_gates)
    ]
    # Set all of the classical toggles before doing any gates
    cliff_circ.add_c_setbits(reversed_ag_toggles, bits)
    # Add a barrier afterwards
    # Not sure whether this is necessary,
    # but I don't want the compiler to simplify the gates
    cliff_circ.add_barrier(cliff_circ.bits)

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
        num_stab_gates // max_register_width + bool(num_stab_gates % max_register_width)
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
                    Qubit(i),  # Circuit qubit
                    Qubit(n + (i % num_leakage_qubits)),  # Postselection qubit
                    Bit(f"leakage_detection{circ_idx}", i),  # Store result here
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
