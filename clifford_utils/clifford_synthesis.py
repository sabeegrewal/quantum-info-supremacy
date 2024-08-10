import numpy as np

from qiskit.quantum_info import random_clifford
from qiskit.synthesis import synth_clifford_layers
from qiskit.circuit import QuantumCircuit

# Gate layers returned by qiskit's synth_clifford_layers method
clifford_layer_names = ["s", "cz", "cx", "h", "s", "cz", "h", "Pauli"]

def full_layer_gates(n, name):
    """A list of gates such that any layer can be implemented
    by toggling a subset of the gates.

    Parameters
    ----------
    n : int
        Number of qubits in the circuit.
    name : str
        A layer ID from `clifford_layer_names`.

    Returns
    -------
    list[(str, tup[int])]
        The gate list for the given number of qubits and layer,
        where the first argument of each tuple is the gate name
        and the second argument is the qubits it applies to.
    """
    
    gates = []
    if name == "s":
        # s gate on each qubit
        for i in range(n):
            gates.append(("s", (i,)))
    elif name == "cz":
        # cz gate on each ordered pair of qubits
        for i in range(n):
            for j in range(i+1,n):
                gates.append(("cz", (i, j)))
    elif name == "cx":
        # cx gates are tricker
        # Here the gate ordering comes from Gaussian elimination
        for i in range(n):
            # First set entry [i][i] to 1
            for j in range(i+1, n):
                # XOR row j into row i if necessary
                gates.append(("cx", (j, i)))

            # Now zero out the rest of column i
            for j in range(n):
                if j != i:
                    # XOR row i into row j if necessary
                    gates.append(("cx", (i, j)))
    elif name == "h":
        # h gate on each qubit
        for i in range(n):
            gates.append(("h", (i,)))
    elif name == "Pauli":
        # Pauli gate on each qubit
        # TODO consider breaking y down into a product of x and z?
        for i in range(n):
            gates.append(("x", (i,)))
            gates.append(("y", (i,)))
            gates.append(("z", (i,)))
    else:
        # Not a valid layer name
        raise Exception("not a Clifford layer: " + name)
    
    return gates

def synth_clifford_toggle(cliff):
    """A list of gates to toggle so as to implement the desired Clifford.

    Parameters
    ----------
    cliff : qiskit Clifford
        A Clifford unitary.

    Returns
    -------
    list[bool]
        A list corresponding to the concatenated output of `full_layer_gates`
        over all `clifford_layer_names` such that toggling the corresponding
        gates implements `cliff`.
    """
    
    n = cliff.num_qubits
    # Use qiskit's layered Clifford synthesis, which returns a layered circuit
    # ordered according to clifford_layer_names
    circ_layers = synth_clifford_layers(cliff,
                                        cx_synth_func=synth_cx,
                                        cz_synth_func=synth_cz)

    result = []

    # Iterate through the layers
    for i in range(len(clifford_layer_names)):
        layer_name = clifford_layer_names[i]
        # All possible gates that could be toggle in this layer
        all_layer_gates = full_layer_gates(n, layer_name)
        # Instructions in this layer as computed by qiskit
        layer_instructions = circ_layers.data[i].operation.definition.data
        # Turn the instructions into (name, qubits) pairs
        applied_gates = list(map(lambda inst: (inst.operation.name,
                                                      tuple(map(lambda q: q._index, inst.qubits))),
                                        layer_instructions))
        # For whatever reason, qiskit doesn't sort the Hadamard layers correctly
        # Since the order doesn't matter, we can safely sort the operations
        if layer_name == "h":
            applied_gates.sort()

        # Index into applied_gates 
        instruction_idx = 0
        # Iterate throguh all possible gates and see which must be toggled on
        for gate in all_layer_gates:
            # Check whether we've iterated through all of the instructions
            if instruction_idx < len(applied_gates):
                inst_gate = applied_gates[instruction_idx]
            # If we reach the end of the list, do nothing
            else:
                inst_gate = None
            # If the applied gate matches, add it to the list and increment the counter
            if gate == inst_gate:
                result.append(True)
                instruction_idx += 1
            # Otherwise try the next gate
            else:
                result.append(False)
        # Checks that we have iterated through all gates in this layer
        assert instruction_idx == len(applied_gates)
    return result

def circuit_from_toggles(n, toggles):
    """Turn a list of toggles back into the corresponding Clifford circuit.

    Parameters
    ----------
    n : int
        Number of qubits in the circuit.
    toggles : list[bool]
        A list of toggles such as that output by `synth_clifford_toggle`.

    Returns
    -------
    QuantumCircuit
        A qiskit `QuantumCircuit` with only the toggled gates.
    """
    
    circ = QuantumCircuit(n)
    # The entire run of gates that could be toggled on
    full_gates = [gate for name in clifford_layer_names for gate in full_layer_gates(n, name)]
    # Each toggle should correspond to a single possible gate.
    assert len(toggles) == len(full_gates)
    for i in range(len(toggles)):
        if toggles[i]:
            # If the gate is toggled on, apply the corresponding gate in the circuit
            name, qubits = full_gates[i]
            if name == "s":
                circ.s(qubits[0])
            elif name == "cz":
                circ.cz(qubits[0], qubits[1])
            elif name == "cx":
                circ.cx(qubits[0], qubits[1])
            elif name == "h":
                circ.h(qubits[0])
            elif name == "x":
                circ.x(qubits[0])
            elif name == "y":
                circ.y(qubits[0])
            elif name == "z":
                circ.z(qubits[0])
            else:
                raise Exception("not a Clifford layer: " + name)
    return circ

########

def synth_cz(symmetric_mat):
    """Synthesize a circuit of controlled-Z gates.

    Parameters
    ----------
    symmetric_mat : list[list[int]]
        A symmetric 0/1 valued matrix, where entry (i, j)
        indicates whether to apply a CZ between qubits i and j.

    Returns
    -------
    QuantumCircuit
        A circuit of controlled-Z gates implementing the operator
        in a canonical form.
    """
    
    n = len(symmetric_mat)
    qc = QuantumCircuit(n, name="CZ")

    for i in range(n):
        for j in range(i+1,n):
            if symmetric_mat[i][j]:
                # Apply the gate
                qc.cz(i, j)
    return qc

def synth_cx(invertible_mat):
    """Synthesize a circuit of controlled-not gates.

    Parameters
    ----------
    invertible_mat : list[list[bool]]
        An invertible matrix over F_2.

    Returns
    -------
    QuantumCircuit
        A circuit of controlled-not gates implementing the operator
        in a canonical form, via Gaussian elimination.
    """
    
    mat = invertible_mat.copy()
    n = len(mat)
    qc = QuantumCircuit(n, name="CX")
    
    for i in range(n):
        # First set entry [i][i] to 1
        for j in range(i+1, n):
            # XOR row j into row i if necessary
            if mat[j][i] and not mat[i][i]:
                # Apply the gate
                qc.cx(j, i)
                # Update the row operation
                mat[i] = np.logical_xor(mat[i], mat[j])

        # Now zero out the rest of column i
        for j in range(n):
            if j != i:
                # XOR row i into row j if necessary
                if mat[j][i]:
                    # Apply the gate
                    qc.cx(i, j)
                    # Update the row operation
                    mat[j] = np.logical_xor(mat[i], mat[j])

    return qc.inverse()

########

# Now let's sanity check that the different Clifford implementations are all the same

num_qubits = 6
from qiskit.quantum_info import Operator
import time

start = time.time()
for i in range(100):
    #print(i)
    # Random clifford
    cliff = random_clifford(num_qubits)

    # First synthesize the Clifford directly using qiskit's default layers function
    circ1 = synth_clifford_layers(cliff)

    # Next synthesize the Clifford using qiskit's default layers function,
    # but with custom cx and cz synthesis
    circ2 = synth_clifford_layers(cliff,
                                 cx_synth_func=synth_cx,
                                 cz_synth_func=synth_cz)

    # Finally, synthesize the Clifford from the list of toggles
    toggles = synth_clifford_toggle(cliff)
    circ3 = circuit_from_toggles(num_qubits, toggles)

    # Now check that the operators are all equal
    # This is the slow part...
    circ_op_1 = Operator.from_circuit(circ1)
    circ_op_2 = Operator.from_circuit(circ2)
    circ_op_3 = Operator.from_circuit(circ2)
    assert circ_op_1 == circ_op_2
    assert circ_op_1 == circ_op_3
    assert circ_op_2 == circ_op_3

print(time.time() - start)
