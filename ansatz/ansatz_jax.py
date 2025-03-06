import jax
import jax.numpy as jnp

# ----------------
# Gates and states
# ----------------

def state2(theta, phi):
    """Arbitrary one-qubit state with two real parameters.

    Parameters
    ----------
    theta : real
    phi  : real

    Returns
    -------
    jax array
        The one-qubit state equivalent to U3(theta, phi, _)|0> as a jax array of shape `(2)`.
    """

    # Matching the convention of Quantinuum: https://tket.quantinuum.com/api-docs/optype.html
    pitheta_2 = (jnp.pi / 2) * theta
    cpt2 = jnp.cos(pitheta_2)
    spt2 = jnp.sin(pitheta_2)
    eipp = jnp.exp((jnp.pi * 1j) * phi)
    return jnp.array([cpt2, eipp * spt2])

def u3(theta, phi, lamda):
    """Arbitrary one-qubit gate with three real parameters.

    Parameters
    ----------
    theta : real
    phi  : real
    lamda : real

    Returns
    -------
    jax array
        The U3 gate parametrized by (theta, phi, lamda) as a jax array of shape (2, 2).
    """

    pitheta_2 = (jnp.pi/2) * theta
    cpt2 = jnp.cos(pitheta_2)
    spt2 = jnp.sin(pitheta_2)
    eipl = jnp.exp((jnp.pi * 1j) * lamda)
    eipp = jnp.exp((jnp.pi * 1j) * phi)
    # See https://docs.quantinuum.com/tket/api-docs/optype.html
    return jnp.array([[cpt2, -eipl * spt2], [eipp * spt2, eipp * eipl * cpt2]])


def rzz(theta):
    """Two-qubit RZZ gate.

    Parameters
    ----------
    theta : real
        Angle of rotation.

    Returns
    -------
    jax array
        The RZZ gate parametrized by theta as a jax array of shape (4, 4).
    """

    eipt2 = jnp.exp((-jnp.pi*0.5j) * theta)
    eipt2c = eipt2.conj()
    # See https://docs.quantinuum.com/tket/api-docs/optype.html
    return jnp.diag(jnp.array([eipt2, eipt2c, eipt2c, eipt2]))



# -------------------
# Circuit application
# -------------------

def product_state(params):
    """Arbitrary product state with given parameters.

    Parameters
    ----------
    params : jax array
        Should have shape `(2 * n)` for some `n`.

    Returns
    -------
    jax array
        The corresponding product state as a jax array of shape `[2] * n`.
    """

    # Infer the first dimension, which is the number of qubits
    params = params.reshape(-1, 2)
    result = 1
    for theta, phi in params:
        result = jnp.tensordot(result, state2(theta, phi), axes=[[], []])
    return result

def apply_two_qubit(state, gate, i, j):
    """Apply a 2-qubit gate.

    Parameters
    ----------
    state : jax array
        Should have shape `[2] * n` for some `n > max(i, j)`.
    gate : jax array
        A 2-qubit gate as a jax array of shape (2, 2, 2, 2).
    i : int
        First index at which to apply the gate.`
    j : int
        Second index at which to apply the gate.
    Returns
    -------
    jax array
        The state with the given gate gate applied at indices (i, j).
    """

    # Names for indices into the state tensor
    indices = "abcdefghijklmnopqrstuvwxyz"[:len(state.shape)]
    # Replace names at locations i and j with "y" and "z", respectively
    new_indices = indices[:i] + "y" + indices[i+1:]
    new_indices = new_indices[:j] + "z" + new_indices[j+1:]
    # Contract indices[i] and indices[j], replacing with "y" and "z", respectively
    einsum_str = indices + ",yz" + indices[i] + indices[j] + "->" + new_indices
    return jnp.einsum(einsum_str, state, gate)


def brickwork_pairs(num_qubits, layer):
    """Which qubits to pair in a 1D cyclic brickwork circuit.

    Parameters
    ----------
    num_qubits : int
        Number of qubits in the circuit.
    layer : int
        Layer at which gates are being applied.

    Returns
    -------
    list[(int, int)]
        The list of qubits paired at this layer.
    """
    return [
        ((layer + i) % num_qubits, (layer + i + 1) % num_qubits)
        for i in range(0, num_qubits - 1, 2)
    ]

def apply_circuit(depth, initial_state, params):
    """Apply a 1D ansatz circuit with given parameters to the initial state.
    This is for a 1D cyclic brickwork architecture with alternating U3 and RZZ gates.

    Parameters
    ----------
    depth : int
        Depth of the ansatz circuit, as measured by number of two-qubit layers.
    initial_state : jax array
        Quantum state to apply the circuit to.
        Should have shape `[2] * n` for some `n`.
    params : jax array
        List of real parameters that define the ansatz circuit.
        Should have length `(n // 2) * depth * 7`.

    Returns
    -------
    jax array
        The output state as an array of shape `[2] * n`.
    """

    # Infer the number of qubits
    num_qubits = len(initial_state.shape)
    
    # Number of parameters in RZZ gates
    num_rzz_params = (num_qubits // 2) * depth

    # Offsets into the parameter array
    rzz_off = 0
    u3_off = num_rzz_params

    state = initial_state
    for layer in range(depth):
        # First identify the paired qubits
        pairs = brickwork_pairs(num_qubits, layer)

        # Apply 2- and 1-qubit gates to each pair
        for i, j in pairs:
            # RZZ gate to apply
            rzz_gate = rzz(params[rzz_off])
            rzz_off += 1
            # U3 gates to apply
            u3_gate_i = u3(params[u3_off],   params[u3_off+1], params[u3_off+2])
            u3_gate_j = u3(params[u3_off+3], params[u3_off+4], params[u3_off+5])
            u3_off += 6

            # Multiply the 2- and 1-qubit gates together first
            # This reduces the total number of tensor contractions by a factor of 3
            u3_gates = jnp.kron(u3_gate_i, u3_gate_j)
            gate_matrix = jnp.matmul(u3_gates, rzz_gate)
            gate_tensor = gate_matrix.reshape([2,2,2,2])

            # Apply the combined gate
            state = apply_two_qubit(state, gate_tensor, i, j)
            
    return state

def zzphase_fidelity(theta):
    """Approximate fidelity with which we can implement a ZZ gate with given parameter.

    Parameters
    ----------
    theta : real or array
        If given an array of thetas, this method returns the array of corresponding fidelities.

    Returns
    -------
    real or array
        The fidelity (or fidelities) corresponding to theta.
    """

    # In qujax, the maximally entangling ZZ gate has theta = 0.5;
    # See https://cqcl.github.io/qujax/gates.html

    # First normalize theta to [0, 1), which is equivalent to applying Z gates after if necessary
    theta = jnp.mod(theta, 1)
    # Then normalize to [0, 1/2), which is equivalent to conjugating one qubit by X if necessary
    theta = 1 / 2 - jnp.abs(theta - 1 / 2)
    # Estimate of 2-qubit gate error rate
    eps_tq = 0.00148 * theta + 0.00027
    # Estimate of 1-qubit memory error rate
    eps_mem = 8e-5
    # Finally compute a linear function
    # The reason for the 5/4 and 3 is a (d+1)/d in going from average fidelity to process fidelity
    # For 2-qubit gates d = 4, for 1-qubit gates d = 2
    # Finally, we account for eps_mem twice, making 3/2 -> 3
    return 1 - (5/4) * eps_tq - 3 * eps_mem
