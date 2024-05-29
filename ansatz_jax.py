import jax
import jax.numpy as jnp

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
    
    theta_2 = theta / 2
    ct2 = jnp.cos(theta / 2)
    st2 = jnp.sin(theta / 2)
    eil = jnp.exp(1j * lamda)
    eip = jnp.exp(1j * phi)
    # See https://docs.quantum.ibm.com/api/qiskit/qiskit.circuit.library.U3Gate
    return jnp.array([[ct2,       -eil * st2     ],
                      [eip * st2, eip * eil * ct2]])

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
    
    eit2 = jnp.exp(-1j * theta / 2)
    eit2c = eit2.conj()
    # See https://docs.quantum.ibm.com/api/qiskit/qiskit.circuit.library.RZZGate
    return jnp.diag(jnp.array([eit2, eit2c, eit2c, eit2]))

def apply_u3(state, i, theta, phi, lamda):
    """Apply a single U3 gate.

    Parameters
    ----------
    state : jax array
        Should have shape `[2] * n` for some `n > i`.
    i : int
        Index at which to apply the gate.
    theta : real
    phi  : real
    lamda : real

    Returns
    -------
    jax array
        The state with the U3(theta, phi, lamda) gate applied at index i.
    """
    
    U = u3(theta, phi, lamda)
    # This will roll the index i to the end
    state = jnp.tensordot(state, U, axes=([i], [1]))
    # Now we need to roll the last index back to i
    state = jnp.rollaxis(state, -1, i)
    return state

def apply_rzz(state, i, j, theta):
    """Apply a single RZZ gate.

    Parameters
    ----------
    state : jax array
        Should have shape `[2] * n` for some `n > j`.
    i : int
        First index at which to apply the gate.`
    j : int
        Second index at which to apply the gate.
    theta : real

    Returns
    -------
    jax array
        The state with the RZZ(theta) gate applied at indices (i, j).
    """
    
    U = rzz(theta).reshape([2, 2, 2, 2])
    # Set i to the smaller and j to the larger index,
    # so that index rolling works as intended
    i, j = sorted((i, j))
    # This will roll the indices i, j to the end
    state = jnp.tensordot(state, U, axes=([i, j], [2, 3]))
    # Now we need to roll the last two indices back to i, j
    state = jnp.rollaxis(state, -2, i)
    state = jnp.rollaxis(state, -1, j)
    return state

def one_qubit_layer(one_qubit_params, state, indices):
    """Apply a layer of U3 gates.

    Parameters
    ----------
    one_qubit_params : jax array
        Parameters of the gates. Should have length `3*len(indices)`.
    state : jax array
        Should have shape `[2] * n` for some `n > max(indices)`.
    indices : list[int]
        Indices at which to apply gates.

    Returns
    -------
    jax array
        The state with the corresponding U3 gates applied.
    """

    for idx in range(len(indices)):
        theta, phi, lamda = one_qubit_params[3*idx:3*idx+3]
        i = indices[idx]
        state = apply_u3(state, i, theta, phi, lamda)
    return state

def two_qubit_layer(two_qubit_params, state, indices):
    """Apply a layer of RZZ gates.

    Parameters
    ----------
    two_qubit_params : jax array
        Parameters of the gates. Should have length `len(indices)`.
    state : jax array
        Should have shape `[2] * n` for some `n > max(max(pair) for pair in indices)`.
    indices : list[(int, int)]
        Pairs of indices at which to apply gates.

    Returns
    -------
    jax array
        The state with the corresponding RZZ gates applied.
    """

    for idx in range(len(indices)):
        theta = two_qubit_params[idx]
        i, j = indices[idx]
        state = apply_rzz(state, i, j, theta)
    return state

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
    return [((layer+i) % num_qubits, (layer+i+1) % num_qubits) for i in range(0, num_qubits-1, 2)]
        

def num_ansatz_params(num_qubits, depth):
    """Numper of continuous parameters in an ansatz circuit. This is for a
    1D cyclic brickwork architecture with alternating U3 and RZZ gates.

    Parameters
    ----------
    num_qubits : int
        Number of qubits in the circuit.
    depth : int
        Depth of the circuit, as measured by number of two-qubit layers.

    Returns
    -------
    int
        The number of parameters needed to build such an ansatz circuit.
    """
    
    num_2_per_layer = num_qubits // 2
    num_1 = num_qubits + 2 * num_2_per_layer * depth
    num_2 = num_2_per_layer * depth
    return num_2 + 3 * num_1

def ansatz_state(params, num_qubits, depth):
    """Compute the output state of a 1D ansatz circuit with given parameters.
    This is for a 1D cyclic brickwork architecture with alternating U3 and RZZ gates.
    The circuit is initialized to the all zero state.

    Parameters
    ----------
    params : jax array
        Parameters of the gates. Should have length `num_ansatz_params(num_qubits, depth)`.
    num_qubits : int
        Number of qubits in the circuit and its output state.
    depth : int
        Depth of the ansatz circuit, as measured by number of two-qubit layers.

    Returns
    -------
    jax array
        The state with the parametrized ansatz circuit applied, as an array of shape `[2] * num_qubits`.
    """

    # Initialize the all-zero state
    state = jnp.zeros(2**num_qubits)
    state = state.at[0].set(1)
    state = state.reshape([2] * num_qubits)

    num_2_per_layer = num_qubits // 2
    num_2 = num_2_per_layer * depth
    # Split the parameter array into the RZZ and U3 parameters
    rzz_params = params[:num_2]
    u3_params = params[num_2:]

    # Start with a one-qubit layer on all qubits
    state = one_qubit_layer(u3_params[:3*num_qubits], state, range(num_qubits))

    for layer in range(depth):
        # First identify the paired qubits
        pairs = brickwork_pairs(num_qubits, layer)
        
        # Apply an RZZ layer
        rzz_off = layer*num_2_per_layer
        state = two_qubit_layer(rzz_params[rzz_off:rzz_off+num_2_per_layer], state, pairs)

        # Apply a U3 layer, but only to the qubits that were paired
        indices = list(sum(pairs, ())) # This flattens the list of paired indices
        u3_off = (3*num_qubits) + (6*layer*num_2_per_layer)
        state = one_qubit_layer(u3_params[u3_off:u3_off+6*num_2_per_layer], state, indices)
    
    return state

def ansatz_circ_quimb(params, num_qubits, depth):
    """Compute the 1D ansatz circuit with given parameters, using quimb.
    This is for a 1D cyclic brickwork architecture with alternating U3 and RZZ gates.
    The circuit is initialized to the all zero state.
    Useful as a sanity check for `ansatz_state`.

    Parameters
    ----------
    params : jax array
        Parameters of the gates. Should have length `num_ansatz_params(num_qubits, depth)`.
    num_qubits : int
        Number of qubits in the circuit and its output state.
    depth : int
        Depth of the ansatz circuit, as measured by number of two-qubit layers.

    Returns
    -------
    quimb Circuit
        The parametrized ansatz circuit.
    """
    import quimb as qu
    import quimb.tensor as qtn
    circ = qtn.Circuit(num_qubits)

    num_2_per_layer = num_qubits // 2
    num_2 = num_2_per_layer * depth
    # Split the parameter array into the RZZ and U3 parameters
    rzz_params = params[:num_2]
    u3_params = params[num_2:]

    # Current offset into the parameter arrays
    u3_off = 0
    rzz_off = 0

    # Start with a one-qubit layer on all qubits
    for i in range(num_qubits):
        theta, phi, lamda = u3_params[u3_off:u3_off+3]
        u3_off += 3
        circ.apply_gate("U3", theta, phi, lamda, i,
                        gate_round=0, parametrize=True)
    
    for layer in range(depth):
        # First identify the paired qubits
        pairs = brickwork_pairs(num_qubits, layer)
        
        # Apply an RZZ layer
        for i, j in pairs:
            theta = rzz_params[rzz_off]
            rzz_off += 1
            circ.apply_gate("RZZ", theta, i, j,
                            gate_round=layer, parametrize=True)

        # Apply a U3 layer, but only to the qubits that were paired
        indices = list(sum(pairs, ())) # This flattens the list of paired indices
        for i in indices:
            theta, phi, lamda = u3_params[u3_off:u3_off+3]
            u3_off += 3
            circ.apply_gate("U3", theta, phi, lamda, i,
                            gate_round=layer+1, parametrize=True)

    return circ

