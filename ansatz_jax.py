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

def apply_one_qubit_layer(one_qubit_params, state, indices):
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

def apply_two_qubit_layer(two_qubit_params, state, indices):
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
        

def num_ansatz_params_per_iter(num_qubits, depth):
    """Numper of continuous parameters in one iteration of the ansatz circuit. This is for a
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

    return (num_qubits // 2) * depth * 7
##    num_2_per_layer = num_qubits // 2
##    num_1 = 2 * num_2_per_layer * depth
##    num_2 = num_2_per_layer * depth
##    return num_2 + 3 * num_1

def apply_ansatz_circuit(params, state, depth):
    """Apply a 1D ansatz circuit with given parameters.
    This is for a 1D cyclic brickwork architecture with alternating U3 and RZZ gates.

    Parameters
    ----------
    params : jax array
        Parameters of the gates. Should have length `num_ansatz_params_per_iter(n, depth)`.
    state : jax array
        State to apply the circuit to. Should have shape `[2] * n` for some `n`.
    depth : int
        Depth of the ansatz circuit, as measured by number of two-qubit layers.

    Returns
    -------
    jax array
        The state with the parametrized ansatz circuit applied.
    """

    # In principle, depth could be inferred from params,
    # just like we infer the number of qubits from the state below
    num_qubits = len(state.shape)
    num_2_per_layer = num_qubits // 2
    num_2 = num_2_per_layer * depth
    # Split the parameter array into the RZZ and U3 parameters
    rzz_params = params[:num_2]
    u3_params = params[num_2:]

    for layer in range(depth):
        # First identify the paired qubits
        pairs = brickwork_pairs(num_qubits, layer)
        
        # Apply an RZZ layer
        rzz_off = layer*num_2_per_layer
        state = apply_two_qubit_layer(rzz_params[rzz_off:rzz_off+num_2_per_layer], state, pairs)

        # Apply a U3 layer, but only to the qubits that were paired
        indices = list(sum(pairs, ())) # This flattens the list of paired indices
        u3_off = 6*layer*num_2_per_layer
        state = apply_one_qubit_layer(u3_params[u3_off:u3_off+6*num_2_per_layer], state, indices)
    
    return state

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
    
    theta_2 = theta / 2
    ct2 = jnp.cos(theta / 2)
    st2 = jnp.sin(theta / 2)
    eip = jnp.exp(1j * phi)
    return jnp.array([ct2, eip * st2])

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
        result = jnp.tensordot(result, state2(theta, phi), axes=[[],[]])
    return result

def iterated_ansatz_state(params, num_qubits, depth):
    """Compute the output state of an iterated 1D ansatz circuit with given parameters.
    This is for a 1D cyclic brickwork architecture with alternating U3 and RZZ gates,
    where the initial state is a product state.

    Parameters
    ----------
    params : jax array
        Parameters of the gates. Should have length `num_total_ansatz_params(num_qubits, depth)`.
    num_qubits : int
        Number of qubits.
    depth : int
        Depth of the ansatz circuit iterations, as measured by number of two-qubit layers.

    Returns
    -------
    jax array
        The output state of the parametrized ansatz circuit..
    """
    
    product_params = params[:2*num_qubits]
    ansatz_params = params[2*num_qubits:]

    initial_state = product_state(product_params)
    
    # A trick borrowed from https://github.com/CQCL/qujax/blob/main/qujax/utils.py#L496
    # This speeds up compilation by iterating the ansatz circuit many times
    def apply_once(state, iter_params):
        return apply_ansatz_circuit(iter_params, state, depth), None
    
    # Infer the number of repetitions in the first dimension
    reshaped_ansatz_params = ansatz_params.reshape(-1, num_ansatz_params_per_iter(num_qubits, depth))
    result, _ = jax.lax.scan(apply_once, initial_state, reshaped_ansatz_params)
    return result

def num_total_ansatz_params(num_qubits, depth, iters):
    """Numper of continuous parameters in an iterated ansatz circuit. This is for a
    1D cyclic brickwork architecture with alternating U3 and RZZ gates, where the initial
    state is a product state.

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

    # 2 * num_qubits is for the product state
    # Rest is for the iterated ansatz circuit
    return 2 * num_qubits + iters * num_ansatz_params_per_iter(num_qubits, depth)

def ansatz_circ_quimb(params, num_qubits, depth):
    """Compute the 1D ansatz circuit with given parameters, using quimb.
    This is for a 1D cyclic brickwork architecture with alternating U3 and RZZ gates.
    Useful as a sanity check for `apply_ansatz_circuit`.

    Parameters
    ----------
    params : jax array
        Parameters of the gates. Should have length `num_ansatz_params_per_iter(num_qubits, depth)`.
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

##    # Start with a one-qubit layer on all qubits
##    for i in range(num_qubits):
##        theta, phi, lamda = u3_params[u3_off:u3_off+3]
##        u3_off += 3
##        circ.apply_gate("U3", theta, phi, lamda, i,
##                        gate_round=0, parametrize=True)
    
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

