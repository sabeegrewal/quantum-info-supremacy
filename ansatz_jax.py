import jax
import jax.numpy as jnp

from functools import partial

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
        The list of qubits paired at this layer, sorted.
    """
    return sorted([((layer+i) % num_qubits, (layer+i+1) % num_qubits) for i in range(0, num_qubits-1, 2)])

# Does this jit decorator do anything?
@partial(jax.jit, static_argnums=[2, 3])
def apply_brickwork_layer(params, state, num_qubits, layer):
    """Apply a layer of RZZ gates followed by a layer of U3 gates in a brickwork fashion.

    Parameters
    ----------
    params : jax array
        Parameters of the gates. Should have length `(num_qubits // 2) * 7`.
    state : jax array
        Should have shape `[2] * num_qubits`.
    num_qubits : int
        Number of qubits in the circuit.
    layer : int
        Layer at which gates are being applied.

    Returns
    -------
    jax array
        The state with the corresponding RZZ and U3 gates applied.
    """

    # First identify the paired qubits
    pairs = brickwork_pairs(num_qubits, layer)

    # Extract the parameters used for this layer
    rzz_params = params[:num_qubits // 2]
    u3_params = params[num_qubits // 2:]
    
    # Apply an RZZ layer
    state = apply_two_qubit_layer(rzz_params, state, pairs)

    # Apply a U3 layer, but only to the qubits that were paired
    indices = list(sum(pairs, ())) # This flattens the list of paired indices
    state = apply_one_qubit_layer(u3_params, state, indices)
    
    return state

def apply_ansatz_circuit(params, state, num_qubits, depth):
    """Apply a 1D ansatz circuit with given parameters.
    This is for a 1D cyclic brickwork architecture with alternating RZZ and U3 gates.

    Parameters
    ----------
    params : jax array
        Parameters of the gates. Should have length `depth * (num_qubits // 2) * 7`.
    state : jax array
        State to apply the circuit to. Should have shape `[2] * num_qubits`.
    num_qubits : int
        Number of qubits.
    depth : int
        Depth of the ansatz circuit, as measured by number of two-qubit layers.

    Returns
    -------
    jax array
        The state with the parametrized ansatz circuit applied.
    """

    # Reshape the parameter array into the parameters used at each layer.
    # There are (num_qubits // 2) ZZ gates at each layer, and each
    # combination of 1 ZZ + 2 U3 gates uses 1 + 2 * 3 = 7 parameters.
    params_reshaped = params.reshape(depth, (num_qubits // 2) * 7)

    if num_qubits % 2 == 0:
        # For even n, the brickwork pattern repeats every 2 layers
        depth_modulus = 2
    else:
        # For odd n, the brickwork pattern repeats every n layers
        depth_modulus = num_qubits

    # A trick borrowed from https://github.com/CQCL/qujax/blob/main/qujax/utils.py#L496
    # This speeds up compilation by iterating the ansatz circuit layers many times
    def apply_once(iter_state_and_layer, iter_params):
        iter_state, layer = iter_state_and_layer
        next_layer = (layer+1) % depth_modulus
        return (apply_brickwork_layer(iter_params, iter_state, num_qubits, layer), next_layer), None
    state, _ = jax.lax.scan(apply_once, (state, 0), reshaped_ansatz_params)

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

def ansatz_state(params, num_qubits, depth):
    """Compute the output state of an iterated 1D ansatz circuit with given parameters.
    This is for a 1D cyclic brickwork architecture with alternating U3 and RZZ gates,
    where the initial state is a product state.

    Parameters
    ----------
    params : jax array
        Parameters of the gates. Should have length `2 * num_qubits + depth * (num_qubits // 2) * 7`.
    num_qubits : int
        Number of qubits.
    depth : int
        Depth of the ansatz circuit, as measured by number of two-qubit layers.

    Returns
    -------
    jax array
        The output state of the parametrized ansatz circuit..
    """
    
    product_params = params[:2*num_qubits]
    circuit_params = params[2*num_qubits:]

    initial_state = product_state(product_params)
    
    return apply_ansatz_circuit(circuit_params, initial_state, num_qubits, depth)

def ansatz_circ_quimb(params, num_qubits, depth):
    """Compute the 1D ansatz circuit with given parameters, using quimb.
    This is for a 1D cyclic brickwork architecture with alternating U3 and RZZ gates.
    Useful as a sanity check for `apply_ansatz_circuit`.

    Parameters
    ----------
    params : jax array
        Parameters of the gates. Should have length `depth * (num_qubits // 2) * 7`.
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

    # Current offset into the parameter array
    off = 0
    
    for layer in range(depth):
        # First identify the paired qubits
        pairs = brickwork_pairs(num_qubits, layer)
        
        # Apply an RZZ layer
        for i, j in pairs:
            theta = params[off]
            off += 1
            circ.apply_gate("RZZ", theta, i, j,
                            gate_round=layer, parametrize=True)

        # Apply a U3 layer, but only to the qubits that were paired
        indices = list(sum(pairs, ())) # This flattens the list of paired indices
        for i in indices:
            theta, phi, lamda = params[off:off+3]
            off += 3
            circ.apply_gate("U3", theta, phi, lamda, i,
                            gate_round=layer+1, parametrize=True)

    return circ

