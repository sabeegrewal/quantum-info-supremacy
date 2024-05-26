import jax
import jax.numpy as jnp

def identity(num_qubits):
    """The identity on a specified number of qubits.

    Parameters
    ----------
    num_qubits : int
        The number of qubits.

    Returns
    -------
    jax array
        The identity matrix of dimension `2**num_qubits`  as a 2D jax array.
    """
    
    return jnp.identity(2**num_qubits)

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

def one_qubit_layer(one_qubit_params, state, num_qubits):
    """Apply a layer of U3 gates.

    Parameters
    ----------
    one_qubit_params : jax array
        Parameters of the gates. Should have length `3*num_qubits`.
    state : jax array
        Should have shape `[2] * num_qubits`.
    num_qubits : int
        Number of qubits in the state.

    Returns
    -------
    jax array
        The state with the corresponding U3 gates applied.
    """
    
    for i in range(num_qubits):
        theta, phi, lamda = one_qubit_params[3*i:3*i+3]
        U = u3(theta, phi, lamda)
        # Notice: we always apply U to the first index of the tensor.
        # This works because tensordot sends the resulting indices
        # to the back. So, at the end of the loop, we will have applied
        # one gate to each index, and the indices will be in their
        # original order again.
        state = jnp.tensordot(state, U, axes=([0], [1]))
    return state

def two_qubit_layer(two_qubit_params, state, num_qubits, evenodd):
    """Apply a layer of RZZ gates in a 1D brickwork fashion.

    Parameters
    ----------
    two_qubit_params : jax array
        Parameters of the gates. Should have length `(num_qubits - evenodd) // 2`.
    state : jax array
        Should have shape `[2] * num_qubits`.
    num_qubits : int
        Number of qubits in the state.
    evenodd : int
        0 of the layer is at even depth, 1 if the layer is at odd depth.

    Returns
    -------
    jax array
        The state with the corresponding RZZ gates applied.
    """

    # When n is even and the layer is even, we apply n/2 gates
    # When n is even and the layer is odd, we apply (n/2)-1 gates
    # When n is odd, we apply (n-1)/2 gates
    # The formula below evaluates this succinctly
    for i in range((num_qubits - evenodd) // 2):
        theta = two_qubit_params[i]
        # Reshape into the appropriate tensor
        U = rzz(theta).reshape([2, 2, 2, 2])
        # In an even layer, apply gates starting at the first index
        # In an odd layer, apply gates starting at the second index
        # Like in the 1-qubit case, this works because of the way
        # tensordot moves indices
        state = jnp.tensordot(state, U, axes=([0+evenodd, 1+evenodd], [2, 3]))
    # Now, if the last qubit did not have a gate applied, it lies in the wrong
    # position. There are two possible cases here
    if evenodd == 0 and num_qubits % 2 == 1:
        # Move the last qubit, which is now at the front, to the back
        # TODO try doing this by actually reordering indices,
        # instead of this easier hack...
        state = jnp.tensordot(state, identity(1), axes=([0], [1]))
    if evenodd == 1 and num_qubits % 2 == 0:
        # Move the last qubit, which is now one after the front, to the back
        state = jnp.tensordot(state, identity(1), axes=([1], [1]))
    return state

def num_ansatz_params(num_qubits, depth):
    """Numper of continuous parameters in an ansatz circuit.
    This is for a 1D brickwork architecture with alternating U3 and RZZ gates.

    Parameters
    ----------
    num_qubits : int
        Number of qubits in the circuit.
    depth : int
        Depth of the circuit, as measured by number of two-qubit layers.

    Returns
    -------
    jax array
        The state with the corresponding U3 gates applied.
    """

    # Number of one-qubit gates
    # Add one to the depth because we do these before/after each two-qubit layer
    num_1 = num_qubits * 3 * (depth + 1)
    if num_qubits % 2 == 0:
        # Alternate between n/2 and (n/2) - 1 qubits
        num_2 = (num_qubits // 2) * depth - (depth // 2)
    else:
        # Always (n-1)/2 qubits
        num_2 = (num_qubits // 2) * depth
    return num_1 + num_2

def ansatz_state(params, num_qubits, depth):
    """Compute the output state of a 1D ansatz circuit with given parameters.
    This is for a 1D brickwork architecture with alternating U3 and RZZ gates.
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
        The state with the corresponding U3 gates applied.
    """

    # Initialize the all-zero state
    state = jnp.zeros(2**num_qubits)
    state = state.at[0].set(1)
    state = state.reshape([2] * num_qubits)

    # Current offset into the parameter array
    off = 0
    for layer in range(depth):
        # Apply a U3 layer
        state = one_qubit_layer(params[off:off+3*num_qubits], state, num_qubits)
        off += 3*num_qubits

        # Apply an RZZ layer
        num_two_qubit = (num_qubits - layer % 2) // 2
        state = two_qubit_layer(params[off:off+num_two_qubit], state, num_qubits, layer % 2)
        off += num_two_qubit

    # One more one-qubit layer
    state = one_qubit_layer(params[off:off+3*num_qubits], state, num_qubits)
    # off += 3*num_qubits
    
    return state

def ansatz_state_inefficient(params, num_qubits, depth):
    """Compute the output state of a 1D ansatz circuit with given parameters.
    This is for a 1D brickwork architecture with alternating U3 and RZZ gates.
    The circuit is initialized to the all zero state.
    The output should be the same as `ansatz_state`, but this implementation
    may be less efficient. Useful as a sanity check.

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
        The state with the corresponding U3 gates applied.
    """

    # Initialize the all-zero state
    state = jnp.zeros(2**num_qubits)
    state = state.at[0].set(1)

    # Current offset into the parameter array
    off = 0
    for layer in range(depth):
        # Apply a U3 layer
        one_qubit_Us = 1
        for i in range(num_qubits):
            theta, phi, lamda = params[off:off+3]
            off += 3
            U = u3(theta, phi, lamda)
            one_qubit_Us = jnp.kron(one_qubit_Us, U)
        state = one_qubit_Us @ state

        # Apply an RZZ layer
        # Apply identity on the first qubit, if necessary
        if layer % 2 == 0:
            two_qubit_Us = 1
        else:
            two_qubit_Us = identity(1)
        for i in range((num_qubits - layer % 2) // 2):
            theta = params[off]
            off += 1
            U = rzz(theta)
            two_qubit_Us = jnp.kron(two_qubit_Us, U)
        # Apply identity on the last qubit, if necessary
        if (layer + num_qubits) % 2 == 1:
            two_qubit_Us = jnp.kron(two_qubit_Us, identity(1))
        state = two_qubit_Us @ state

    # One more one-qubit layer
    one_qubit_Us = 1
    for i in range(num_qubits):
        theta, phi, lamda = params[off:off+3]
        U = u3(theta, phi, lamda)
        one_qubit_Us = jnp.kron(one_qubit_Us, U)
        off += 3
    state = one_qubit_Us @ state

    return state
