import jax
import jax.numpy as jnp

from pytket import Circuit
from qiskit import QuantumCircuit

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



# ----------------
# Gate application
# ----------------

def product_state(product_params):
    """Arbitrary product state with given parameters.

    Parameters
    ----------
    product_params : jax array
        Should have length `2 * n` for some `n`.

    Returns
    -------
    jax array
        The corresponding product state as a jax array of shape `[2] * n`.
    """

    # Infer the first dimension, which is the number of qubits
    product_params = product_params.reshape(-1, 2)
    result = 1
    for theta, phi in product_params:
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
    # Technically, this restricts us to at most 50 qubits...
    indices = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWX"[:len(state.shape)]
    # Replace names at locations i and j with "Y" and "Z", respectively
    new_indices = indices[:i] + "Y" + indices[i+1:]
    new_indices = new_indices[:j] + "Z" + new_indices[j+1:]
    # Contract indices[i] and indices[j], replacing with "Y" and "Z", respectively
    einsum_str = indices + ",YZ" + indices[i] + indices[j] + "->" + new_indices
    return jnp.einsum(einsum_str, state, gate)



# -------------------
# 1D brickwork ansatz
# -------------------

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

def depth_modulus(num_qubits):
    """The period at which the set of `brickwork_pairs` repeats.

    Parameters
    ----------
    num_qubits : int
        Number of qubits in the circuit.
        
    Returns
    -------
    int
        2 if `num_qubits` is even, otherwise `num_qubits`.
    """
    
    if num_qubits % 2 == 0:
        return 2
    else:
        return num_qubits

def num_gate_params(num_qubits, depth):
    """The number of gate parameters in a 1D brickwork ansatz circuit of the given depth,
    with alternating U3 and RZZ gates.

    Parameters
    ----------
    num_qubits : int
        Number of qubits in the circuit.
    depth : int
        Depth of the ansatz circuit, as measured by number of two-qubit layers.
        
    Returns
    -------
    int
        `depth * (num_qubits // 2) * 7`
    """

    # There are (num_qubits // 2) RZZ gates per layer
    # For each RZZ gate, there are 2 U3 gates
    # RZZ gates have 1 parameter, U3 hates have 3
    # Hence (num_qubits // 2) * 7 parameters per layer
    return depth * (num_qubits // 2) * 7

def apply_circuit(initial_state, circ_params):
    """Apply a 1D ansatz circuit with given parameters to the initial state.
    This is for a 1D cyclic brickwork architecture with alternating U3 and RZZ gates.

    Parameters
    ----------
    initial_state : jax array
        Quantum state to apply the circuit to.
        Should have shape `[2] * n` for some `n`.
    circ_params : jax array
        List of real parameters that define the ansatz circuit.
        Should have length `num_gate_params(n, depth)` for some `depth`.

    Returns
    -------
    jax array
        The output state as an array of shape `[2] * n`.
    """

    # Infer the number of qubits and depth
    num_qubits = len(initial_state.shape)
    # ((num_qubits // 2) * 7) is the number of gates per layer
    depth = len(circ_params) // ((num_qubits // 2) * 7)
    
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
            rzz_gate = rzz(circ_params[rzz_off])
            rzz_off += 1
            # U3 gates to apply
            u3_gate_i = u3(circ_params[u3_off],   circ_params[u3_off+1], circ_params[u3_off+2])
            u3_gate_j = u3(circ_params[u3_off+3], circ_params[u3_off+4], circ_params[u3_off+5])
            u3_off += 6

            # Multiply the 2- and 1-qubit gates together first
            # This reduces the total number of tensor contractions by a factor of 3
            u3_gates = jnp.kron(u3_gate_i, u3_gate_j)
            gate_matrix = jnp.matmul(u3_gates, rzz_gate)
            gate_tensor = gate_matrix.reshape([2,2,2,2])

            # Apply the combined gate
            state = apply_two_qubit(state, gate_tensor, i, j)
            
    return state

def reshape_params_by_mod(num_qubits, circ_params):
    """Reshape the circuit parameters into a matrix where each row contains
    the parameters for layers of the given modulus.

    Parameters
    ----------
    num_qubits : int
        Number of qubits in the circuit.
    circ_params : jax array
        List of real parameters that define the ansatz circuit.
        Should have length `num_gate_params(n, depth)` for some `depth`
        divisible by `depth_modulus(num_qubits)`.

    Returns
    -------
    jax array
        The circuit parameters reshaped into an array of shape `(x, y)`,
        where `y = num_gate_params(num_qubits, depth_modulus(num_qubits))`.
    """
    
    mod = depth_modulus(num_qubits)
    num_params_per_mod = num_gate_params(num_qubits, mod)
    # Infer the number of repetitions, which is the first coordinate
    return circ_params.reshape(-1, num_params_per_mod)

def apply_circuit_repeated(initial_state, circ_params):
    """Apply `apply_circuit` to the initial state as many times as possible.
    This is a helper method to reduce jit compilation time in jax.

    Parameters
    ----------
    initial_state : jax array
        Quantum state to apply the circuit to.
        Should have shape `[2] * n` for some `n`.
    params : jax array
        List of real parameters that define the ansatz circuit.
        Should have length `num_gate_params(n, depth)` for some `depth`
        divisible by `depth_modulus(n)`.

    Returns
    -------
    jax array
        The output state as an array of shape `[2] * n`.
    """

    # Infer the number of qubits
    num_qubits = len(initial_state.shape)
    
    # Reshape the parameter array into blocks of the appropriate length for the repeated circuit
    reshaped_params = reshape_params_by_mod(num_qubits, circ_params)

    # Helper function used to apply a single round of gates of length depth_modulus
    def f(state, p):
        return apply_circuit(state, p), None

    # Use jax.lax.scan to improve compilation time, instead of unrolling the entire for-loop
    # Inspiration taken from qujax:
    # https://github.com/CQCL/qujax/blob/0a69ced74084301e087ad02429c47a54044ad6ae/qujax/utils.py#L496
    result, _ = jax.lax.scan(f, initial_state, reshaped_params)
    return result

def output_state(num_qubits, all_params):
    """Compute the state output by the ansatz circuit with given parameters.

    Parameters
    ----------
    num_qubits : int
        Number of qubits in the circuit.
    all_params : jax array
        Parameters for the ansatz circuit.
        Should have length `2 * num_qubits + num_gate_params(num_qubits, depth)`
        for some `depth` divisible by `depth_modulus(num_qubits)`.
        The first `2 * num_qubits` parameters specify the initial product state,
        and the rest specify the gates in the circuit.

    Returns
    -------
    jax array
        The output state as an array of shape `[2] * n`.
    """

    product_params = all_params[:2*num_qubits]
    circ_params = all_params[2*num_qubits:]

    initial_state = product_state(product_params)
    return apply_circuit_repeated(initial_state, circ_params)

def make_circuit(num_qubits, all_params, method):
    """Convert the ansatz parameters into a pytket or qiskit circuit
    that prepares the same state.

    Parameters
    ----------
    num_qubits : int
        Number of qubits in the circuit.
    all_params : jax array
        Parameters for the ansatz circuit.
        Should have length `2 * num_qubits + num_gate_params(num_qubits, depth)`
        for some `depth` divisible by `depth_modulus(num_qubits)`.
        The first `2 * num_qubits` parameters specify the initial product state,
        and the rest specify the gates in the circuit.
    method : str
        One of "pytket" or "qiskit".

    Returns
    -------
    pytket Circuit or qiskit QuantumCircuit
        The corresponding circuit.
    """

    if method == "pytket":
        qc = Circuit(num_qubits)
        # Functions that we'll use to apply gates
        apply_u3 = qc.U3
        apply_rzz = qc.ZZPhase
    elif method == "qiskit":
        # Need to multiply by pi because pytket's conventions are different from qiskit's
        all_params = all_params * jnp.pi
        
        qc = QuantumCircuit(num_qubits)
        # Functions that we'll use to apply gates
        apply_u3 = qc.u
        apply_rzz = qc.rzz
    else:
        raise Exception("Unsupported circuit method: " + method)

    product_params = all_params[: 2 * num_qubits]
    circ_params = all_params[2 * num_qubits :]

    product_params_reshaped = product_params.reshape(num_qubits, 2)
    for i in range(num_qubits):
        theta, phi = product_params_reshaped[i]
        apply_u3(theta, phi, 0, i)

    mod = depth_modulus(num_qubits)
    # Reshape the parameter array into blocks of the appropriate length for the repeated circuit
    circ_params_reshaped = reshape_params_by_mod(num_qubits, circ_params)
    for iter_circ_params in circ_params_reshaped:
        # Number of parameters in RZZ gates
        num_rzz_params = (num_qubits // 2) * mod

        # Offsets into the parameter array
        rzz_off = 0
        u3_off = num_rzz_params

        for layer in range(mod):
            # First identify the paired qubits
            pairs = brickwork_pairs(num_qubits, layer)

            # Apply 2- and 1-qubit gates to each pair
            for i, j in pairs:
                # RZZ gate to apply
                apply_rzz(iter_circ_params[rzz_off], i, j)
                rzz_off += 1

                # U3 gates to apply
                apply_u3(iter_circ_params[u3_off],   iter_circ_params[u3_off+1], iter_circ_params[u3_off+2], i)
                apply_u3(iter_circ_params[u3_off+3], iter_circ_params[u3_off+4], iter_circ_params[u3_off+5], j)
                u3_off += 6
                
    return qc

