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

    pitheta_2 = (jnp.pi/2) * theta
    cpt2 = jnp.cos(pitheta_2)
    spt2 = jnp.sin(pitheta_2)
    eipl = jnp.exp((jnp.pi * 1j) * lamda)
    eipp = jnp.exp((jnp.pi * 1j) * phi)
    # See https://docs.quantum.ibm.com/api/qiskit/qiskit.circuit.library.U3Gate
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
    # See https://docs.quantum.ibm.com/api/qiskit/qiskit.circuit.library.RZZGate
    return jnp.diag(jnp.array([eipt2, eipt2c, eipt2c, eipt2]))

def two_qubit(params):
    u3_gate_a = u3(*params[:3])
    u3_gate_b = u3(*params[3:6])
    rzz_gate = rzz(params[6])
    
    u3_gates = jnp.kron(u3_gate_a, u3_gate_b)
    return jnp.matmul(u3_gates, rzz_gate)
    
def four_qubit(params):
    gate_1 = jnp.kron(jnp.kron(jnp.identity(2), two_qubit(params[:7])), jnp.identity(2))
    gate_2 = jnp.kron(two_qubit(params[7:14]), two_qubit(params[14:21]))
    gate_3 = jnp.kron(jnp.kron(jnp.identity(2), two_qubit(params[21:])), jnp.identity(2))
    return jnp.matmul(gate_3, jnp.matmul(gate_2, gate_1))

def apply_four_qubit(state, gate, i, j, k, l):
    indices = "abcdefghijkl"
    new_indices = list(indices)
    new_indices[i] = "w"
    new_indices[j] = "x"
    new_indices[k] = "y"
    new_indices[l] = "z"
    einsum_str = (indices + "," +
                  "wxyz" + indices[i] + indices[j] + indices[k] + indices[l] +
                  "->" + "".join(new_indices))
    return jnp.einsum(einsum_str, state, gate.reshape((2,2,2,2,2,2,2,2)))

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

    # Matching the conventions of qujax: https://cqcl.github.io/qujax/gates.html
    # Also matches Quantinuum: https://tket.quantinuum.com/api-docs/optype.html
    pitheta_2 = (jnp.pi / 2) * theta
    cpt2 = jnp.cos(pitheta_2)
    spt2 = jnp.sin(pitheta_2)
    eipp = jnp.exp((jnp.pi * 1j) * phi)
    return jnp.array([cpt2, eipp * spt2])

def apply_circuit(num_qubits, depth, initial_state, params):
    state = initial_state
    gate_1 = four_qubit(params[:28])
    state = apply_four_qubit(state, gate_1, 0,  1,  2,  3)
    gate_2 = four_qubit(params[28:56])
    state = apply_four_qubit(state, gate_2, 4,  5,  6,  7)
    gate_3 = four_qubit(params[56:84])
    state = apply_four_qubit(state, gate_3, 8,  9,  10, 11)
    gate_4 = four_qubit(params[84:112])
    state = apply_four_qubit(state, gate_4, 2,  3,  4,  5)
    gate_5 = four_qubit(params[112:140])
    state = apply_four_qubit(state, gate_5, 6,  7,  8,  9)
    gate_6 = four_qubit(params[140:168])
    state = apply_four_qubit(state, gate_6, 10, 11, 0,  1)   
    return state

def num_params(num_qubits, depth):
    return 168

def make_brickwork_ansatz_fn(num_qubits, depth):
    def repeated_circuit(params, initial_state):
        def f(state, p):
            return apply_circuit(num_qubits, depth, state, p), None

        reshaped_parameters = params.reshape(-1, num_params(num_qubits, depth))
        result, _ = jax.lax.scan(f, initial_state, reshaped_parameters)
        return result

    return repeated_circuit

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
