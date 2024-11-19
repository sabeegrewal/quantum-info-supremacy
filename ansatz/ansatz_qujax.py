import jax
import jax.numpy as jnp

import qujax
from qujax import print_circuit, repeat_circuit, all_zeros_statetensor

# Based on https://github.com/CQCL/qujax/blob/main/examples/reducing_jit_compilation_time.ipynb


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


def brickwork_circuit_gates(num_qubits, depth):
    """Compute the positions of gates in a 1D ansatz circuit. This is for a
    1D cyclic brickwork architecture with alternating RZZ and U3 gates.

    Parameters
    ----------
    num_qubits : int
        Number of qubits in the circuit and its output state.
    depth : int
        Depth of the ansatz circuit, as measured by number of two-qubit layers.

    Returns
    -------
    (gates, qubit_inds, param_inds, num_params)
        gates : names of gates in the circuit, in order of application.
        qubit_inds : indices at which the gates are applied.
        param_inds : indices into the parameter array to be used for each gate.
        num_params : the number of continuous parameters in the circuit.
    """

    gates: List[str] = []
    qubit_inds: List[List[int]] = []
    param_inds: List[List[int]] = []

    # Number of parameters in RZZ gates
    num_rzz_params = (num_qubits // 2) * depth
    # Number of parameters in U3 gates
    # There are 2 U3 gates per RZZ, and each has 3 parameters
    num_u3_params = num_rzz_params * 6
    num_params = num_rzz_params + num_u3_params

    # Offsets into the parameter array
    rzz_off = 0
    u3_off = num_rzz_params

    for layer in range(depth):
        # First identify the paired qubits
        pairs = brickwork_pairs(num_qubits, layer)

        # Apply an RZZ layer
        for i, j in pairs:
            gates.append("ZZPhase")
            qubit_inds.append([i, j])
            param_inds.append([rzz_off])
            rzz_off += 1

        # Apply a U3 layer, but only to the qubits that were paired
        indices = list(sum(pairs, ()))  # This flattens the list of paired indices
        for i in indices:
            gates.append("U3")
            qubit_inds.append([i])
            param_inds.append([u3_off, u3_off + 1, u3_off + 2])
            u3_off += 3

    return gates, qubit_inds, param_inds, num_params


def make_brickwork_ansatz_fn(num_qubits, depth):
    """Returns a function to evaluate the output state of a parametrized circuit.
    This is for a 1D cyclic brickwork architecture with alternating RZZ and U3 gates.

    Parameters
    ----------
    num_qubits : int
        Number of qubits in the circuit and its output state.
    depth : int
        Depth of the ansatz circuit, as measured by number of two-qubit layers.

    Returns
    -------
    callable
        A function that takes a pair of inputs (params, initial_state) and applies the circuit with
        the parameters as many times as possible to the input state given the length of params.
    """

    gates, qubit_inds, param_inds, num_params = brickwork_circuit_gates(
        num_qubits, depth
    )
    # Get function that returns one application of the circuit
    params_to_statetensor = qujax.get_params_to_statetensor_func(
        gates, qubit_inds, param_inds
    )
    return repeat_circuit(params_to_statetensor, num_params)


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
    pitheta_2 = jnp.pi * theta / 2
    cpt2 = jnp.cos(pitheta_2)
    spt2 = jnp.sin(pitheta_2)
    eipp = jnp.exp(1j * jnp.pi * phi)
    return jnp.array([cpt2, eipp * spt2])


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
