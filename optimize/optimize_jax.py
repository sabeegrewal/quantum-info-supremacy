import jax
import jax.numpy as jnp

import numpy as np
import scipy

from ansatz.ansatz_jax import *

# --------------
# Noise modeling
# --------------

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

def zzphase_params(num_qubits, all_params):
    """Given a list of parameters to the ansatz circuit, identify the parameters
    corresponding to ZZ gates.

    Parameters
    ----------
    all_params : jax array
        Parameters for the ansatz circuit.
        Should have length `n*2 + 7*(n//2)*depth` for some `depth` dividing `depth_modulus`.

    Returns
    -------
    jax array
        A jax array of length `(n//2)*depth` containing all of the ZZ parameters.
    """

    mod = depth_modulus(num_qubits)
    num_params_per_mod = num_gate_params(num_qubits, mod)
    
    # Ignore the initial state parameters at the front
    circ_params = all_params[2*num_qubits:]
    # Reshape the parameter array into blocks of the appropriate length for the repeated circuit
    # Infer the number of repetitions, which is the first coordinate
    reshaped_params = circ_params.reshape(-1, num_params_per_mod)
    # In each row, the first (n // 2) * depth_modulus parameters correspond to ZZs
    return reshaped_params[:, : (num_qubits // 2) * mod].flatten()

def fidelity_from_noise(num_qubits, all_params):
    """Given a parameterized ansatz circuit, compute an estimate of the multiplicative
    factor on fidelity due to noise

    Parameters
    ----------
    all_params : jax array
        Parameters for the ansatz circuit.
        Should have length `n*2 + 7*(n//2)*depth` for some `depth` dividing `depth_modulus`.

    Returns
    -------
    real
        Estimated noise rate.
    """

    noisy_params = zzphase_params(num_qubits, all_params)
    # Overall fidelity is the product of individual gate fidelities
    return jnp.prod(zzphase_fidelity(noisy_params))



# --------------
# Loss functions
# --------------

def loss(all_params, target_state):
    """Given a parameterized ansatz circuit and a target state, compute the loss function
    for the ansatz circuit's output fidelity.

    Parameters
    ----------
    all_params : jax array
        Parameters for the ansatz circuit.
        Should have length `num_params(depth)` for some `depth` divisible by `depth_modulus`.
    target_state : jax array
        The target state. Should have shape `[2] * n`.

    Returns
    -------
    real
        The loss for this target state.
    """

    num_qubits = len(target_state.shape)
    out = output_state(num_qubits, all_params)
    return -abs(jnp.vdot(target_state, out)) ** 2

def noisy_loss(all_params, target_state):
    """Given a parameterized ansatz circuit and a target state, compute the loss function
    for the ansatz circuit's output fidelity, assuming that the circuit is corrupted by
    experimental noise.

    Parameters
    ----------
    all_params : jax array
        Parameters for the ansatz circuit.
        Should have length `n*2 + 7*(n//2)*depth` for some `depth` dividing `depth_modulus`.
    target_state : jax array
        The target state. Should have shape `[2] * n`.

    Returns
    -------
    real
        The loss for this target state.
    """

    num_qubits = len(target_state.shape)
    return fidelity_from_noise(num_qubits, all_params) * loss(all_params, target_state)

loss_and_grad = jax.jit(jax.value_and_grad(loss))
noisy_loss_and_grad = jax.jit(jax.value_and_grad(noisy_loss))



# ------------
# Optimization
# ------------

def optimize(
    target_state,
    depth,
    method="L-BFGS-B",
    noisy=False,
    maxiter=10000,
    init_params=None,
):
    """Optimize the ansatz circuit with respect to the target state.

    Parameters
    ----------
    target_state : jax array
        The target state. Should have shape `[2] * n`.
    depth : int
        Depth of the ansatz circuit, as measured by the number of two-qubit ZZ layers.
        Must be divisible by `depth_modulus`.
    method : str
        Scipy optimizer method to use. Defaults to `"L-BFGS-B"`.
    noisy : bool
        Whether to optimize the loss function accounting for experimental noise.
        Defaults to `False`.
    maxiter : int
        Maximum number of iterations for the optimizer. Defaults to `2500`.
    init_params : jax_array or None
        If provided, the initial parameters for the ansatz circuit.
        Must have length `num_params(depth)`.
        If `None`, the initial parameters are chosen randomly.

    Returns
    -------
    scipy OptimizeResult
        The result of running the scipy optimization.
    """

    num_qubits = len(target_state.shape)
    if depth % depth_modulus(num_qubits) != 0:
        raise Exception("depth must be an integer multiple of depth_modulus")

    total_num_params = num_qubits * 2 + num_gate_params(num_qubits, depth)
    # TODO do this with a seed
    if init_params is None:
        init_params = np.random.normal(scale=0.2, size=total_num_params)
    init_params = jnp.array(init_params)
    if init_params.shape != (total_num_params,):
        raise Exception(
            "init_params must be a 1D array of length num_params(depth)"
        )

    if noisy:
        value_and_grad = noisy_loss_and_grad
    else:
        value_and_grad = loss_and_grad

    opt = scipy.optimize.minimize(
        value_and_grad,
        init_params,
        args=(target_state),
        method=method,
        jac=True,
        options={"maxiter": maxiter},
    )
    return opt
