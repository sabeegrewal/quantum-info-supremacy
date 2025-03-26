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
    return 1 - (5 / 4) * eps_tq - 3 * eps_mem


def zzphase_params(num_qubits, all_params):
    """Given a list of parameters to the ansatz circuit, identify the parameters
    corresponding to ZZ gates.

    Parameters
    ----------
    num_qubits : int
        Number of qubits in the circuit.
    all_params : jax array
        Parameters for the ansatz circuit.
        Should have length `2 * num_qubits + num_gate_params(num_qubits, depth)`
        for some `depth` divisible by `depth_modulus(num_qubits)`.

    Returns
    -------
    jax array
        A jax array of length `(num_qubits//2)*depth` containing all of the ZZ parameters.
    """

    # Ignore the initial state parameters at the front
    circ_params = all_params[2 * num_qubits :]
    # Reshape the parameter array into blocks of the appropriate length for the repeated circuit
    reshaped_params = reshape_params_by_mod(num_qubits, circ_params)

    mod = depth_modulus(num_qubits)
    # In each row, the first (n // 2) * mod parameters correspond to ZZs
    return reshaped_params[:, : (num_qubits // 2) * mod].flatten()


def fidelity_from_noise(num_qubits, all_params):
    """Given a parameterized ansatz circuit, compute an estimate of the multiplicative
    factor on fidelity due to noise

    Parameters
    ----------
    num_qubits : int
        Number of qubits in the circuit.
    all_params : jax array
        Parameters for the ansatz circuit.
        Should have length `2 * num_qubits + num_gate_params(num_qubits, depth)`
        for some `depth` divisible by `depth_modulus(num_qubits)`.

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
        Should have length `2 * n + num_gate_params(n, depth)`
        for some `n` and `depth` divisible by `depth_modulus(n)`.
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
        Should have length `2 * n + num_gate_params(n, depth)`
        for some `n` and `depth` divisible by `depth_modulus(n)`.
    target_state : jax array
        The target state. Should have shape `[2] * n`.

    Returns
    -------
    real
        The loss for this target state.
    """

    num_qubits = len(target_state.shape)
    return fidelity_from_noise(num_qubits, all_params) * loss(all_params, target_state)


# jit-compiled functions computing loss and gradient simultaneously
loss_and_grad = jax.jit(jax.value_and_grad(loss))
noisy_loss_and_grad = jax.jit(jax.value_and_grad(noisy_loss))


# ------------
# Optimization
# ------------


def haar_u3_params(rng):
    """
    Generate parameters for a Haar-random U3 gate.

    Parameters
    ----------
    rng : numpy Generator
        Random number generator to use for sampling.

    Returns
    -------
    (real, real, real)
        (theta, phi, lamda) distributed such that u3(theta, phi, lambda)
        equals the one-qubit unitary Haar measure, up to global phase.
    """

    # This method uses properties of the Euler angle parametrization of SU(2)
    # See Section 2.3 of:
    # http://home.lu.lv/~sd20008/papers/essays/Random%20unitary%20[paper].pdf
    # sin^2(pi*theta/2) should be uniform in [0,1]
    s2pt2 = rng.uniform(0, 1)
    theta = 2 / np.pi * np.arcsin(np.sqrt(s2pt2))
    phi = rng.uniform(0, 2)
    lamda = rng.uniform(0, 2)
    return (theta, phi, lamda)


def random_init_params(num_qubits, depth, rng=np.random):
    """
    Randomly initialize parameters for an ansatz circuit of the given depth.
    In this initialization, the U3 and product state parameters are chosen
    Haar-randomly, while the RZZ parameters are all set to 0.

    Parameters
    ----------
    num_qubits : int
        Number of qubits in the circuit.
    depth : int
        Depth of the ansatz circuit, as measured by number of two-qubit layers.
    rng : numpy Generator
        Random number generator to use for sampling.
        Defaults to `np.random`.

    Returns
    -------
    jax array
        Randomly generated initial parameters that can be used for optimization.
        Array has length `num_qubits * 2 + num_gate_params(num_qubits, depth)`.
    """

    mod = depth_modulus(num_qubits)
    num_rzz_params = (num_qubits // 2) * depth

    # First generate the product parameters
    # This means taking (theta, phi) but not lamda
    product_params = np.concatenate(
        [haar_u3_params(rng)[:2] for i in range(num_qubits)]
    )

    # Then generate the U3 parameters
    u3_params = np.concatenate([haar_u3_params(rng) for i in range(2 * num_rzz_params)])
    # Initialize all RZZ parameters to 0
    rzz_params = np.zeros(num_rzz_params)

    # Reshape into the appropriate number of blocks for the repeated circuit
    u3_params_reshaped = u3_params.reshape((depth // mod, -1))
    rzz_params_reshaped = rzz_params.reshape((depth // mod, -1))
    # Stack RZZ and U3 parameters horizontally, then flatten
    circ_params = np.hstack((rzz_params_reshaped, u3_params_reshaped)).flatten()
    return np.concatenate((product_params, circ_params))


def optimize(
    target_state,
    depth,
    method="L-BFGS-B",
    noisy=False,
    maxiter=10000,
    init_params=None,
    rng=np.random,
):
    """Optimize the ansatz circuit with respect to the target state.

    Parameters
    ----------
    target_state : jax array
        The target state. Should have shape `[2] * n` for some `n`.
    depth : int
        Depth of the ansatz circuit, as measured by the number of two-qubit ZZ layers.
        Must be divisible by `depth_modulus(n)`.
    method : str
        Scipy optimizer method to use. Defaults to `"L-BFGS-B"`.
    noisy : bool
        Whether to optimize the loss function accounting for experimental noise.
        Defaults to `False`.
    maxiter : int
        Maximum number of iterations for the optimizer. Defaults to `10000`.
    init_params : jax_array or None
        If provided, the initial parameters for the ansatz circuit.
        Must have length `2 * n + num_gate_params(n, depth)`.
        If `None`, the initial parameters are chosen randomly.
    rng : numpy Generator
        Random number generator to use for sampling, if `init_params` not provided.
        Defaults to `np.random`.

    Returns
    -------
    scipy OptimizeResult
        The result of running the scipy optimization.
    """

    num_qubits = len(target_state.shape)
    if depth % depth_modulus(num_qubits) != 0:
        raise Exception("depth must be an integer multiple of depth_modulus")

    total_num_params = num_qubits * 2 + num_gate_params(num_qubits, depth)
    if init_params is None:
        init_params = random_init_params(num_qubits, depth, rng=rng)
    init_params = jnp.array(init_params)
    if init_params.shape != (total_num_params,):
        raise Exception("init_params must be a 1D array of length num_params(depth)")

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
