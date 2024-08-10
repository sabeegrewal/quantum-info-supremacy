import jax
import jax.numpy as jnp
import scipy
import numpy as np

from ansatz.ansatz_jax import *


def loss_fn(params, target_state, num_qubits, depth):
    output_state = ansatz_state(params, num_qubits, depth)
    return -abs(jnp.vdot(target_state, output_state)) ** 2


value_and_grad = jax.jit(jax.value_and_grad(loss_fn), static_argnums=[2, 3])

n = 9
import time

for depth in range(60):
    avg = 0
    # print(time.asctime())
    num_samples = 10
    for i in range(num_samples):
        target_state = np.random.normal(size=([2] * n)) + 1j * np.random.normal(
            size=([2] * n)
        )
        target_state = target_state / jnp.linalg.norm(target_state)

        init_params = np.random.uniform(size=(num_ansatz_params(n, depth),))

        opt = scipy.optimize.minimize(
            value_and_grad,
            init_params,
            args=(target_state, n, depth),
            method="L-BFGS-B",
            jac=True,
            options={"maxiter": 2500},
        )
        avg -= opt.fun
        # print(opt.fun)
    avg = avg / num_samples
    est = avg * (0.9989 ** (n * depth / 2))
    print(
        f"depth {depth}: avg {avg}, est {est}, expected {num_ansatz_params(n, depth) / 2**n}"
    )


##n = 2
##depth = 1
##target_state = np.random.normal(size=([2] * n)) + 1j * np.random.normal(size=([2] * n))
##target_state = target_state / jnp.linalg.norm(target_state)
##init_params = np.random.uniform(size=(num_ansatz_params(n, depth),))
##print(jax.make_jaxpr(loss_fn, static_argnums=[2, 3])(init_params, target_state, n, depth))


##init_params = jax.random.uniform(jax.random.PRNGKey(0), shape=(num_ansatz_params(n, depth),))
##state = ansatz_state(init_params, n, depth)
##
##jax.config.update("jax_enable_x64", True)
##sanity_check_circ = ansatz_circ_quimb(init_params, n, depth)
##sanity_check_state = sanity_check_circ.psi.to_dense().reshape([2] * n)
##
##norm = jnp.linalg.norm(state - sanity_check_state)
##print(norm)
