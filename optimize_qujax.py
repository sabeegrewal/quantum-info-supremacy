import jax
import jax.numpy as jnp

import numpy as np

from ansatz_qujax import *

import scipy

import matplotlib.pyplot as plt

class AnsatzOptimizer:
    def __init__(self, n):
        self.n = n
        if n % 2 == 0:
            self.depth_modulus = 2
        else:
            self.depth_modulus = n
        self.ansatz_fn = make_brickwork_ansatz_fn(n, self.depth_modulus)
        self.loss_and_grad = jax.jit(jax.value_and_grad(self.loss))
        self.noisy_loss_and_grad = jax.jit(jax.value_and_grad(self.noisy_loss))

    def zzphase_params(self, all_params):
        # Ignore the initial state parameters at the front
        circ_params = all_params[2*self.n:]
        # Reshape the parameter array into blocks of the appropriate length for the repeated circuit
        # Infer the number of repetitions, which is the first coordinate
        reshaped_params = circ_params.reshape(-1, 7 * (self.n // 2) * self.depth_modulus)
        # In each row, the first (n // 2) * self.depth_modulus parameters correspond to ZZs
        return reshaped_params[:,:(self.n // 2) * self.depth_modulus]

    def loss(self, all_params, target_state):
        product_params = all_params[:2*self.n]
        circ_params = all_params[2*self.n:]

        initial_state = product_state(product_params)
        output_state = self.ansatz_fn(circ_params, initial_state)
        return -abs(jnp.vdot(target_state, output_state))**2

    def noisy_loss(self, all_params, target_state):
        noisy_params = self.zzphase_params(all_params).flatten()
        # Overall fidelity is the product of individual gate fidelities
        fidelity_from_noise = jnp.prod(zzphase_fidelity(noisy_params))
        return fidelity_from_noise * self.loss(all_params, target_state)

    def num_params(self, depth):
        if depth % self.depth_modulus != 0:
            raise Exception("depth must be an integer multiple of depth_modulus")
        # 2*n: initial product state parameters
        # 7*(n//2): ansatz circuit paramaters per layer of 2-qubit gates (1 for ZZ, 2*3 for U3)
        # There are depth many layers total
        return 2*self.n + 7*(self.n//2)*depth

    def optimize(self, target_state, depth,
                 method="L-BFGS-B", noisy=False, maxiter=2500, init_params=None):
        if depth % self.depth_modulus != 0:
            raise Exception("depth must be an integer multiple of depth_modulus")

        # TODO do this with a seed
        if init_params is None:
            init_params = np.random.normal(scale=0.2, size=self.num_params(depth))
        init_params = jnp.array(init_params)
        if init_params.shape != (self.num_params(depth),):
            raise Exception("init_params must be a 1D array of length num_params(depth)")
        
        if noisy:
            value_and_grad = self.noisy_loss_and_grad
        else:
            value_and_grad = self.loss_and_grad

        opt = scipy.optimize.minimize(value_and_grad,
                                      init_params,
                                      args=(target_state),
                                      method=method,
                                      jac=True,
                                      options={"maxiter": maxiter})
        return opt

jax.config.update("jax_traceback_filtering", "off")

n = 10
optimizer = AnsatzOptimizer(n)

for depth in range(6, 7):
    all_two_qubit_params = []
    for i in range(10):
        init_params = np.random.normal(size=optimizer.num_params(depth), scale=1.0)
        
        target_state = np.random.normal(size=([2] * n)) + 1j * np.random.normal(size=([2] * n))
        target_state = target_state / np.linalg.norm(target_state)

        opt = optimizer.optimize(target_state, depth)
        print(opt.fun)
        all_two_qubit_params.append(optimizer.zzphase_params(opt.x))
    plt.hist(jnp.array(all_two_qubit_params).flatten(), bins=50)
    plt.show()
