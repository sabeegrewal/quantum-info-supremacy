import jax
import jax.numpy as jnp

import numpy as np
import scipy

from ansatz_qujax import *

# Class for optimizing ansatz circuits for quantum state preparation
class AnsatzOptimizer:
    def __init__(self, n):
        """Class constructor for ansatz circuit optimizer.

        Parameters
        ----------
        n : int
            Number of qubits in the state.
        """
        
        self.n = n
        if n % 2 == 0:
            self.depth_modulus = 2
        else:
            self.depth_modulus = n
        self.ansatz_fn = make_brickwork_ansatz_fn(n, self.depth_modulus)
        self.loss_and_grad = jax.jit(jax.value_and_grad(self.loss))
        self.noisy_loss_and_grad = jax.jit(jax.value_and_grad(self.noisy_loss))

    def zzphase_params(self, all_params):
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
        
        # Ignore the initial state parameters at the front
        circ_params = all_params[2*self.n:]
        # Reshape the parameter array into blocks of the appropriate length for the repeated circuit
        # Infer the number of repetitions, which is the first coordinate
        reshaped_params = circ_params.reshape(-1, 7 * (self.n // 2) * self.depth_modulus)
        # In each row, the first (n // 2) * self.depth_modulus parameters correspond to ZZs
        return reshaped_params[:,:(self.n // 2) * self.depth_modulus].flatten()

    def output_state(self, all_params):
        """Given a parameterized ansatz circuit and a target state, compute state
        output by the circuit.

        Parameters
        ----------
        all_params : jax array
            Parameters for the ansatz circuit.
            Should have length `num_params(depth)` for some `depth` divisible by `depth_modulus`.

        Returns
        -------
        jax array
            The output state as an array of shape `[2] * n`.
        """
        
        product_params = all_params[:2*self.n]
        circ_params = all_params[2*self.n:]

        initial_state = product_state(product_params)
        return self.ansatz_fn(circ_params, initial_state)

    def loss(self, all_params, target_state):
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

        output_state = self.output_state(all_params)
        return -abs(jnp.vdot(target_state, output_state))**2

    def noisy_loss(self, all_params, target_state):
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
        
        noisy_params = self.zzphase_params(all_params)
        # Overall fidelity is the product of individual gate fidelities
        fidelity_from_noise = jnp.prod(zzphase_fidelity(noisy_params))
        return fidelity_from_noise * self.loss(all_params, target_state)

    def num_params(self, depth):
        """The total number of continuous parameters in an ansatz circuit of the given depth.
        
        Parameters
        ----------
        depth : int
            Depth of the ansatz circuit, as measured by the number of two-qubit ZZ layers.
            Must be divisible by `depth_modulus`.

        Returns
        -------
        int
            Number of ansatz parameters.
        """
        
        if depth % self.depth_modulus != 0:
            raise Exception("depth must be an integer multiple of depth_modulus")
        # 2*n: initial product state parameters
        # 7*(n//2): ansatz circuit paramaters per layer of 2-qubit gates (1 for ZZ, 2*3 for U3)
        # There are depth many layers total
        return 2*self.n + 7*(self.n//2)*depth

    def optimize(self, target_state, depth,
                 method="L-BFGS-B", noisy=False, maxiter=2500, init_params=None):
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
