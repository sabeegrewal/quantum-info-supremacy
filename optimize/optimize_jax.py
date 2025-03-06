import jax
import jax.numpy as jnp

import numpy as np
import scipy

from qiskit import QuantumCircuit

from pytket import Circuit

from ansatz.ansatz_jax import *

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
        self.num_params_per_mod = (n // 2) * 7 * self.depth_modulus
        self.loss_and_grad = jax.jit(jax.value_and_grad(self.loss))
        self.noisy_loss_and_grad = jax.jit(jax.value_and_grad(self.noisy_loss))

    def repeated_circuit(self, initial_state, circ_params):
        """Apply `apply_circuit` to the initial state as many times as possible.
        This is a helper method to reduce jit compilation time in jax.

        Parameters
        ----------
        initial_state : jax array
            Quantum state to apply the circuit to.
            Should have shape `[2] * n` for some `n`.
        params : jax array
            List of real parameters that define the ansatz circuit.
            Should have length `(n // 2) * depth * 7 * k` for some integer `k`.

        Returns
        -------
        jax array
        The output state as an array of shape `[2] * n`.
        """

        # Reshape the parameter array into blocks of the appropriate length for the repeated circuit
        # Infer the number of repetitions, which is the first coordinate
        reshaped_params = circ_params.reshape(-1, self.num_params_per_mod)

        # Helper function used to apply a single round of gates of length depth_modulus
        def f(state, p):
            return apply_circuit(self.depth_modulus, state, p), None

        # Use jax.lax.scan to improve compilation time, instead of unrolling the entire for-loop
        # Inspiration taken from qujax:
        # https://github.com/CQCL/qujax/blob/0a69ced74084301e087ad02429c47a54044ad6ae/qujax/utils.py#L496
        result, _ = jax.lax.scan(f, initial_state, reshaped_params)
        return result
        

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

        product_params = all_params[: 2 * self.n]
        circ_params = all_params[2 * self.n :]

        initial_state = product_state(product_params)
        return self.repeated_circuit(initial_state, circ_params)

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
        return -abs(jnp.vdot(target_state, output_state)) ** 2

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
        circ_params = all_params[2 * self.n :]
        # Reshape the parameter array into blocks of the appropriate length for the repeated circuit
        # Infer the number of repetitions, which is the first coordinate
        reshaped_params = circ_params.reshape(-1, self.num_params_per_mod)
        # In each row, the first (n // 2) * self.depth_modulus parameters correspond to ZZs
        return reshaped_params[:, : (self.n // 2) * self.depth_modulus].flatten()

    def fidelity_from_noise(self, all_params):
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

        noisy_params = self.zzphase_params(all_params)
        # Overall fidelity is the product of individual gate fidelities
        return jnp.prod(zzphase_fidelity(noisy_params))

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

        return self.fidelity_from_noise(all_params) * self.loss(
            all_params, target_state
        )

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
        return 2 * self.n + 7 * (self.n // 2) * depth

    def optimize(
        self,
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

        if depth % self.depth_modulus != 0:
            raise Exception("depth must be an integer multiple of depth_modulus")

        # TODO do this with a seed
        if init_params is None:
            init_params = np.random.normal(scale=0.2, size=self.num_params(depth))
        init_params = jnp.array(init_params)
        if init_params.shape != (self.num_params(depth),):
            raise Exception(
                "init_params must be a 1D array of length num_params(depth)"
            )

        if noisy:
            value_and_grad = self.noisy_loss_and_grad
        else:
            value_and_grad = self.loss_and_grad

        opt = scipy.optimize.minimize(
            value_and_grad,
            init_params,
            args=(target_state),
            method=method,
            jac=True,
            options={"maxiter": maxiter},
        )
        return opt

    def circuit(self, all_params, method):
        """Output a pytket or qiskit circuit with the desired parameters.

        Parameters
        ----------
        all_params : jax array
            Parameters for the ansatz circuit.
            Must have length `num_params(depth)` for some `depth` dividing `depth_modulus`.
        method : str
            One of "pytket" or "qiskit".

        Returns
        -------
        pytket Circuit or qiskit QuantumCircuit
            The corresponding circuit.
        """

        if method == "pytket":
            qc = Circuit(self.n)
        elif method == "qiskit":
            # Need to multiply by pi because pytket's conventions are different from qiskit's
            all_params = all_params * np.pi
            qc = QuantumCircuit(self.n)
        else:
            raise Exception("Unsupported circuit method: " + method)

        product_params = all_params[: 2 * self.n]
        circ_params = all_params[2 * self.n :]

        product_params_reshaped = product_params.reshape(self.n, 2)
        for i in range(self.n):
            theta, phi = product_params_reshaped[i]
            if method == "pytket":
                qc.U3(theta, phi, 0, i)
            else:
                qc.u(theta, phi, 0, i)

        circ_params_reshaped = circ_params.reshape(-1, self.num_params_per_mod)
        for iter_circ_params in circ_params_reshaped:
            # Number of parameters in RZZ gates
            num_rzz_params = (self.n // 2) * self.depth_modulus

            # Offsets into the parameter array
            rzz_off = 0
            u3_off = num_rzz_params

            for layer in range(self.depth_modulus):
                # First identify the paired qubits
                pairs = brickwork_pairs(self.n, layer)

                # Apply 2- and 1-qubit gates to each pair
                for i, j in pairs:
                    if method == "pytket":
                        # RZZ gate to apply
                        qc.ZZPhase(iter_circ_params[rzz_off], i, j)
                        # U3 gates to apply
                        qc.U3(iter_circ_params[u3_off],   iter_circ_params[u3_off+1], iter_circ_params[u3_off+2], i)
                        qc.U3(iter_circ_params[u3_off+3], iter_circ_params[u3_off+4], iter_circ_params[u3_off+5], j)
                    else:
                        # RZZ gate to apply
                        qc.rzz(iter_circ_params[rzz_off], i, j)
                        # U3 gates to apply
                        qc.u(iter_circ_params[u3_off],   iter_circ_params[u3_off+1], iter_circ_params[u3_off+2], i)
                        qc.u(iter_circ_params[u3_off+3], iter_circ_params[u3_off+4], iter_circ_params[u3_off+5], j)
                    rzz_off += 1
                    u3_off += 6
                    
        return qc
