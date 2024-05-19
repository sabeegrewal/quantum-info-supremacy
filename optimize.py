import quimb as qu
import quimb.tensor as qtn
import jax


def optimize(circ, target_state_dense):
    n = circ.N
    target_state = qtn.Tensor(
        data=target_state_dense.reshape([2] * n),
        inds=[f"k{i}" for i in range(n)],
        tags={"psi_TARGET"},
    )
    V = circ.uni
    zero_dense = qu.computational_state("0" * n)
    zero = qtn.Tensor(
        data=zero_dense.reshape([2] * n),
        inds=[f"b{i}" for i in range(n)],
        tags={"zero"},
    )

    def loss(V, target_state):
        return 1 - abs((zero & V & target_state).contract()) ** 2

    initial_loss = loss(V, target_state)

    tnopt = qtn.TNOptimizer(
        V,  # the tensor network we want to optimize
        loss,  # the function we want to minimize
        loss_constants={"target_state": target_state},
        tags=["U3", "RZZ"],  # only optimize U3 tensors
        autodiff_backend="jax",  # use 'autograd' for non-compiled optimization
        optimizer="L-BFGS-B",  # the optimization algorithm
    )
    jax.config.update("jax_enable_x64", True)

    V_opt = tnopt.optimize_basinhopping(n=500, nhop=10)
    final_loss = loss(V_opt, target_state)
    return (V_opt, initial_loss, final_loss)
