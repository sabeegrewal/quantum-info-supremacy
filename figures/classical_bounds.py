import jax
from jax.lax import erf
import jax.numpy as jnp
import jaxopt

import numpy as np
import scipy.integrate as integrate

# Seems to need 64-bit precision for numerical stability of erf
jax.config.update('jax_enable_x64', True)

# Bound on Frobenius norm
def frob_norm(n, ensemble):
    if ensemble in {'haar', 'design', 'cliff'}:
        return jnp.sqrt(2 / (jnp.exp2(n) + 1))
    elif ensemble == 'cliff_prod':
        return jnp.sqrt((2 / 3)**n)
    else:
        raise Exception(ensemble + ' not in {haar, design, cliff, cliff_prod')

# Bound on Clifford operator norm via t-th moment
def cliff_op_norm(n, t):
    prod_list = jnp.array([(1 + jnp.exp2(i)) / (jnp.exp2(n) + jnp.exp2(i)) for i in range(t - 1)])
    return jnp.prod(prod_list)**(1 / t)

# Bound on design operator norm via t-th moment
def design_op_norm(n, t):
    prod_list = jnp.array([(1 + i) / (jnp.exp2(n) + i) for i in range(1, t)])
    return jnp.prod(prod_list)**(1 / t)

# Bound on operator norm
def op_norm(n, ensemble):
    if ensemble == 'haar':
        H = 0
        for i in range(2**n):
            H += 1 / (i + 1)
        return H / jnp.exp2(n)
    elif ensemble == 'cliff_prod':
        return jnp.sqrt((2 / 3)**n)
    elif ensemble == 'cliff':
        # Bound is valid for all t >= 2
        # Should be optimized around t = sqrt(n)
        # Try everything between 2 and n+2
        norm_list = jnp.array([cliff_op_norm(n, t) for t in range(2, n+3)])
        return jnp.min(norm_list)
    elif ensemble == 'design':
        # Bound is valid for all t >= 2
        # Should be optimized around t = n
        # Try everything between 2 and n+2
        norm_list = jnp.array([design_op_norm(n, t) for t in range(2, n+3)])
        return jnp.min(norm_list)
    else:
        raise Exception(ensemble + ' not in {haar, design, cliff, cliff_prod')

# The quantity $\gamma$ that appears in the $\psi_{11}$ Orlicz norm
def gamma(a):
    return a * jnp.exp(1/a) / (a+1) + 2 / (jnp.e*(a**3 - a)) - 1

def bound_large_m(a, m, g, A, B, t_star):
    return t_star + a * B

def bound_small_m(a, m, g, A, B, t_star):
    term1 = jnp.exp2(m) * jnp.sqrt(jnp.pi * g) * a * A * (erf(jnp.sqrt(g)*A/B) - erf(t_star / (2*jnp.sqrt(g)*a*A)))
    term2 = jnp.exp2(m) * a * B * jnp.exp(-g * A**2 / B**2)
    return t_star + term1 + term2

# The bound computed in Theorem 3
# Should probably break into more functions
def classical_xeb_bound(a, m, n, ensemble):
    g = gamma(a)
    A = frob_norm(n, ensemble)
    B = op_norm(n, ensemble)
    
    ln2 = jnp.log(2)
    thr = g * A**2 / (ln2 * B**2)

    t_star = jax.lax.select(
        m >= thr,
        a * B * (ln2 * m + g * A**2 / B**2),
        jnp.sqrt(ln2 * m * g) * 2 * a * A
    )

    # Using cond here seems more numerically stable because
    # the small_m bound sometimes returns nans when m is really large
    return jax.lax.cond(
        m >= thr,
        bound_large_m,
        bound_small_m,
        a, m, g, A, B, t_star
    )

# Just compute the best a
def optimum_a(m, n, ensemble):
    def wrapped_bound(a):
        return classical_xeb_bound(a, m, n, ensemble)
    
    solver = jaxopt.BFGS(fun=jax.value_and_grad(wrapped_bound), value_and_grad=True)
    res = solver.run(init_params=1.5)
    a, state = res
    return a
optimum_a_jit = jax.jit(optimum_a, static_argnums=[1, 2])

# Minimize the bound over a
def max_classical_xeb(m, n, ensemble):
    def wrapped_bound(a):
        return classical_xeb_bound(a, m, n, ensemble)
    
    solver = jaxopt.BFGS(fun=jax.value_and_grad(wrapped_bound), value_and_grad=True)
    res = solver.run(init_params=1.5)
    a, state = res
    return classical_xeb_bound(a, m, n, ensemble)
max_classical_xeb_jit = jax.jit(max_classical_xeb, static_argnums=[1, 2])

# Approximate inverse of the function above
def min_communication(xeb, n, ensemble, tol=1e-6):
    # We could use gradient methods here, but this seems to be much faster and less buggy
    # Perform an exponential search starting at one bit of communication
    lo = 0.001
    if xeb < max_classical_xeb_jit(lo, n, ensemble):
        return None
    # Try to find an upper bound
    hi = 2.0
    while xeb > max_classical_xeb_jit(hi, n, ensemble):
        # I'm worried about the numerical stability of max_classical_xeb,
        # so that's why we shouldn't increase the upper bound too quickly
        hi *= 1.25
    # Now binary search
    while hi - lo > tol:
        mid = (hi + lo) / 2
        if xeb >= max_classical_xeb_jit(mid, n, ensemble):
            lo = mid
        else:
            hi = mid
    return mid

# Achievable lower bound on XEB
def achievable_classical_xeb(m, n):
    N = 2**n
    M = 2**m

    def integrand(u):
        return np.exp(-M * u**(N - 1))

    integral = integrate.quad(integrand, 0, 1)[0]
    H_N_minus_1 = sum(1 / i for i in range(2, N+1))
    return H_N_minus_1 * (1 - N * integral / (N - 1))
