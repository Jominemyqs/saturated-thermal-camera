"""Numerical attribution for a constant endpoint gradient; no fitting."""
import numpy as np
from numpy.polynomial.legendre import leggauss


def constant_gradient_attribution(q, *, gamma, dt, propagate):
    """propagate(q, age) includes ONE lift and return, including at zero age.

    Never feed the already remapped field back into the operator. The physical
    interpretation remains conditional on this discretized surrogate.
    """
    if dt < 0:
        raise ValueError('Negative timestep')
    sources = dict(S0=gamma*np.asarray(q)*dt,
                   S_map=gamma*propagate(q, 0.)*dt)
    for order in (2, 4):
        nodes, weights = leggauss(order)
        sources[f'S_evolved_GL{order}'] = gamma*dt*sum(
            w/2*propagate(q, dt*(x+1)/2) for x,w in zip(nodes, weights))
    return sources
