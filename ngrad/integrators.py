"""
Implementation of integration methods.

"""
import jax.numpy as jnp
from jax import random

class DeterministicIntegrator():
    """
    Integration using domain.deterministic_integration_points().
    
    This integrator uses the deterministic_integration_points
    method of the domain given as an argument to perform
    integration via taking an average. Once instantiated, an
    Integrator is intended to be used as a Callable which
    then performs integration.

    Parameters
    ----------
    domain
        A domain class. Needs to provide the two methods
        domain.measure() and 
        domain.deterministic_integration_points(<int>).
    N: int = 50
        The number or density of integration points drawn.
        How many points exactly are being used depends on the
        implementation of the concrete domain used.
    K: int or None
        If K is not None then this splits the integration points
        in K chunks that are processed sequentially instead of 
        parallel. Can be used when GPU memory is limited.

    """
    def __init__(self, domain, N=50, K=None):
        self._domain = domain
        self._x = domain.deterministic_integration_points(N)
        self._K = K

        if K is not None:
            splits = [i * len(self._x)//K for i in range(1, K)]
            self.x_split = jnp.split(self._x, splits, axis=0)
        
        
    def __call__(self, f):
        mean = 0
        if self._K is not None:
            means = []
            for x in self.x_split:
                means.append(jnp.mean(f(x), axis=0))
            
            mean = jnp.mean(jnp.array(means), axis=0)

        else:
            y = f(self._x)
            mean = jnp.mean(y, axis=0)

        return self._domain.measure() * mean
    
class TrapezoidalIntegrator():
    """
    Integration over intervals using trapezoidal rule.

    """
    def __init__(self, domain, N=50, K=None):
        self._domain = domain
        self._x = domain.deterministic_integration_points(N)
        self._K = K

        if K is not None:
            splits = [i * len(self._x)//K for i in range(1, K)]
            self.x_split = jnp.split(self._x, splits, axis=0)
        
    def __call__(self, f):
        """
        Integration happens here. 

        f must map (n,1) tensors to (n,...) tensors.
        
        """
        
        sum = 0
        if self._K is not None:
            sums = []
            for x in self.x_split:
                sums.append(jnp.sum(f(x), axis=0))
            
            sum = jnp.sum(jnp.array(sums), axis=0)

        else:
            y = f(self._x)
            sum = jnp.sum(y, axis=0)

        x_first = jnp.expand_dims(self._x[0], axis=0)
        x_last  = jnp.expand_dims(self._x[-1], axis=0)
        dx = self._x[1,0] - self._x[0,0]

        # get rid of first axis of f(...) by evaluating at [0]
        return dx * (sum - 0.5 * (f(x_first)[0] + f(x_last)[0]))
    
class EvolutionaryIntegrator():
    """
    Implementation of an evolutionary integrator following
    the proposed algorithm of Daw et al in "Mitigating Propagation
    Failure in PINNs using Evolutionary Sampling". 

    Parameters
    ----------
    domain
        A domain class. Needs to provide the two methods
        domain.measure() and 
        domain.deterministic_integration_points(<int>).
    key
        A PRNGKey to sample random points.
    N: int = 50
        The number or density of integration points drawn.
        How many points exactly are being used depends on the
        implementation of the concrete domain used.
    K: int or None
        If K is not None then this splits the integration points
        in K chunks that are processed sequentially instead of 
        parallel. Can be used when GPU memory is limited.
    
    """
    def __init__(self, domain, key, N=50, K=None):
        self._domain = domain
        self._N = N
        self._key, subkey = random.split(key)
        self._x = self._domain.random_integration_points(subkey, N)
        self._K = K

        if K is not None:
            splits = [i * len(self._x)//K for i in range(1, K)]
            self.x_split = jnp.split(self._x, splits, axis=0)
        

    def __call__(self, f):
        mean = 0
        if self._K is not None:
            means = []
            for x in self.x_split:
                means.append(jnp.mean(f(x), axis=0))
            
            mean = jnp.mean(jnp.array(means), axis=0)

        else:
            y = f(self._x)
            mean = jnp.mean(y, axis=0)

        return self._domain.measure() * mean

    def update(self, residual):
        # compute fitness from residual
        fitness = jnp.abs(residual(self._x))
        
        # set the threshold
        threshold = jnp.mean(fitness)
        
        # remove non-fit collocation points
        mask = jnp.where(fitness > threshold, False, True)
        x_fit = jnp.delete(self._x, mask, axis=0)
        
        # advance random key
        self._key, subkey = random.split(self._key)

        # add new uniformly drawn collocation points to fill up
        N_fit = len(self._x) - len(x_fit)
        x_add = self._domain.random_integration_points(subkey, N_fit)
        self._x = jnp.concatenate([x_fit, x_add], axis=0)

    def new_rand_points(self):
        # advance random key
        self._key, subkey = random.split(self._key)

        # draw new random points
        self._x = self._domain.random_integration_points(subkey, self._N)

        if self._K is not None:
            splits = [i * len(self._x)//self._K for i in range(1, self._K)]
            self.x_split = jnp.split(self._x, splits, axis=0)