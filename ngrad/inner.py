"""
Contains functions to compute inner products on tangent space.

"""
import jax.numpy as jnp
from jax import jacfwd, hessian
from jax.flatten_util import ravel_pytree

# transformations to be used in the definition of gramians
def model_del_i_factory(argnum=0):
    """
    Partial derivative for a function of signature (d,) ---> PyTree
    Intended to use when defining inner products for gramians.
    
    Parameters
    ----------
    N: int = 0
        Which partial derivative to take.

    """
    def model_del_i(u_theta, del_theta_model):
        """
        Parameters
        ----------
        u_theta: Callable
            Required only for nonlinear problems.
        del_theta_model: Callable
            typically: lambda z: grad(model)(params, z)
        """
        
        def del_theta_model_splitvar(*args):
            x_ = jnp.array(args)
            return del_theta_model(x_)

        d_splitvar_di = jacfwd(del_theta_model_splitvar, argnum)

        def d_del_theta_model_di(x):
            return d_splitvar_di(*x)
        
        return d_del_theta_model_di
    
    return model_del_i

def model_laplace(u_theta, del_theta_u):
    """
    Computes the laplacian componentwise of a function that maps
    into parameter space. Typically the only usage for this method is
    to be passed to the gramian as the trafo argument.

    Parameters
    ----------
    u_theta: Callable
        for fixed params theta: x -> u(theta, x). The function 
        model_laplace does not depend on this argument as this
        is only required in nonlinear settings
    del_theta_u: Callable
        Typically: x -> del_theta u(theta, x)
    
    """
    def del_theta_u_ravel(x):
        return ravel_pytree(del_theta_u(x))[0]

    def del_theta_u_laplace(x):
        unravel = ravel_pytree(del_theta_u(x))[1]
        return unravel(jnp.trace(hessian(del_theta_u_ravel)(x), axis1=1, axis2=2))

    return del_theta_u_laplace

def model_identity(u_theta, del_theta_u):
    return del_theta_u