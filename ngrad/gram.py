"""
Implementation of Gramians and natural gradients.

"""
import jax.numpy as jnp
from jax import grad, vmap
import jax.flatten_util
from jax.numpy.linalg import lstsq


def pre_gram_factory(model, trafo):

    def del_theta_model(params, x):
        return grad(model)(params, x)
    
    def pre_gram(params, x):
        
        def g(y):
            return trafo(
                lambda z: model(params, z),
                lambda z: del_theta_model(params, z),
            )(y)
        
        flat = jax.flatten_util.ravel_pytree(g(x))[0]
        flat_col = jnp.reshape(flat, (len(flat), 1))
        flat_row = jnp.reshape(flat, (1, len(flat)))
        return jnp.matmul(flat_col, flat_row)

    return pre_gram

def gram_factory(model, trafo, integrator):

    pre_gram = pre_gram_factory(model, trafo)
    v_pre_gram = vmap(pre_gram, (None, 0))
    
    def gram(params):
        gram_matrix = integrator(lambda x: v_pre_gram(params, x))
        return gram_matrix
    
    return gram

def nat_grad_factory(gram):

    def natural_gradient(params, tangent_params):

        gram_matrix = gram(params)
        flat_tangent, retriev_pytree  = jax.flatten_util.ravel_pytree(tangent_params)
        
        # solve gram dot flat_tangent.
        flat_nat_grad = lstsq(gram_matrix, flat_tangent)[0]

        # if gramian is zero then lstsq gives back nan...
        if jnp.isnan(flat_nat_grad[0]):
            return retriev_pytree(jnp.zeros_like(flat_nat_grad))

        else:
            return retriev_pytree(flat_nat_grad)

    return natural_gradient