"""
Contains implementation of a MLP, i.e., a fully connected model.

"""
import jax.numpy as jnp
from jax import random

# ------Code from Jax documentation-----
def random_layer_params(m, n, key, scale=1e-1):
  w_key, b_key = random.split(key)
  return scale * random.normal(w_key, (n, m)), scale * random.normal(b_key, (n,))

def init_params(sizes, key):
  keys = random.split(key, len(sizes))
  return [random_layer_params(m, n, k) for m, n, k in zip(sizes[:-1], sizes[1:], keys)]
# ----------Copy ends----------------------

def mlp(activation): 
    def model(params, inpt):
        hidden = inpt
        for w, b in params[:-1]:
            outputs = jnp.dot(w, hidden) + b
            hidden = activation(outputs)
  
        final_w, final_b = params[-1]
        return jnp.reshape(jnp.dot(final_w, hidden) + final_b, ())
    return model