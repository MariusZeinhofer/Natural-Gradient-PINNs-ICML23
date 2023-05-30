"""
ENGD Optimization.
Minimization of the variational energy over integration domain (-1,1)

E(v) = 1/2 int |nabla(v)|^2 + 1/4 v^4 dx - int f v dx

where f = pi**2 * cos(x) + cos(x)**3.

The solution is u(x) = cos(x).

"""
import jax
import jax.numpy as jnp
from jax import random, grad, vmap, jit
from jax.flatten_util import ravel_pytree

from ngrad.models import mlp, init_params
from ngrad.domains import Interval
from ngrad.integrators import TrapezoidalIntegrator
from ngrad.inner import model_del_i_factory
from ngrad.gram import gram_factory, nat_grad_factory
from ngrad.utility import grid_line_search_factory

jax.config.update("jax_enable_x64", True)

# random seed
seed = 0

# integration
interval = Interval(-1., 1.)
integrator = TrapezoidalIntegrator(interval, 20000, K=4)
eval_integrator = TrapezoidalIntegrator(interval, 200000, K=4)

# model
activation = lambda x : jnp.tanh(x)
layer_sizes = [1, 32, 1]
params = init_params(layer_sizes, random.PRNGKey(seed))
model = mlp(activation)
v_model = vmap(model, (None, 0))

# solution and right-hand side
u_star = lambda x: jnp.reshape(jnp.cos(jnp.pi * x), ())
f = lambda x: (jnp.pi**2) * u_star(x) + u_star(x)**3

# inner product
def model_nonlinear(u_theta, del_theta_u):
    """
    Trafo for the u_theta dependent inner product coming from
    a(u_theta; v, w) = \int 3 * u_theta^2 v w dx 

    """
    def g_unravel(x):
        del_theta_u_flat, unravel = ravel_pytree(del_theta_u(x))
        nonlinear_flat = jnp.sqrt(3.) * u_theta(x) * del_theta_u_flat
        return unravel(nonlinear_flat)

    return g_unravel

# gramians
model_gradient = model_del_i_factory()
gram_grad = gram_factory(
    model = model,
    trafo = model_gradient,
    integrator = integrator,
)

gram_nonlinear = gram_factory(
    model = model,
    trafo = model_nonlinear,
    integrator = integrator,
)

@jit
def gram(params):
    return gram_grad(params) + gram_nonlinear(params)

nat_grad = nat_grad_factory(gram)

# loss function
@jit
def loss_gradient(params):
    grad_model = vmap(grad(lambda x: model(params, x)), (0))
    grad_squared = lambda x: 0.5 * jnp.reshape(grad_model(x)**2, (len(x)))
    return integrator(grad_squared)

@jit
def loss_lower_order_term(params):
    model_to_four = lambda x: 0.25 * v_model(params, x)**4
    return integrator(model_to_four)

@jit
def loss_rhs(params):
    rhs = lambda x: vmap(f, (0))(x) * v_model(params, x)
    return integrator(rhs)

@jit 
def loss(params):
    return loss_gradient(params) + loss_lower_order_term(params) - loss_rhs(params)

# set up grid line search
grid = jnp.linspace(0, 30, 31)
steps = 0.5**grid
ls_update = grid_line_search_factory(loss, steps)    

# errors
error = lambda x: model(params, x) - u_star(x)
v_error = vmap(error, (0))
v_error_abs_grad = vmap(
        lambda x: jnp.dot(grad(error)(x), grad(error)(x))**0.5
        )

def l2_norm(f, integrator):
    return integrator(lambda x: (f(x))**2)**0.5    

# training loop
for iteration in range(101):
    grads = grad(loss)(params)
    nat_grads = nat_grad(params, grads)
    params, actual_step = ls_update(params, nat_grads)

    if iteration % 50 == 0:
        l2_error = l2_norm(v_error, eval_integrator)
        h1_error = l2_error + l2_norm(v_error_abs_grad, eval_integrator)

        print(
            f'Seed: {seed} ENGD Iteration: {iteration} with loss: '
            f'{loss(params)} with error L2: {l2_error} and error H1: '
            f'{h1_error} and step: {actual_step}'
        )

