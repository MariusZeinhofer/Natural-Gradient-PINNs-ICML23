"""
ENGD Optimization.
Five dimensional Poisson equation example. Solution given by

u(x) = sum_{i=1}^5 sin(pi * x_i)

"""
import jax
import jax.numpy as jnp
from jax import random, grad, vmap, jit

from ngrad.domains import Hyperrectangle, HypercubeBoundary
from ngrad.models import mlp, init_params
from ngrad.integrators import EvolutionaryIntegrator
from ngrad.utility import laplace, grid_line_search_factory
from ngrad.inner import model_laplace, model_identity
from ngrad.gram import gram_factory, nat_grad_factory


jax.config.update("jax_enable_x64", True)

# random seed
seed = 0

# domains
dim = 5
interior = Hyperrectangle([(0., 1.) for _ in range(0, dim)])
boundary = HypercubeBoundary(dim)

# integrators
interior_integrator = EvolutionaryIntegrator(interior, key=random.PRNGKey(0), N=4000)
boundary_integrator = EvolutionaryIntegrator(boundary, key= random.PRNGKey(1), N=500)
eval_integrator = EvolutionaryIntegrator(interior, key=random.PRNGKey(0), N= 10 * 4000)

# model
activation = lambda x : jnp.tanh(x)
layer_sizes = [dim, 64, 1]
params = init_params(layer_sizes, random.PRNGKey(seed))
model = mlp(activation)
v_model = vmap(model, (None, 0))

# solution
@jit
def u_star(x):
    return (jnp.sum(jnp.sin(jnp.pi * x)))

v_u_star = vmap(u_star, (0))
v_grad_u_star = vmap(
    lambda x: jnp.dot(grad(u_star)(x), grad(u_star)(x))**0.5, (0)
    )

# rhs
@jit
def f(x):
    return jnp.pi**2 * u_star(x)

# gramians
gram_bdry = gram_factory(
    model = model,
    trafo = model_identity,
    integrator = boundary_integrator
)

gram_laplace = gram_factory(
    model = model,
    trafo = model_laplace,
    integrator = interior_integrator
)

@jit
def gram(params):
    return (gram_bdry(params) + gram_laplace(params))

# natural gradient
nat_grad = nat_grad_factory(gram)

# compute residual
laplace_model = lambda params: laplace(lambda x: model(params, x))
residual = lambda params, x: (laplace_model(params)(x) + f(x))**2.
v_residual =  jit(vmap(residual, (None, 0)))

# loss
@jit
def interior_loss(params):
    return 0.5 * interior_integrator(lambda x: v_residual(params, x))

@jit
def boundary_loss(params):
    return (
        0.5 * boundary_integrator(lambda x: (v_model(params, x) - v_u_star(x))**2.)
    )

@jit
def loss(params):
    return interior_loss(params) + boundary_loss(params)

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

norm_sol_l2 = l2_norm(v_u_star, eval_integrator)
norm_sol_h1 = norm_sol_l2 + l2_norm(v_grad_u_star, eval_integrator)    


# training loop
for iteration in range(201):
    grads = grad(loss)(params)
    nat_grads = nat_grad(params, grads)
    params, actual_step = ls_update(params, nat_grads)

    if iteration % 10 == 0 and iteration > 0:
        l2_error = l2_norm(v_error, eval_integrator)
        h1_error = l2_error + l2_norm(v_error_abs_grad, eval_integrator)

        print(
            f'NG Iteration: {iteration} with loss: '
            f'{loss(params)} L2 rel_error: {l2_error/norm_sol_l2} H1 rel_error: '
            f'{h1_error/norm_sol_h1}'
        )

    # draw new points -- this can slow down the optimization
    if iteration % 1 == 0:
        interior_integrator.new_rand_points()
        boundary_integrator.new_rand_points()