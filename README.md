**Achieving High Accuracy with PINNs via Energy Natural Gradient Descent ICML 2023**

Topic: *Github Repo for the ICML 2023 paper: Energy Natural gradient methods for PINNs and Deep Ritz.*

Paper: [arxiv](https://arxiv.org/abs/2302.13163)

Abstract: *We propose energy natural gradient descent, a natural gradient method with respect to a Hessian-induced Riemannian metric as an optimization algorithm for physics-informed neural networks (PINNs) and the deep Ritz method. As a main motivation we show that the update direction in function space resulting from the energy natural gradient corresponds to the Newton direction modulo an orthogonal projection onto the model's tangent space. We demonstrate experimentally that energy natural gradient descent yields highly accurate solutions with errors several orders of magnitude smaller than what is obtained when training PINNs with standard optimizers like gradient descent, Adam or BFGS, even when those are allowed significantly more computation time. We show that the approach can be combined with deterministic and stochastic discretizations of the integral terms and with deep networks allowing for an application in higher dimensional settings.*

## Requirements
- python 3.10.10 or later
- jax 0.4.8 or later
- jaxopt 0.6 or later
- optax 0.1.4 or later

## Installation
All required packages (jax, jaxopt, optax) are pip installable. For GPU support the jax version must be compatible with the CUDA version and the NVIDIA driver installed on the machine.

## Usage
The four main examples of the paper (Poisson 2d, Poisson 5d, Heat 1d, Nonlinear 1d) can be found in the top level scripts with the corresponding names. If the required packages are installed these scripts can directly be executed. The prefix of the scripts indicates which solver is used, with engd corresponding to the energy natural gradient descent. For brevity, the Adam, gradient descent (gd) and BFGS optimizers are only included for the 2d Poisson example. 