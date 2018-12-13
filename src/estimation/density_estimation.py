"""
Attempting to replicate dependent density regression example,
https://docs.pymc.io/notebooks/dependent_density_regression.html

Updating model after new observations:
https://stackoverflow.com/questions/48517719/how-to-update-observations-over-time-in-pymc3
https://github.com/pymc-devs/pymc3/blob/master/docs/source/notebooks/updating_priors.ipynb
"""
import sys
import pdb
import os
this_dir = os.path.dirname(os.path.abspath(__file__))
project_dir = os.path.join(this_dir, '..', '..')
sys.path.append(project_dir)

import pymc3 as pm
from statsmodels.nonparametric.kernel_regression import KernelReg
from sklearn.metrics.pairwise import pairwise_kernels
from sklearn.linear_model import LinearRegression
import numpy as np
import scipy as sp
import pandas as pd
import theano
from theano import shared, tensor as tt
from src.environments.Glucose import Glucose

SEED = 972915 # from random.org; for reproducibility
np.random.seed(SEED)
theano.config.compute_test_value = 'off'


def norm_cdf(z):
    return 0.5 * (1 + tt.erf(z / np.sqrt(2)))


def stick_breaking_for_probit(v):
  return v * tt.concatenate([tt.ones_like(v[:, :1]),
                             tt.extra_ops.cumprod(1 - v, axis=1)[:, :-1]],
                            axis=1)


def stick_breaking_for_unconditional(beta):
  portion_remaining = tt.concatenate([[1], tt.extra_ops.cumprod(1 - beta)[:-1]])
  return beta * portion_remaining


def normal_bayesian_regression(X, y, test=False):
  """

  :param X:
  :param y:
  :return:
  """
  n, p = X.shape.eval()
  with pm.Model() as model:
    tau = pm.Gamma('tau', 1, 1)
    # mu_ = pm.Deterministic('mu', tt.dot(X[:, :3], beta))
    lambda_ = pm.Gamma('lambda', 1, 1)
    beta = pm.Normal('beta', 0.0, tau*lambda_, shape=p)
    mu_ = pm.Deterministic('mu', tt.dot(X, beta))
    obs = pm.Normal('obs', mu_, tau=tau, observed=y)

  if not test:
    SAMPLES = 1000
    BURN = 100000
  else:
    SAMPLES = BURN = 1

  with model:
    # ToDo: different algo (conjugate?)
    # step = pm.NUTS()
    step = pm.Metropolis()
    trace = pm.sample(SAMPLES, step, chains=1, tune=BURN, random_seed=SEED)

  return model, trace


def gaussian_kernel(x, bandwidth):
  """
  Helper for density estimation.

  :param x:
  :param bandwidth:
  :return:
  """
  return np.exp(-np.dot(x, x) / bandwidth) / bandwidth







def dirichlet_mixture_regression(X, y, alpha_mean=0.0, test=False):
  n, p = X.shape.eval()
  K = 20
  # X = shared(np.column_stack((np.ones(n), X)))
  # X_for_model = shared(X, broadcastable=(False, True))
  # Specify model
  with pm.Model() as model:
    # Dirichlet priors
    # alpha = pm.Normal('alpha', alpha_mean, 5.0, shape=K)
    # beta = pm.Normal('beta', 0.0, 5.0, shape=(p, K))
    # v = norm_cdf(alpha + tt.dot(X, beta))
    alpha = pm.Gamma('alpha', 1.0, 1.0)
    beta = pm.Beta('beta', 1.0, alpha, shape=K)
    w = pm.Deterministic('w', stick_breaking_for_unconditional(beta))
    # v = norm_cdf(tt.dot(X, beta))
    # w = pm.Deterministic('w', stick_breaking_for_probit(v))

  print('dirichlet prior')

  with model:
    # Linear model
    tau = pm.Gamma('tau', 1.0, 1.0, shape=K)
    lambda_ = pm.Uniform('lambda', 0, 5, shape=K)
    theta = pm.Normal('theta', 0.0, tau=lambda_*tau, shape=(p, K))
    mu_ = pm.Deterministic('mu', tt.dot(X, theta))

  print('linear model')

  with model:
    # tau = pm.Gamma('tau', 1.0, 1.0, shape=K)
    obs = pm.NormalMixture('obs', w, mu_, tau=tau, observed=y)

  print('ready to go')

  # ToDo: can samples be 1 if we want multiple ppd samples??
  if not test:
    SAMPLES = 1000
    BURN = 100000
  else:
    SAMPLES = BURN = 1

  with model:
    # step = pm.NUTS()
    step = pm.Metropolis()
    # step = pm.HamiltonianMC()
    trace = pm.sample(SAMPLES, step, chains=1, tune=BURN, random_seed=SEED, init='adapt_diag')
    # trace = pm.sample(SAMPLES, chains=1, tune=BURN, random_seed=SEED, init='adapt_diag')
    # approx = pm.fit(n=30000, method=pm.ADVI())
    # trace = approx.sample(draws=SAMPLES)

  model.name = 'nonparametric'
  return model, trace


def np_density_estimation(X, test=False):
  """
  Density estimation of distribution of X using dirichlet mixture https://docs.pymc.io/notebooks/dp_mix.html.
  This is used for density estimation of Food and Activity, which are distributed as
    0 with probability 1 - p
    F with probability p, where F is some distribution
  So we estimate p and estimate density only for those X's which aren't 0.

  :param X: one-dimensional array of observations
  :return:
  """
  if test:
    BURN = SAMPLES = 1
  else:
    BURN = 10000
    SAMPLES = 1000

  # Estimate p
  p = np.mean(X != 0.0)

  # DP mixture density estimation hyperparameters
  K = 20
  X_nonzero = X[np.where(X != 0)]
  # n = X_nonzero.shape[0]

  with pm.Model() as model:
    alpha = pm.Gamma('alpha', 1.0, 1.0)
    beta = pm.Beta('beta', 1.0, alpha, shape=K)
    w = pm.Deterministic('w', stick_breaking_for_unconditional(beta))
    tau = pm.Gamma('tau', 1.0, 1.0, shape=K)
    lambda_ = pm.Uniform('lambda', 0, 5, shape=K)
    mu = pm.Normal('mu', 0, tau=lambda_ * tau, shape=K)
    obs = pm.NormalMixture('obs', w, mu, tau=lambda_ * tau, observed=X_nonzero)
    step = pm.Metropolis()
    trace = pm.sample(SAMPLES, step, chains=1, tune=BURN, random_seed=SEED)

  return model, trace, p


# Frequentist CDE
# Referring to https://www.ssc.wisc.edu/~bhansen/papers/ncde.pdf
def I1_and_I2_hat(X, y, h1, h2):
  """
  Helper function for CV bandwidth selection from pdf pg 6.

  :param X:
  :param y:
  :param K_h1:
  :param K_h2:
  :return:
  """
  n, p = X.shape
  k1 = lambda x: gaussian_kernel(x, h1)
  k2 = lambda x: gaussian_kernel(x, h2)
  ksqrt2_h1 = lambda x: gaussian_kernel(x, np.sqrt(2)*h1)

  I1_hat = 0.0
  I2_hat = 0.0
  for i in range(n):
    num_1_i = 0.0
    sum_k2_i = 0.0
    for j in range(n):
      if j != i:
        for k in range(n):
          if k != i:
            k2_ij = k2(X[i] - X[j])
            k1_ij = k1(y[i] - y[j])
            num_1_i += k2_ij * k2(X[i] - X[k]) * ksqrt2_h1(y[k] - y[j])
            num_2_i += k2_ij * k1_ij
        sum_k2_i += k2(X[i] - X[j])
    I1_hat += (num_1_i / sum_k2_i**2) / n
    I2_hat += (num_2_i / sum_k2_i) / n

  return I1_hat - 2*I2_hat


def least_squares_cv(X, y, b0):
  X_k = pairwise_kernels(X, metric="rbf", **{'gamma': 1 / b0})
  reg = LinearRegression()
  reg.fit(X_k, y)
  H = np.dot(X_k, np.dot(np.linalg.inv(np.dot(X_k.T, X_k)), X_k.T))
  loo_residual = (reg.predict(X_k) - y) / (1 - np.diag(H))
  loo_error = np.dot(loo_residual, loo_residual)
  return loo_error


def two_step_ckde_cv(X, y):
  """
  Compute loocv error for two step conditional kde with bandwidths b0, b1, b2 (see two_step_ckde for what these do).

  :param X:
  :param y:
  :param b0:
  :param b1:
  :param b2:
  :return:
  """
  # Do bandwidth selection in two steps
  # Step 1: select b0 with least-squares CV
  b0_bandwidth_grid = [0.01, 0.1, 1]
  b0 = None
  best_err = float("inf")
  for b0_ in b0_bandwidth_grid:
    err = least_squares_cv(X, y)
    if err < best_err:
      b0 = b0_

  # ToDo: target should be residuals from first-step estimator (y - m(x)), not y!
  # Step 2: select b1, b2 using two step CV method from https://www.ssc.wisc.edu/~bhansen/papers/ncde.pdf
  bandwidth_grid = [0.01, 0.1, 1]
  b1 = b2 = None
  best_err = float("inf")
  for b1_ in bandwidth_grid:
    for b2_ in bandwidth_grid:
      err = I1_and_I2_hat(X, y, b1_, b2_)
      if err < best_err:
        b1 = b1_
        b2 = b2_

  return b0, b1, b2


def two_step_ckde(X, y):
  """
  Frequentist conditional kernel density estimator from https://www.ssc.wisc.edu/~bhansen/papers/ncde.pdf

  :param X:
  :param y:
  :return:
  """
  # Select bandwidth with CV
  b0, b1, b2 = two_step_ckde_cv(X, y)

  # Fit estimator using chosen bandwidths
  # First step: ND conditional mean
  K_b0 = np.array([np.array([gaussian_kernel(x - x_i, b0) for x_i in X])
                   for x in X])
  conditional_mean_estimate = np.dot(K_b0, y) / np.sum(K_b0, axis=1)
  e_hat = y - conditional_mean_estimate

  # Return sampling function
  def sample_from_conditional_kde(x_):
    """

    :param x_: Covariates at which to sample from conditional density.
    :return:
    """
    # Get mixing weights
    K_b2 = np.array([gaussian_kernel(x_ - x_i, b2) for x_i in X])
    mixing_weights = K_b2 / np.sum(K_b2)

    # Sample normal from mixture
    mixture_component = np.random.choice(len(mixing_weights), p=mixing_weights)

    # Sample from normal with mean m(x) + e_i, where e_i is residual from first step
    m_x = None  # ToDo: implement function for getting conditional mean!
    y = np.random.normal(loc=m_x + e_hat[mixture_component], scale=b1)
    return y

  return sample_from_conditional_kde
