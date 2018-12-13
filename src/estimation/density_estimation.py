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
  sum_k2 = []  # \sum_{j != i} k2(x_i - x_j) ; used in both I_1, I_2

  I1_hat = 0.0
  I2_hat = 0.0
  for i in range(n):
    num_1_i = 0.0
    sum_k2_i = 0.0
    for j in range(n):
      if j != i:
        for k in range(n):
          if k != i:
            num_1_i += k2(X[i] - X[j]) * k2(X[i] - X[k]) * ksqrt2_h1(y[k] - y[j])
        sum_k2_i += k2(X[i] - X[j])
    I1_hat += (num_1_i / sum_k2_i**2) / n
    sum_k2.append(sum_k2_i)

  # Get I2_hat










def two_step_ckde_cv_error(X, y, b0, b1, b2):
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
  b0 = least_squares_np_regression_cv(X, y)  # ToDo: implement this

  # Step 2: select b1, b2 using two step CV method from https://www.ssc.wisc.edu/~bhansen/papers/ncde.pdf
  return


def two_step_ckde(X, y):
  """
  Frequentist conditional kernel density estimator from https://www.ssc.wisc.edu/~bhansen/papers/ncde.pdf

  :param X:
  :param y:
  :return:
  """
  def two_step_ckde(b0, b1, b2):
    """

    :param x:
    :param b0: Bandwidths
    :param b1:
    :param b2:
    :return:
    """
    # First step: ND conditional mean
    K_b0 = np.array([np.array([gaussian_kernel(x - x_i, b0) for x_i in X])
                     for x in X])
    conditional_mean_estimate = np.dot(K_b0, y) / np.sum(K_b0, axis=1)

    # Second step: density estimation
    e_hat = y - conditional_mean_estimate
    K_b2 = np.array([np.array([gaussian_kernel(x - x_i, b2) for x_i in X])
                     for x in X])
    K_b1 = np.array([np.array([gaussian_kernel(e - e_i, b1) for e_i in e_hat])
                     for e in e_hat])
    g_hat = np.dot(K_b1, K_b2) / np.sum(K_b2, axis=1)
    return
  return
