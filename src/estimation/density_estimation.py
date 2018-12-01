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
    beta = pm.Normal('beta', 0.0, 5.0, shape=3)
    tau = pm.Gamma('tau', 0.001, 0.001, shape=1)
    # mu_ = pm.Deterministic('mu', tt.dot(X[:, :3], beta))
    mu_ = pm.Deterministic('mu', tt.dot(X[:, :3], beta))
    obs = pm.Normal('obs', mu_, tau=tau, observed=y)

  if not test:
    SAMPLES = 1000
    BURN = 10000
  else:
    SAMPLES = BURN = 1

  with model:
    # ToDo: different algo (conjugate?)
    step = pm.NUTS()
    trace = pm.sample(SAMPLES, step, chains=1, tune=BURN, random_seed=SEED)

  return model, trace


def dirichlet_mixture_regression(X, y, alpha_mean=0.0, test=False):
  n, p = X.shape.eval()
  K = 20
  # X = shared(np.column_stack((np.ones(n), X)))
  # X_for_model = shared(X, broadcastable=(False, True))
  # Specify model
  with pm.Model() as model:
    # Dirichlet priors
    alpha = pm.Normal('alpha', alpha_mean, 5.0, shape=K)
    beta = pm.Normal('beta', 0.0, 5.0, shape=(p, K))
    # v = norm_cdf(tt.dot(X, beta))
    v = norm_cdf(alpha + tt.dot(X, beta))
    w = pm.Deterministic('w', stick_breaking_for_probit(v))

  print('dirichlet prior')

  with model:
    # Linear model
    theta = pm.Normal('theta', 0.0, 10.0, shape=(p, K))
    mu_ = pm.Deterministic('mu', tt.dot(X, theta))

  print('linear model')

  with model:
    tau = pm.Gamma('tau', 1.0, 1.0, shape=K)
    obs = pm.NormalMixture('obs', w, mu_, tau=tau, observed=y)

  print('ready to go')

  # ToDo: can samples be 1 if we want multiple ppd samples??
  if not test:
    SAMPLES = 1000
    BURN = 10000
  else:
    SAMPLES = BURN = 1

  with model:
    step = pm.Metropolis()
    trace = pm.sample(SAMPLES, step, chains=1, tune=BURN, random_seed=SEED)

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

