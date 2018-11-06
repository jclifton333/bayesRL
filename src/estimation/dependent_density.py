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
import pandas as pd
import theano
from theano import shared, tensor as tt
from src.environments.Glucose import Glucose

SEED = 972915 # from random.org; for reproducibility
np.random.seed(SEED)
theano.config.compute_test_value = 'off'


def norm_cdf(z):
    return 0.5 * (1 + tt.erf(z / np.sqrt(2)))


def stick_breaking(v):
  return v * tt.concatenate([tt.ones_like(v[:, :1]),
                             tt.extra_ops.cumprod(1 - v, axis=1)[:, :-1]],
                            axis=1)


def dependent_density_regression(X, y):
  n, p = X.shape.eval()
  K = 20
  # X = shared(np.column_stack((np.ones(n), X)))
  # X_for_model = shared(X, broadcastable=(False, True))

  # Specify model
  with pm.Model() as model:
    # Dirichlet priors
    beta = pm.Normal('beta', 0.0, 5.0, shape=(p, K))
    v = norm_cdf(tt.dot(X, beta))
    w = pm.Deterministic('w', stick_breaking(v))

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

  SAMPLES = 1
  BURN = 10000
  # BURN = 1

  with model:
    step = pm.Metropolis()
    trace = pm.sample(SAMPLES, step, chains=1, tune=BURN, random_seed=SEED)

  return model, trace


def posterior_predictive_transition(trace, model, shared_x, new_x):
  """
  Sample from estimated transition density at x, using posterior predictive density as the estimated transition
  density.

  :param trace:
  :param model:
  :param shared_x:
  :param new_x:
  :return:
  """
  shared_x.set_value(new_x)
  pp_sample = pm.sample_ppc(trace, model=model, samples=1)
  return pp_sample, shared_x




# def main():
#   DATA_URI = 'http://www.stat.cmu.edu/~larry/all-of-nonpar/=data/lidar.dat'
#
#   def standardize(x):
#     return (x - x.mean()) / x.std()
#
#   df = (pd.read_csv(DATA_URI, sep=' *', engine='python')
#         .assign(std_range=lambda df: standardize(df.range),
#                 std_logratio=lambda df: standardize(df.logratio)))
#
#   N, _ = df.shape
#   K = 20
#
#   std_range = df.std_range.values[:, np.newaxis]
#   std_logratio = df.std_logratio.values[:, np.newaxis]
#
#   x_lidar = shared(std_range, broadcastable=(False, True))
#
#   with pm.Model() as model:
#     alpha = pm.Normal('alpha', 0., 5., shape=K)
#     beta = pm.Normal('beta', 0., 5., shape=K)
#     v = norm_cdf(alpha + beta * x_lidar)
#     w = pm.Deterministic('w', stick_breaking(v))
#
#   print('defined dirichlet priors')
#
#   with model:
#     gamma = pm.Normal('gamma', 0., 10., shape=K)
#     delta = pm.Normal('delta', 0., 10., shape=K)
#     mu = pm.Deterministic('mu', gamma + delta * x_lidar)
#
#   print('defined lm')

#   with model:
#     tau = pm.Gamma('tau', 1., 1., shape=K)
#     obs = pm.NormalMixture('obs', w, mu, tau=tau, observed=std_logratio)
#
#   SAMPLES = 20000
#   BURN = 10000
#
#   with model:
#     step = pm.Metropolis()
#     trace = pm.sample(SAMPLES, step, chains=1, tune=BURN, random_seed=SEED)

#   return

if __name__ == '__main__':
  # n_patients = 20
  # env = Glucose(nPatients=n_patients)

  # # Take random actions to get some data
  # env.step(np.random.choice(2, size=n_patients))
  # X_, Sp1 = env.get_state_transitions_as_x_y_pair()

  X_ = shared(np.random.multivariate_normal(np.zeros(3), np.eye(3), size=10))
  y_ = np.random.normal(np.zeros(10))
  # y_ = Sp1[:, 0]

  # Fit model
  m, t = dependent_density_regression(X_, y_)

  # Posterior predictive transition
  new_x_ = np.array([np.random.multivariate_normal(np.zeros(3), np.eye(3))])
  posterior_predictive_transition(t, m, X_, new_x_)
