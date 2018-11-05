"""
Attempting to replicate dependent density regression example,
https://docs.pymc.io/notebooks/dependent_density_regression.html
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
from theano import shared, tensor as tt
from src.environments.Glucose import Glucose

SEED = 972915 # from random.org; for reproducibility
np.random.seed(SEED)


def norm_cdf(z):
    return 0.5 * (1 + tt.erf(z / np.sqrt(2)))


def stick_breaking(v):
  return v * tt.concatenate([tt.ones_like(v[:, :1]),
                             tt.extra_ops.cumprod(1 - v, axis=1)[:, :-1]],
                            axis=1)


def dependent_density_regression(X, y):
  n, p = X.shape
  K = 20
  X = np.column_stack((np.ones(n), X))
  # X_for_model = shared(X, broadcastable=(False, True))

  # Specify model
  with pm.Model() as model:
    # Dirichlet priors
    beta = pm.Normal('beta', 0.0, 5.0, shape=(p + 1, K))
    v = norm_cdf(tt.dot(X, beta))
    w = pm.Deterministic('w', stick_breaking(v))

  print('dirichlet prior')

  with model:
    # Linear model
    theta = pm.Normal('theta', 0.0, 10.0, shape=(p + 1, K))
    mu_ = pm.Deterministic('mu', tt.dot(X, theta))

  print('linear model')

  with model:
    tau = pm.Gamma('tau', 1.0, 1.0, shape=K)
    obs = pm.NormalMixture('obs', w, mu_, tau=tau, observed=y)

  print('ready to go')

  SAMPLES = 20000
  BURN = 10000

  with model:
    step = pm.Metropolis()
    trace = pm.sample(SAMPLES, step, chains=1, tune=BURN, random_seed=SEED)

  return


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

  X_ = np.random.multivariate_normal(np.zeros(3), np.eye(3), size=10)
  y_ = np.random.normal(np.zeros(10))
  # y_ = Sp1[:, 0]
  # Do the do
  dependent_density_regression(X_, y_)
