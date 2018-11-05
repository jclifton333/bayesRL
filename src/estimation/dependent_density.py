"""
Attempting to replicate dependent density regression example,
https://docs.pymc.io/notebooks/dependent_density_regression.html
"""

import pymc3 as pm
import numpy as np
import pandas as pd
from theano import shared, tensor as tt

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
  X_for_model = shared(X, broadcastable=(False, True))

  # Specify model
  with pm.Model() as model:
    # Dirichlet priors
    beta = pm.Normal('beta', mu=np.zeros(p + 1), sd=5.0*np.ones(p + 1), shape=K)
    v = norm_cdf(np.dot(X_for_model, beta))
    w = pm.Deterministic('w', stick_breaking(v))

    # Linear model
    theta = pm.Normal('theta', mu=np.zeros(p + 1), sd=10.0*np.ones(p + 1), shape=K)
    mu_ = pm.Deterministic('mu', np.dot(X_for_model, theta))

    tau = pm.Gamma('tau', 1.0, 1.0, shape=K)
    obs = pm.NormalMixture('obs', w, mu_, tau=tau, observed=y)

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
  main()
