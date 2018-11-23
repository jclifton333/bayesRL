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


def normal_bayesian_regression(X, y):
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

  SAMPLES = 200
  BURN = 10000
  # SAMPLES = BURN = 1

  with model:
    # ToDo: different algo (conjugate?)
    step = pm.NUTS()
    trace = pm.sample(SAMPLES, step, chains=1, tune=BURN, random_seed=SEED)

  return model, trace


def dirichlet_mixture_regression(X, y):
  n, p = X.shape.eval()
  K = 20
  # X = shared(np.column_stack((np.ones(n), X)))
  # X_for_model = shared(X, broadcastable=(False, True))

  # Specify model
  with pm.Model() as model:
    # Dirichlet priors
    # alpha = pm.Normal('alpha', 0.0, 5.0, shape=K)
    beta = pm.Normal('beta', 0.0, 5.0, shape=(p, K))
    v = norm_cdf(tt.dot(X, beta))
    # v = norm_cdf(alpha + tt.dot(X, beta))
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
  SAMPLES = 1000
  BURN = 10000
  # SAMPLES = BURN = 1

  with model:
    step = pm.Metropolis()
    trace = pm.sample(SAMPLES, step, chains=1, tune=BURN, random_seed=SEED)

  model.name = 'nonparametric'
  return model, trace


def np_density_estimation(X):
  """
  Density estimation of distribution of X using dirichlet mixture https://docs.pymc.io/notebooks/dp_mix.html.
  This is used for density estimation of Food and Activity, which are distributed as
    0 with probability 1 - p
    F with probability p, where F is some distribution
  So we estimate p and estimate density only for those X's which aren't 0.

  :param X: one-dimensional array of observations
  :return:
  """
  BURN = 10000
  SAMPLES = 1000

  # Estimate p
  p = np.mean(X != 0.0)

  # DP mixture density estimation hyperparameters
  K = 20
  X_nonzero = X[np.where(X != 0)]
  n = len(X_nonzero)
  alpha = 2.0

  with pm.Model() as model:
    beta = pm.Beta('beta', 1.0, alpha, shape=K)
    w = pm.Deterministic('w', stick_breaking_for_unconditional(beta))
    tau = pm.Gamma('tau', 1.0, 1.0, shape=K)
    mu = pm.Normal('mu', 0, tau=tau, shape=K)
    obs = pm.NormalMixture('obs', w, mu, tau=tau, observed=X_nonzero)
    step = pm.Metropolis()
    trace = pm.sample(SAMPLES, step, chains=2, tune=BURN, random_seed=SEED)

  return model, trace, p


def dependent_density_regression(X, y, X_p=None):
  """

  :param X:
  :param y:
  :param X_p: features for parametric model; if not None, will average np and parametric models.
  :return:
  """
  n, p = X.shape.eval()
  K = 20
  # X = shared(np.column_stack((np.ones(n), X)))
  # X_for_model = shared(X, broadcastable=(False, True))

  # Specify model
  with pm.Model() as model:
    # Dirichlet priors
    # alpha = pm.Normal('alpha', 0.0, 5.0, shape=K)
    beta = pm.Normal('beta', 0.0, 5.0, shape=(p, K))
    v = norm_cdf(tt.dot(X, beta))
    # v = norm_cdf(alpha + tt.dot(X, beta))
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
  SAMPLES = 200 
  BURN = 10000
  # SAMPLES = BURN = 1

  with model:
    step = pm.Metropolis()
    trace = pm.sample(SAMPLES, step, chains=1, tune=BURN, random_seed=SEED)

  model.name = 'nonparametric'

  if X_p is not None:
    model_p, trace_p = normal_bayesian_regression(X_p, y)
    model_p.name = 'parametric'
    compare_ = pm.compare({model_p: trace_p, model: trace}, method='BB-pseudo-BMA')
    models = [model, model_p]
    traces = [trace, trace_p]
  else:
    models = model
    traces = trace
    compare_ = None

  return models, traces, compare_


def stack_parametric_and_nonparametric_dependent_densities(X, y):
  model_p, trace_p = normal_bayesian_regression(X, y)
  model_np, trace_np = dependent_density_regression(X, y)

  compare_ = pm.compare({model_p: trace_p, model_np: trace_np}, method='BB-pseudo-BMA')
  combined_ppd = pm.sample_ppc_w([trace_p, trace_np], 1, [model_p, model_np],
                                 weights=compare_.weight.sort_index(ascending=True))

  return combined_ppd


# def posterior_predictive_transition(trace, model, shared_x_np, new_x, shared_x_p=None, compare_=None):
#   """
#   Sample from estimated transition density at x, using posterior predictive density as the estimated transition
#   density.
#
#   :param trace:
#   :param model:
#   :param shared_x:
#   :param new_x:
#   :param compare_: compare object (needed if multiple traces/models given!) or None
#   :return:
#   """
#   if compare_ is None:
#     shared_x_np.set_value(new_x)
#     shared_x_p = None
#     pp_sample = pm.sample_ppc(trace, model=model, samples=1)
#   else:
#     weights_ = np.array(compare_.weight.sort_index(ascending=True)).astype(float)
#     ix_ = np.random.choice(len(weights_), p=weights_)
#     shared_x_np.set_value(new_x)
#     shared_x_p.set_value(new_x[:3])
#     if ix_ == 0:
#       pp_sample = pm.sample_ppc(trace[ix_], model=model[ix_])['obs'][0]
#     elif ix_ == 1:
#       # ToDo: Still don't understand why sample_ppc doesn't return correct shape here
#       pp_sample = pm.sample_ppc(trace[ix_], model=model[ix_])['obs'][0, 0]
#     # pp_sample = pm.sample_ppc_w(traces=trace, samples=1, models=model,
#     #                             weights=compare_.weight.sort_index(ascending=True), size=1)
#   return pp_sample, shared_x_np, shared_x_p




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
  # m, t = dependent_density_regression(X_, y_)
  compare = stack_parametric_and_nonparametric_dependent_densities(X_, y_)

  # Posterior predictive transition
  # new_x_ = np.array([np.random.multivariate_normal(np.zeros(3), np.eye(3))])
  # posterior_predictive_transition(t, m, X_, new_x_)