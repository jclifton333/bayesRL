"""
Classes for estimated transition model, to be used for tuning exploration.
"""
import sys
import pdb
import os
this_dir = os.path.dirname(os.path.abspath(__file__))
project_dir = os.path.join(this_dir, '..', '..')
sys.path.append(project_dir)
from scipy.stats import norm
import numpy as np
import pymc3 as pm
import src.estimation.density_estimation as dd
from theano import shared


class GlucoseTransitionModel(object):
  FEATURE_INDICES_FOR_PARAMETRIC_MODEL = [0, 1, 2, 3, 7]

  def __init__(self, method='np'):
    """

    :param method: string in ['np', 'p', 'averaged']
    """
    assert method in ['np', 'p', 'averaged']
    self.method = method

    self.glucose_model = None
    self.glucose_trace = None
    self.compare = None
    self.shared_x_np = None
    self.shared_x_p = None

    self.food_model = None
    self.food_trace = None
    self.food_nonzero_prob = None
    self.activity_model = None
    self.activity_trace = None
    self.activity_nonzero_prob = None
    self.shared_nonzero_food = None
    self.shared_nonzero_activity = None

  def fit(self, X, y):
    # Update shared features

    # Food and activity are modeled with np density estimation in all cases
    self.food_model, self.food_trace, self.food_nonzero_prob = dd.np_density_estimation(X[:, 3])
    self.activity_model, self.activity_trace, self.activity_nonzero_prob = dd.np_density_estimation(X[:, 4])

    if self.method == 'np':
      self.shared_x_np = shared(X)
      model_, trace_ = dd.dirichlet_mixture_regression(self.shared_x_np, y)
    elif self.method == 'p':
      self.shared_x_p = shared(X[:, self.FEATURE_INDICES_FOR_PARAMETRIC_MODEL])  # ToDo: Make sure these are the right indices!
      model_, trace_ = dd.normal_bayesian_regression(self.shared_x_p, y)
    elif self.method == 'averaged':
      self.shared_x_np = shared(X)
      self.shared_x_p = shared(X[:, self.FEATURE_INDICES_FOR_PARAMETRIC_MODEL])
      model_np, trace_np = dd.dirichlet_mixture_regression(self.shared_x_np, y)
      model_p, trace_p = dd.normal_bayesian_regression(self.shared_x_p, y)
      model_ = [model_p, model_np]
      trace_ = [trace_p, trace_np]
      self.compare = pm.compare({model_p: trace_p, model_np: trace_np}, method='BB-pseudo-BMA')

    self.model = model_
    self.trace = trace_
    # self.compare = compare_

  def draw_from_ppd(self, x):
    """

    :param x:
    :return:
    """
    food, activity = self.draw_from_food_and_activity_ppd()

    # If performing model averaging, need to compute weights and draw from mixed ppd
    # Sample from np or parametric model based on weights
    if self.method == 'averaged':
      weights_ = np.array(self.compare.weight.sort_index(ascending=True)).astype(float)
      ix_ = np.random.choice(len(weights_), p=weights_)
      self.shared_x_np.set_value(x)
      if ix_ == 1:
        glucose, r = self.draw_from_np_ppd(x)
      elif ix_ == 0:
        glucose, r = self.draw_from_parametric_ppd(x)

    # Sample from np model
    elif self.method == 'np':
      glucose, r = self.draw_from_np_ppd(x)

    # Sample from parametric model
    elif self.method == 'p':
      glucose, r = self.draw_from_parametric_ppd(x)

    s = np.array([glucose, food, activity])
    return s, r

  def draw_from_food_and_activity_ppd(self):
    # Draw food
    if np.random.random() < self.food_nonzero_prob:
      food = 0.0
    else:
      food = pm.sample_ppc(self.food_trace, model=self.food_model, progressbar=False)['obs'][0, 0]

    # Draw activity
    if np.random.random() < self.activity_nonzero_prob:
      activity = 0.0
    else:
      activity = pm.sample_ppc(self.activity_trace, model=self.activity_model, progressbar=False)['obs'][0, 0]

    return food, activity

  def draw_from_np_ppd(self, x):
    # Draw glucose
    self.shared_x_np.set_value(x)
    if self.method == 'averaged':
      glucose = pm.sample_ppc(self.trace[1], model=self.model[1], progressbar=False)['obs'][0]
    else:
      glucose = pm.sample_ppc(self.trace, model=self.model, progressbar=False)['obs'][0]

    r = self.reward_function(glucose)
    return glucose, r

  def draw_from_parametric_ppd(self, x):
    # ToDo: Still don't understand why sample_ppc doesn't return correct shape here
    # Draw glucose
    self.shared_x_p.set_value(x[:, self.FEATURE_INDICES_FOR_PARAMETRIC_MODEL])
    if self.method == 'averaged':
      glucose = pm.sample_ppc(self.trace[0], model=self.model[0], progressbar=False)['obs'][0, 0]
    else:
      glucose = pm.sample_ppc(self.trace, model=self.model, progressbar=False)['obs'][0, 0]

    r = self.reward_function(glucose)
    return glucose, r

  @staticmethod
  def reward_function(glucose):
    """

    :param glucose:
    :return:
    """
    # Reward from this timestep
    r1 = (glucose < 70) * (-0.005 * glucose ** 2 + 0.95 * glucose - 45) + \
         (glucose >= 70) * (-0.00017 * glucose ** 2 + 0.02167 * glucose - 0.5)
    return r1

  def cluster_trajectories(self, x, policy, time_horizon, n_draw=100):
    """
    Draw n_draw trajectories of length time_horizon, starting at state x, under policy, and cluster these.

    :param x:
    :param policy:
    :param time_horizon:
    :return:
    """
    pass


def transition_model_from_np_parameter(np_parameter):
  """

  :param np_parameter: Parameter corresponding to probit mixture of gaussians as in GlucoseTransitionModel np option.
  :return:
  """
  tau, beta, theta = \
    np_parameter['tau'], np_parameter['beta'], np_parameter['theta']

  def transition_model(x):
    # Draw cluster
    cluster_probs = np.array([norm.cdf(np.dot(x, beta_i)) for beta_i in beta])
    cluster = np.random.choice(range(len(cluster_probs)), p=cluster_probs)
    theta_i = theta[cluster]
    s_mean = np.dot(theta_i, x)
    s_tilde = np.random.multivariate_normal(s_mean, cov=tau[cluster]*np.eye(len(s_mean)))
    return s_tilde

  return transition_model

