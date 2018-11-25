"""
Classes for estimated transition model, to be used for tuning exploration.
"""
import sys
import pdb
import os
this_dir = os.path.dirname(os.path.abspath(__file__))
project_dir = os.path.join(this_dir, '..', '..')
sys.path.append(project_dir)
from scipy.stats import norm, multivariate_normal
import numpy as np
import pymc3 as pm
import src.estimation.density_estimation as dd
import matplotlib.pyplot as plt
from theano import shared


class GlucoseTransitionModel(object):
  FEATURE_INDICES_FOR_PARAMETRIC_MODEL = [0, 1, 2, 3, 7]
  COEF = np.array([10, 0.9, 0.1, -0.01, 0.0, 0.1, -0.01, -10, -4])
  SIGMA_GLUCOSE = 25

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

    # Covariates and subsequent glucoses; hang on to these after model is fit for plotting
    self.X_ = None
    self.food_nonzero = None
    self.activity_nonzero = None
    self.y_ = None

  def fit(self, X, y):
    self.X_ = X
    self.y_ = y
    self.food_nonzero = X[:, 3][np.where(X[:, 3]) != 0.0]
    self.activity_nonzero = X[:, 4][np.where(X[:, 4]) != 0.0]

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
      food = pm.sample_ppc(self.food_trace, model=self.food_model, samples=1, progressbar=False)['obs'][0, 0]

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
      glucose = pm.sample_ppc(self.trace[1], samples=1, model=self.model[1], progressbar=False)['obs'][0]
    else:
      glucose = pm.sample_ppc(self.trace, samples=1, model=self.model, progressbar=False)['obs'][0]

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

  def plot(self):
    """
    Plot some info associated with the posterior.
    :return:
    """
    # Posterior of food, activity densities; following https://docs.pymc.io/notebooks/dp_mix.html
    # Food
    # x_plot_food = np.linspace(np.min(self.X_[:, 3]), np.max(self.X_[:, 3]), 200)
    # post_food_pdf_contribs = norm.pdf(np.atleast_3d(x_plot_food),
    #                                   self.food_trace['mu'][:, np.newaxis, :],
    #                                   1.0 / np.sqrt(self.food_trace['lambda'] * self.food_trace['tau'])[:, np.newaxis, :])
    # post_food_pdfs = (self.food_trace['w'][:, np.newaxis, :] * post_food_pdf_contribs).sum(axis=-1)
    # # # plt.hist(self.food_nonzero)
    # plt.plot(x_plot_food, post_food_pdfs.T, c='gray', label='Posterior sample densities')
    # plt.plot(x_plot_food, post_food_pdfs.mean(axis=0), c='k', label='Posterior pointwise expected density')
    # plt.title('Posterior density estimates for food')
    # plt.show()
    # # Activity
    # x_plot_activity = np.linspace(np.min(self.X_[:, 4]), np.max(self.X_[:, 4]), 200)
    # post_activity_pdf_contribs = norm.pdf(np.atleast_3d(x_plot_activity),
    #                                   self.activity_trace['mu'][:, np.newaxis, :],
    #                                   1.0 / np.sqrt(self.activity_trace['lambda'] * self.activity_trace['tau'])[:, np.newaxis, :])
    # post_activity_pdfs = (self.activity_trace['w'][:, np.newaxis, :] * post_activity_pdf_contribs).sum(axis=-1)
    # # plt.hist(self.activity_nonzero)
    # plt.plot(x_plot_activity, post_activity_pdfs.T, c='gray', label='Posterior sample densities')
    # plt.plot(x_plot_activity, post_activity_pdfs.mean(axis=0), c='k', label='Posterior pointwise expected density')
    # plt.title('Posterior density estimates for activity')
    # plt.show()

    # Posterior of hyper- and hypo-glycemic densities with and without treatment
    # ToDo: Display true distributions, too
    # ToDo: Assuming method=np!
    x_plot = np.linspace(np.min(self.X_[:, 1]), np.max(self.X_[:, 1]), 200)

    # Test states
    hypoglycemic_0 = np.array([[1.0, 50, 0, 33, 50, 0, 0, 0, 0]])
    hypoglycemic_1 = np.array([[1.0, 50, 0, 33, 50, 0, 0, 1, 0]])
    hyperglycemic_0 = np.array([[1.0, 200, 0, 30, 200, 0, 0, 0, 0]])
    hyperglycemic_1 = np.array([[1.0, 200, 0, 30, 200, 0, 0, 1, 0]])

    # Conditional pdfs
    hypo0_pdfs = self.posterior_glucose_pdfs_at_x(hypoglycemic_0, x_plot)
    true_hypo0_pdf = self.true_glucose_pdf_at_x(hypoglycemic_0, x_plot)
    hypo1_pdfs = self.posterior_glucose_pdfs_at_x(hypoglycemic_1, x_plot)
    true_hypo1_pdf = self.true_glucose_pdf_at_x(hypoglycemic_1, x_plot)
    hyper0_pdfs = self.posterior_glucose_pdfs_at_x(hyperglycemic_0, x_plot)
    true_hyper0_pdf = self.true_glucose_pdf_at_x(hyperglycemic_0, x_plot)
    hyper1_pdfs = self.posterior_glucose_pdfs_at_x(hyperglycemic_1, x_plot)
    true_hyper1_pdf = self.true_glucose_pdf_at_x(hyperglycemic_1, x_plot)

    # Plot
    # Hypo
    plt.figure()
    plt.plot(x_plot, hypo0_pdfs.T, c='gray')
    plt.plot(x_plot, hypo0_pdfs.mean(axis=0), c='k')
    plt.plot(x_plot, hypo1_pdfs.T, c='cyan')
    plt.plot(x_plot, hypo1_pdfs.mean(axis=0), c='green')
    plt.show()

    # Hyper
    plt.figure()
    plt.plot(x_plot, hyper0_pdfs.T, c='gray')
    plt.plot(x_plot, hyper0_pdfs.mean(axis=0), c='k')
    plt.plot(x_plot, hyper1_pdfs.T, c='cyan')
    plt.plot(x_plot, hyper1_pdfs.mean(axis=0), c='green')
    plt.show()

  def true_glucose_pdf_at_x(self, x, x_grid):
    mu = np.dot(x, self.COEF)
    true_glucose_pdf = norm.pdf(x_grid, mu, self.SIGMA_GLUCOSE)
    return true_glucose_pdf

  def posterior_glucose_pdfs_at_x(self, x, x_grid):
    """
    For plottng posterior conditional glucose densities at a given feature vector x.

    :param x:
    :param x_grid: Grid of values at which to evaluate pdf
    :return:
    """
    # Get conditional means at each cluster
    mu = np.array([np.dot(x, t) for t in self.trace['theta'][:100]])
    sigma = 1.0 / np.sqrt(self.trace['tau'][:100])
    # mu = mu[:, np.newaxis, :]
    sigma = sigma[:, np.newaxis, :]
    post_glucose_pdf_contribs = norm.pdf(np.atleast_3d(x_grid), mu, sigma)
    v_ = np.array([norm.cdf(np.dot(x, b)) for b in self.trace['beta'][:100]])
    w_ = np.array([stick_breaking_for_probit_numpy_version(v_[i, 0, :]) for i in range(v_.shape[0])])
    w_ = w_[:, np.newaxis, :]
    post_glucose_pdfs = (w_ * post_glucose_pdf_contribs).sum(axis=-1)
    return post_glucose_pdfs


def transition_model_from_np_parameter(np_parameter):
  """

  :param np_parameter: Parameter corresponding to probit mixture of gaussians as in GlucoseTransitionModel np option.
  :return:
  """
  tau, beta, theta = \
    np_parameter['tau'], np_parameter['beta'], np_parameter['theta']

  def transition_model(x):
    # Draw cluster
    stick_breaking_weights = np.array([norm.cdf(np.dot(x.flatten(), beta_i)) for beta_i in beta.T])
    cluster_probs = stick_breaking_for_probit_numpy_version(stick_breaking_weights)
    cluster = np.random.choice(range(len(cluster_probs)), p=cluster_probs)
    theta_i = theta[cluster]
    s_mean = np.dot(theta_i, x)
    s_tilde = np.random.multivariate_normal(s_mean, cov=tau[cluster]*np.eye(len(s_mean)))
    return s_tilde

  return transition_model


def stick_breaking_for_probit_numpy_version(v):
  w = v * np.concatenate(([1.0], np.cumprod(1-v)[:-1]))
  w = np.concatenate((w[:-1], [1 - np.sum(w[:-1])]))  # Correct rounding error
  return w
