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
try:
  import matplotlib.pyplot as plt
except:
  pass
from sklearn.ensemble import RandomForestRegressor
import src.estimation.bellman_error as be
import src.policies.simulation_optimization_policies as opt
from theano import shared


class GlucoseTransitionModel(object):
  FEATURE_INDICES_FOR_PARAMETRIC_MODEL = [0, 1, 2, 3, 8]
  COEF = np.array([10, 0.9, 0.1, -0.01, 0.0, 0.1, -0.01, -10, -4])
  SIGMA_GLUCOSE = 25

  def __init__(self, method='np', alpha_mean=0.0, test=False):
    """

    :param method: string in ['np', 'p', 'averaged']
    :param test: if True, this is just for testing - only draw one sample in MCMC
    """
    assert method in ['np', 'p', 'averaged']
    self.method = method
    self.alpha_mean = alpha_mean
    self.test = test

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
    self.food_nonzero = X[:, 2][np.where(X[:, 2] != 0.0)]
    self.activity_nonzero = X[:, 3][np.where(X[:, 3] != 0.0)]

    # Food and activity are modeled with np density estimation in all cases
    self.food_model, self.food_trace, self.food_nonzero_prob = dd.np_density_estimation(X[:, 2], test=self.test)
    self.activity_model, self.activity_trace, self.activity_nonzero_prob = dd.np_density_estimation(X[:, 3],
                                                                                                    test=self.test)

    if self.method == 'np':
      self.fit_np_conditional_density(X, y)
    elif self.method == 'p':
      self.shared_x_p = shared(X[:, self.FEATURE_INDICES_FOR_PARAMETRIC_MODEL])  # ToDo: Make sure these are the right indices!
      model_, trace_ = dd.normal_bayesian_regression(self.shared_x_p, y, test=self.test)
      self.model, self.trace = model_, trace_
    elif self.method == 'averaged':
      self.shared_x_np = shared(X)
      self.shared_x_p = shared(X[:, self.FEATURE_INDICES_FOR_PARAMETRIC_MODEL])
      model_np, trace_np = dd.dirichlet_mixture_regression(self.shared_x_np, y, test=self.test)
      model_p, trace_p = dd.normal_bayesian_regression(self.shared_x_p, y, test=self.test)
      model_ = [model_p, model_np]
      trace_ = [trace_p, trace_np]
      self.compare = pm.compare({model_p: trace_p, model_np: trace_np}, method='BB-pseudo-BMA')
      self.mode, self.trace = model_, trace_


    # self.compare = compare_

  def fit_np_conditional_density(self, X, y):
    self.shared_x_np = shared(X)
    model_, trace_ = dd.dirichlet_mixture_regression(self.shared_x_np, y, alpha_mean=self.alpha_mean,
                                                     test=self.test)
    self.model, self.trace = model_, trace_

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

    r = glucose_reward_function(glucose)
    return glucose, r

  def draw_from_parametric_ppd(self, x):
    # ToDo: Still don't understand why sample_ppc doesn't return correct shape here
    # Draw glucose
    # self.shared_x_p.set_value(x[:, self.FEATURE_INDICES_FOR_PARAMETRIC_MODEL])
    self.shared_x_p.set_value(np.array(x)[:, self.FEATURE_INDICES_FOR_PARAMETRIC_MODEL])
    if self.method == 'averaged':
      glucose = pm.sample_ppc(self.trace[0], model=self.model[0], progressbar=False)['obs'][0, 0]
    else:
      glucose = pm.sample_ppc(self.trace, model=self.model, progressbar=False)['obs'][0, 0]

    r = glucose_reward_function(glucose)
    return glucose, r

  def cluster_trajectories(self, x, policy, time_horizon, n_draw=100):
    """
    Draw n_draw trajectories of length time_horizon, starting at state x, under policy, and cluster these.

    :param x:
    :param policy:
    :param time_horizon:
    :return:
    """
    pass

  def plot_regression_line(self):
    """
    # ToDo: For np only!
    Plot posterior predictive line and 95% credible bands.

    :return:
    """
    # Get features at which to evaluate
    NUM_PPD_SAMPLES = 1000

    test_glucose = np.linspace(50, 200, 50)
    treat_test_features = np.array([[1.0, g, 50, 0, 33, 50, 0, 1, 0] for g in test_glucose])
    no_treat_test_features = np.array([[1.0, g, 50, 0, 33, 50, 0, 0, 0] for g in test_glucose])

    # From from ppd at each point
    treat_glucoses = np.zeros((50, NUM_PPD_SAMPLES))  # 100 ppd draws at each of 50 values
    no_treat_glucoses = np.zeros((50, NUM_PPD_SAMPLES))

    for ix, x in enumerate(treat_test_features):
      for draw in range(NUM_PPD_SAMPLES):
        g, r = self.draw_from_np_ppd([x])
        treat_glucoses[ix, draw] = g
    for ix, x in enumerate(no_treat_test_features):
      for draw in range(NUM_PPD_SAMPLES):
        g, r = self.draw_from_np_ppd([x])
        no_treat_glucoses[ix, draw] = g

    treat_glucoses_mean = treat_glucoses.mean(axis=1)
    no_treat_glucoses_mean = no_treat_glucoses.mean(axis=1)
    treat_glucoses_credible = np.percentile(treat_glucoses, [2.5, 97.5], axis=1).T  # 50x2 array [lower credible bound, upper credible bound]
    no_treat_glucoses_credible = np.percentile(no_treat_glucoses, [2.5, 97.5], axis=1).T

    # Plot
    plt.figure()
    plt.plot(test_glucose, treat_glucoses_mean, color='blue', label='Treatment')
    plt.fill_between(test_glucose, treat_glucoses_credible[:, 0], treat_glucoses_credible[:, 1], alpha=0.2,
                     label='95% PI for treatment curve')
    plt.plot(test_glucose, no_treat_glucoses_mean, color='green', label='No treatment')
    plt.fill_between(test_glucose, no_treat_glucoses_credible[:, 0], no_treat_glucoses_credible[:, 1], alpha=0.2,
                     label='95% PI for no treatment curve')
    plt.title('Conditional glucose\nalpha={}'.format(self.alpha_mean))
    plt.legend()
    plt_name = 'conditional-glucose-alpha={}.png'.format(self.alpha_mean)
    plt_name = os.path.join(project_dir, 'src', 'analysis', plt_name)
    plt.savefig(plt_name)
    plt.close()
    # plt.show()

  def plot_density_estimates(self):
    """
    Plot some info associated with the posterior.
    :return:
    """
    # Posterior of food, activity densities; following https://docs.pymc.io/notebooks/dp_mix.html
    # Food
    # f, axarr = plt.subplots(2)

    # x_plot_food = np.linspace(np.min(self.X_[:, 3]), np.max(self.X_[:, 2]), 200)
    # post_food_pdf_contribs = norm.pdf(np.atleast_3d(x_plot_food),
    #                                   self.food_trace['mu'][:, np.newaxis, :],
    #                                   1.0 / np.sqrt(self.food_trace['lambda'] * self.food_trace['tau'])[:, np.newaxis, :])
    # post_food_pdfs = (self.food_trace['w'][:, np.newaxis, :] * post_food_pdf_contribs).sum(axis=-1)
    # axarr[0].hist(self.food_nonzero, normed=True, alpha=0.5)
    # axarr[0].plot(x_plot_food, post_food_pdfs.T, c='gray', label='Posterior sample densities')
    # axarr[0].plot(x_plot_food, post_food_pdfs.mean(axis=0), c='k', label='Posterior pointwise expected density')
    # # axarr[0].title('Posterior density estimates for food')

    # # Activity
    # x_plot_activity = np.linspace(np.min(self.X_[:, 3]), np.max(self.X_[:, 2]), 200)
    # post_activity_pdf_contribs = norm.pdf(np.atleast_3d(x_plot_activity),
    #                                   self.activity_trace['mu'][:, np.newaxis, :],
    #                                   1.0 / np.sqrt(self.activity_trace['lambda'] * self.activity_trace['tau'])[:, np.newaxis, :])
    # post_activity_pdfs = (self.activity_trace['w'][:, np.newaxis, :] * post_activity_pdf_contribs).sum(axis=-1)
    # axarr[1].hist(self.activity_nonzero, normed=True, alpha=0.5)
    # axarr[1].plot(x_plot_activity, post_activity_pdfs.T, c='gray', label='Posterior sample densities')
    # axarr[1].plot(x_plot_activity, post_activity_pdfs.mean(axis=0), c='k', label='Posterior pointwise expected density')
    # axarr[1].title('Posterior density estimates for activity')
    # plt.show()

    # Posterior of hyper- and hypo-glycemic densities with and without treatment
    #  ToDo: Display true distributions, too
    # ToDo: Assuming method=np!
    x_plot = np.linspace(np.min(self.X_[:, 1]), np.max(self.X_[:, 1]), 200)

    # Test states
    hypoglycemic_0 = np.array([[1.0, 50, 0, 33, 50, 0, 0, 0, 0]])
    hypoglycemic_1 = np.array([[1.0, 50, 0, 33, 50, 0, 0, 1, 0]])
    hyperglycemic_0 = np.array([[1.0, 200, 0, 30, 200, 0, 0, 0, 0]])
    hyperglycemic_1 = np.array([[1.0, 200, 0, 30, 200, 0, 0, 1, 0]])

    # # Conditional pdfs
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
    plt_name = 'conditional-glucose-hypo-alpha={}.png'.format(self.alpha_mean)
    plt_name = os.path.join(project_dir, 'src', 'analysis', plt_name)
    plt.savefig(plt_name)
    plt.close()
    # plt.show()

    # Hyper
    plt.figure()
    plt.plot(x_plot, hyper0_pdfs.T, c='gray')
    plt.plot(x_plot, hyper0_pdfs.mean(axis=0), c='k')
    plt.plot(x_plot, hyper1_pdfs.T, c='cyan')
    plt.plot(x_plot, hyper1_pdfs.mean(axis=0), c='green')
    plt_name = 'conditional-glucose-hyper-alpha={}.png'.format(self.alpha_mean)
    plt_name = os.path.join(project_dir, 'src', 'analysis', plt_name)
    plt.savefig(plt_name)
    plt.close()
    # plt.show()

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

  def one_step_value_function_ppc(self, X, S, R):
    """
    Compare posterior predictive one-step value function to model-free one-step value function.
    :param X:
    :param R:
    :return:
    """
    NUM_PP_SAMPLES = 100
    NUM_MC_SAMPLES = 1000

    glucose_grid = np.linspace(50, 200, 100)
    s_mean = np.mean(np.vstack(S), axis=0)
    x_mean = np.mean(X, axis=0)
    grid = []  # Grid of s, x pairs at which to evaluate value functions
    for g in glucose_grid:
      sg = np.concatenate(([g], s_mean[1:]))
      xg = np.concatenate(([1, g], x_mean[2:]))
      grid.append((sg, xg))
    feature_function = opt.glucose_feature_function

    # Fit model-free value
    reg_mf = RandomForestRegressor()
    reg_mf.fit(X, R)
    v = lambda s_, x_: np.max([reg_mf.predict(feature_function(s_, a_, x_).reshape(1, -1)) for a_ in range(2)])
    v_mf_eval = [v(s, x) for s, x in grid]

    # Get ppd for model-based value
    v_mb_eval = []
    for _ in range(NUM_PP_SAMPLES):
      glucose_param = np.random.choice(self.trace)
      transition_model = transition_model_from_np_parameter(glucose_param, self.draw_from_food_and_activity_ppd)
      X_, S_, R_ = opt.simulate_from_transition_model(s_mean, x_mean, transition_model, 10, 2,
                                                      lambda s, x: np.random.binomial(1, 0.3), feature_function,
                                                      mc_rollouts=NUM_MC_SAMPLES)
      reg_ = RandomForestRegressor()
      reg_.fit(X_, R_)
      q_ = lambda x_: reg_.predict(x_.reshape(1, -1))
      v_ = lambda s_, x_: np.max([q_(feature_function(s_, a_, x_)) for a_ in range(2)])

      v_mb_eval.append([v_(s, x) for s, x in grid])

    # Plots
    plt.figure()
    plt.plot(glucose_grid, v_mf_eval, col='red', label='model free')
    plt.plot(glucose_grid, v_mb_eval, col='gray', label='ppd of model based values')
    plt.legend()
    plt.show()
    return v_mf_eval

  def bellman_error_weighted_np_posterior_expectation(self, q, S_ref, A_ref, tau=1):
    """
    Weight posterior by exp[ -tau/2 * BE(q, \theta) ], where \theta is parameter, q is q function.

    :param q:
    :param tau:
    :return:
    """
    gamma = 0.9
    reward_function = lambda s: glucose_reward_function(s[0])
    posterior_expectation = None
    n = 1
    for param in self.trace:
      transition_model = transition_model_from_np_parameter(param, self.draw_from_food_and_activity_ppd)
      bellman_error = be.transition_distribution_bellman_error(q, transition_model, S_ref, A_ref, feature_function,
                                                               reward_function, gamma, 2)
      exp_bellman_error = np.exp(-tau/2 * bellman_error)
      posterior_expectation += (param * exp_bellman_error - posterior_expectation) / n
      n += 1
    return posterior_expectation


# Helpers
def transition_model_from_np_parameter(glucose_parameter, food_and_activity_model):
  """
  Fix glucose transition model at a certain parameter value, and use food and activity ppds.

  :param glucose_parameter:
  :param food_and_activity_model:
  :return:
  """
  tau, beta, theta = \
    glucose_parameter['tau'], glucose_parameter['beta'], glucose_parameter['theta']

  def transition_model(x):
    # Draw glucose from conditional mixture
    stick_breaking_weights = np.array([norm.cdf(np.dot(x.flatten(), beta_i)) for beta_i in beta.T])
    cluster_probs = stick_breaking_for_probit_numpy_version(stick_breaking_weights)
    cluster = np.random.choice(range(len(cluster_probs)), p=cluster_probs)
    theta_i = theta.T[cluster]
    g_mean = np.dot(x.flatten(), theta_i)
    g_tilde = np.random.normal(g_mean, 1 / np.sqrt(tau.T[cluster]))

    # Draw food and activity
    f_tilde, a_tilde = food_and_activity_model()
    return np.array([g_tilde, f_tilde, a_tilde]), glucose_reward_function(g_tilde)

  return transition_model


def stick_breaking_for_probit_numpy_version(v):
  w = v * np.concatenate(([1.0], np.cumprod(1-v)[:-1]))
  w = np.concatenate((w[:-1], [1 - np.sum(w[:-1])]))  # Correct rounding error
  return w


def glucose_reward_function(glucose):
    """

    :param glucose:
    :return:
    """
    # Reward from this timestep
    r1 = (glucose < 70) * (-0.005 * glucose ** 2 + 0.95 * glucose - 45) + \
         (glucose >= 70) * (-0.00017 * glucose ** 2 + 0.02167 * glucose - 0.5)
    return r1

