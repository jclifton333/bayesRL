#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 20 15:27:56 2018

@author: lili
"""
import sys
import pdb
import os
this_dir = os.path.dirname(os.path.abspath(__file__))
project_dir = os.path.join(this_dir, '..', '..')
sys.path.append(project_dir)

import copy
import numpy as np
import src.policies.linear_algebra as la
from abc import ABCMeta, abstractmethod

ABC = ABCMeta('ABC', (object, ), {'__slots__': ()})


class Bandit(ABC):   
  def __init__(self):
    self.U = []
    self.A = []
      
  def reset(self):
    self.U = []
    self.A = []

  def step(self, a):
    """
    return utility given an action and current context
    """
    u = self.reward_dbn(a)
    self.U = np.append(self.U, u)
    self.A = np.append(self.A, a)

    return u

  @abstractmethod
  def reward_dbn(self, a):
    pass
  

class MAB(Bandit):
  def __init__(self, list_of_reward_mus):
    Bandit.__init__(self)
    self.list_of_reward_mus = list_of_reward_mus
    self.number_of_actions = len(list_of_reward_mus)
    self.number_of_pulls = np.zeros(self.number_of_actions)
    self.estimated_means = np.zeros(self.number_of_actions)
    self.estimated_vars = np.zeros(self.number_of_actions)
    self.standard_errors = np.zeros(self.number_of_actions)
    self.draws_from_each_arm = [[]] * self.number_of_actions

    # Initial pulls
    for a in range(self.number_of_actions):
      for N in range(10):
        self.step(a)
    #   self.step(a)

  def step(self, a):
    u = super(MAB, self).step(a)
    self.number_of_pulls[a] += 1
    self.estimated_means[a] += (u - self.estimated_means[a]) / self.number_of_pulls[a]
    self.draws_from_each_arm[a] = np.append(self.draws_from_each_arm[a], u)
    std = np.std(self.draws_from_each_arm[a])
    self.estimated_vars[a] = std ** 2 
    self.standard_errors[a] = std / np.sqrt(self.number_of_pulls[a])
    return u

  def reset(self):
    super(MAB, self).reset()
    self.number_of_pulls = np.zeros(self.number_of_actions)
    self.estimated_means = np.zeros(self.number_of_actions)
    self.estimated_vars = np.zeros(self.number_of_actions)
    self.standard_errors = np.zeros(self.number_of_actions)
    self.draws_from_each_arm = [[]] * self.number_of_actions

  @abstractmethod
  def reward_dbn(self, a):
    pass

  def generate_mc_samples(self, mc_reps, time_horizon, reward_means=None, reward_vars=None):
    """

    :param mc_reps:
    :param time_horizon:
    :param reward_means: List of means for each arm, for each rep.  If None, use the ones in
                         this environment.
    :param reward_vars:
    :return:
    """

    if reward_means is not None and reward_vars is not None:
      normal_parameters_provided = True

      def reward_distribution(a, mean_list, var_list):
        mean = mean_list[a]
        var = var_list[a]
        return np.random.normal(loc=mean, scale=np.sqrt(var))
    else:
      normal_parameters_provided = False

      def reward_distribution(a, mean_list, var_list):
        return self.reward_dbn(a)

    results = []
    optimal_reward = np.max(self.list_of_reward_mus)
    for rep in range(mc_reps):
      if normal_parameters_provided:
        mean_list = reward_means[rep]
        var_list = reward_vars[rep]
      else:
        mean_list = None
        var_list = None

      each_rep_result = dict()
      initial_model = {'sample_mean_list': copy.copy(self.estimated_means),
                       'number_of_pulls': copy.copy(self.number_of_pulls),
                       'standard_error_list': copy.copy(self.standard_errors),
                       'sample_var_list': copy.copy(self.estimated_vars)}

      rewards = np.zeros((0, self.number_of_actions))
      regrets = np.zeros((0, self.number_of_actions))

      for t in range(time_horizon):
        rewards_t = np.array([reward_distribution(a, mean_list, var_list) for a in range(self.number_of_actions)])
        rewards = np.vstack((rewards, rewards_t))
        regrets_t = np.array([optimal_reward-self.list_of_reward_mus[a] for a in range(self.number_of_actions)])
        regrets = np.vstack((regrets, regrets_t))
        
      each_rep_result['rewards'] = rewards
      each_rep_result['regrets'] = regrets
      each_rep_result['initial_model'] = initial_model
      results = np.append(results, each_rep_result)
    return results


class NormalMAB(MAB):
  def __init__(self, list_of_reward_mus=[[1], [2]], list_of_reward_vars=[[1], [1]]):
    self.list_of_reward_vars = list_of_reward_vars

    # Hyperparameters for conjugate normal model of mean and variance
    self.lambda0 = 1.0 / 10.0  # lambda is inverse variance
    # self.alpha0 = 10e-3
    # self.beta0 = 10e-3
    self.alpha0 = 10e-3
    self.beta0 = 10e-3
    self.posterior_params_dict = {a: {'lambda_post': self.lambda0, 'alpha_post': self.alpha0, 'beta_post': self.beta0,
                                      'mu_post': 0.0} for a in range(len(list_of_reward_mus))}
    MAB.__init__(self, list_of_reward_mus)

  def sample_from_bootstrap(self):
    draws_dict = {}
    for a in range(self.number_of_actions):
      rewards_at_arm = self.draws_from_each_arm[a]
      mean_draw, var_draw = multiplier_bootstrap_mean_and_variance(rewards_at_arm)
      draws_dict[a] = {'mu_draw': mean_draw, 'var_draw': var_draw}
    return draws_dict

  def sample_from_posterior(self, variance_shrinkage=1.0):
    """
    Sample from normal-gamma posterior.

    :param variance_shrinkage: number <= 1 for tuning exploration.
    :return:
    """
    draws_dict = {}
    for a in range(self.number_of_actions):
      # Posterior parameters for this arm
      alpha_post = self.posterior_params_dict[a]['alpha_post']
      beta_post = self.posterior_params_dict[a]['beta_post']
      lambda_post = self.posterior_params_dict[a]['lambda_post']
      mu_post = self.posterior_params_dict[a]['mu_post']

      tau_draw = np.random.gamma(alpha_post, 1.0 / beta_post)
      mu_given_tau_draw = np.random.normal(mu_post, variance_shrinkage * np.sqrt(1.0 / (lambda_post * tau_draw)))
      draws_dict[a] = {'mu_draw': mu_given_tau_draw, 'var_draw': 1.0 / tau_draw}

    return draws_dict

  def conjugate_bayes_mean_and_variance(self, a, ipw_means=None):
    """
    Get bayes estimators of mean and variance from current list of rewards at each arm, and update.

    :param ipw_means: Array of mean estimates of length num_actions, or None. If provided, use these to calculate
    empirical mean and variance (and therefore posterior), rather than xbar.
    :return:
    """
    rewards_at_arm = np.array(self.draws_from_each_arm[a])  # ToDo: can be optimized by not calling array() every time.
    if ipw_means is None:
      xbar = np.mean(rewards_at_arm)
      s_sq = np.var(rewards_at_arm)
    else:
      xbar = ipw_means[a]
      s_sq = np.mean((rewards_at_arm - xbar)**2)
    n = self.number_of_pulls[a]
    post_mean = (n * xbar) / (self.lambda0 + n)
    post_alpha = self.alpha0 + n/2.0
    post_beta = self.beta0 + (1/2.0) * (n*s_sq + (self.lambda0 * n * xbar**2) / (self.lambda0 + n))
    post_lambda = self.lambda0 + n
    post_precision = post_alpha / post_beta
    post_var = 1 / post_precision
    self.estimated_means[a] = post_mean
    self.estimated_vars[a] = post_var

    # Update posterior params  (needed for sampling from posterior)
    self.posterior_params_dict[a]['lambda_post'] = post_lambda
    self.posterior_params_dict[a]['alpha_post'] = post_alpha
    self.posterior_params_dict[a]['beta_post'] = post_beta
    self.posterior_params_dict[a]['mu_post'] = post_mean

  def reward_dbn(self, a):
    # utility is distributed as Normal(mu, var)
    return np.random.normal(self.list_of_reward_mus[a], np.sqrt(self.list_of_reward_vars[a]))

  def step(self, a, ipw_means=None):
    u = super(NormalMAB, self).step(a)
     # Update means and variances with new posterior expectations
    self.conjugate_bayes_mean_and_variance(a, ipw_means=ipw_means)
    self.standard_errors[a] = np.sqrt(self.estimated_vars[a] / self.number_of_pulls[a])
    return {'Utility': u}


class BernoulliMAB(MAB):
  def __init__(self, list_of_reward_mus=[0.3, 0.7]):
    # Hyperparameters for conjugate normal model of mean and variance
    self.alpha0 = 1.0
    self.beta0 = 1.0
    self.posterior_params_dict = {a: {'alpha_post': self.alpha0, 'beta_post': self.beta0} for a in range(len(list_of_reward_mus))}

    MAB.__init__(self, list_of_reward_mus)
    
  def reward_dbn(self, a):
    # utility is distributed as Bernoulli(p)
    prob = self.list_of_reward_mus[a]
    u = np.random.binomial(n=1, p=prob) 
    return u  
  
  def sample_from_sampling_dbn(self):
    """
    Sample from sampling distribution, used for UCB.

    :return:
    """
    draws_dict = {}
    for a in range(self.number_of_actions):
      # Posterior parameters for this arm
      p_hat = float(sum(self.draws_from_each_arm[a]))/self.number_of_pulls[a]
      standard_errors = np.sqrt(p_hat*(1-p_hat)/self.number_of_pulls[a])
      p_draw = np.random.normal(loc=p_hat, scale=standard_errors)
      draws_dict[a] = {'mu_draw': p_draw}
    return draws_dict
  
  def sample_from_posterior(self, variance_shrinkage=1.0):
    """
    Sample from normal-gamma posterior.

    :param variance_shrinkage: number <= 1 for tuning exploration.
    :return:
    """
    draws_dict = {}
    for a in range(self.number_of_actions):
      # Posterior parameters for this arm
      alpha_post = self.posterior_params_dict[a]['alpha_post']
      beta_post = self.posterior_params_dict[a]['beta_post']

      p_draw = np.random.beta(alpha_post/variance_shrinkage, beta_post/variance_shrinkage)
      draws_dict[a] = {'mu_draw': p_draw}
    return draws_dict

  def conjugate_bayes_mean_and_variance(self, a):
    """
    Get bayes estimators of mean and variance from current list of rewards at each arm, and update.
    :return:
    """
    rewards_at_arm = self.draws_from_each_arm[a]
    xsum = sum(rewards_at_arm)
    n = self.number_of_pulls[a]
    post_alpha = self.alpha0 + xsum
    post_beta = self.beta0 + n - xsum
    post_p = post_alpha/(post_alpha + post_beta)
    
    self.estimated_means[a] = post_p
#    self.estimated_vars[a] = post_var

    # Update posterior params  (needed for sampling from posterior)
    self.posterior_params_dict[a]['alpha_post'] = post_alpha
    self.posterior_params_dict[a]['beta_post'] = post_beta
 
  def step(self, a):
    u = super(BernoulliMAB, self).step(a)
    self.conjugate_bayes_mean_and_variance(a)  # Update means and variances with new posterior expectations
    self.standard_errors[a] = np.sqrt(self.estimated_vars[a] / self.number_of_pulls[a])
    return {'Utility': u}
    
  def generate_mc_samples_bernoulli(self, mc_reps, time_horizon, reward_means=None):
    """

    :param mc_reps:
    :param time_horizon:
    :param reward_means: List of means for each arm, for each rep.  If None, use the ones in
                         this environment.
    :param reward_vars:
    :return:
    """
    optimal_reward = np.max(self.list_of_reward_mus)
    if reward_means is not None:
      normal_parameters_provided = True

      def reward_distribution(a, mean_list_):
        mean = mean_list_[a]
        return np.random.binomial(n=1, p=mean) 
    else:
      normal_parameters_provided = False

      def reward_distribution(a, mean_list_):
        return self.reward_dbn(a)

    results = []
    for rep in range(mc_reps):
      if normal_parameters_provided:
        mean_list = reward_means[rep]
      else:
        mean_list = None

      each_rep_result = dict()
      initial_model = {'sample_mean_list': copy.deepcopy(self.estimated_means),
                       'number_of_pulls': copy.deepcopy(self.number_of_pulls),
                       'standard_error_list': copy.deepcopy(self.standard_errors),
                       'sample_var_list': copy.deepcopy(self.estimated_vars)}

      rewards = np.zeros((0, self.number_of_actions))
      regrets = np.zeros((0, self.number_of_actions))

      for t in range(time_horizon):
        rewards_t = np.array([reward_distribution(a, mean_list) for a in range(self.number_of_actions)])
        rewards = np.vstack((rewards, rewards_t))
        regrets_t = np.array([optimal_reward-self.list_of_reward_mus[a] for a in range(self.number_of_actions)])
        regrets = np.vstack((regrets, regrets_t))

      each_rep_result['rewards'] = rewards
      each_rep_result['regrets'] = regrets
      each_rep_result['initial_model'] = initial_model
      results = np.append(results, each_rep_result)
    return results


class LinearCB(Bandit):
  """
  ToDo: transform + covariate-dependent-variance needs to be implemented!
  Contextual bandit where rewards are normal-linear in a transformation of the original features. Variance
  is in general covariate-dependent:
    log V(r | x, a) = feature_function(x) . \theta_a
  """
  def __init__(self, num_initial_pulls, feature_function, context_mean,
               list_of_reward_betas=[[0.4, 0.4, -0.4], [0.6, 0.6, -0.4]],
               list_of_reward_theats=[[0., 0.], [0., 0.]], list_of_reward_vars=[1, 1]):
    Bandit.__init__(self)
    ## list_of_reward_vars: a list of scalars
    ## context_mean: the mean vetor, same length as each vector in the list "list_of_reward_betas";
    ## context_var: the covariance matrix
    self.num_initial_pulls = num_initial_pulls
    self.context_mean = context_mean
    self.number_of_actions = len(list_of_reward_vars)
    self.context_dimension = len(list_of_reward_betas[0]) 
    self.curr_context = None
    self.feature_function = feature_function
    self.list_of_reward_betas = list_of_reward_betas
    self.list_of_reward_vars = list_of_reward_vars
    self.X = np.zeros((0, self.context_dimension))

    # Hyperparameters for normal-gamma model: https://en.wikipedia.org/wiki/Bayesian_linear_regression#Conjugate_prior_distribution
    self.a0 = 0.001
    self.b0 = 0.001
    self.lambda0 = 1.0 / 10.0
    self.Lambda_0 = self.lambda0 * np.eye(self.context_dimension)
    self.Lambda_inv0 = 1.0 / self.lambda0 * np.eye(self.context_dimension)
    self.posterior_params_dict = {a: {'a_post': self.a0, 'b_post': self.b0, 'Lambda_post': self.Lambda_0,
                                      'beta_post': np.zeros(self.context_dimension),
                                      'Lambda_inv_post': self.Lambda_inv0}
                                  for a in range(self.number_of_actions)}

    # For updating linear model estimates incrementally
    self.beta_hat_list = [None]*self.number_of_actions
    self.Xprime_X_inv_list = [None]*self.number_of_actions
    self.Xprime_X_list = [None]*self.number_of_actions
    self.X_list = [np.zeros((0, self.context_dimension)) for a in range(self.number_of_actions)]
    self.y_list = [np.zeros(0) for a in range(self.number_of_actions)]
    self.X_dot_y_list = [np.zeros(self.context_dimension) for a in range(self.number_of_actions)]
    self.sampling_cov_list = [None]*self.number_of_actions
    self.sigma_hat_list = [0.0]*self.number_of_actions
    self.max_X = -float("inf")

    # Get initial pulls and model; do this at initialization so we don't have to re-fit on initial obs every time
    self.initial_pulls()
    self.beta_hat_list_initial = self.beta_hat_list[:]
    self.Xprime_X_inv_list_initial = self.Xprime_X_inv_list[:]
    self.Xprime_X_list_initial = self.Xprime_X_list[:]
    self.X_list_initial = self.X_list[:]
    self.y_list_initial = self.y_list[:]
    self.X_dot_y_list_initial = self.X_dot_y_list[:]
    self.sampling_cov_list_initial = self.sampling_cov_list[:]
    self.sigma_hat_list_initial = self.sigma_hat_list[:]
    self.X_initial = copy.copy(self.X)
    self.estimated_context_mean = np.mean(self.X, axis=0)
    self.estimated_context_cov = np.cov(self.X, rowvar=False)

  def update_posterior(self, a, x, y):
    """
    Get bayes estimators of mean and variance from current list of rewards at each arm, and update.
    :return:
    """
    new_Lambda_inv = la.sherman_woodbury(self.posterior_params_dict[a]['Lambda_inv_post'], x, x)
    new_Lambda = self.posterior_params_dict[a]['Lambda_post'] + np.outer(x, x)
    beta_hat = self.beta_hat_list[a]
    Xprime_X = self.Xprime_X_list[a]
    new_beta = np.dot(new_Lambda_inv, np.dot(Xprime_X, beta_hat))
    n = self.X_list[a].shape[0]
    # new_a = self.a0 + (n / 2.0)
    # new_b = self.b0 + 0.5 * (np.sum(self.y_list[a]**2) + np.dot(new_beta, np.dot(new_Lambda, new_beta)))

    # Update posterior params  (needed for sampling from posterior)
    self.posterior_params_dict[a]['Lambda_inv_post'] = new_Lambda_inv
    self.posterior_params_dict[a]['Lambda_post'] = new_Lambda
    # self.posterior_params_dict[a]['a_post'] = new_a
    # self.posterior_params_dict[a]['b_post'] = new_b
    self.posterior_params_dict[a]['beta_post'] = new_beta

  def sample_from_posterior(self, variance_shrinkage=1.0):
    """
    Sample from normal-gamma posterior.
    :return:
    """
    draws_dict = {}
    for a in range(self.number_of_actions):
      # Posterior parameters for this arm
      # a_post = self.posterior_params_dict[a]['a_post']
      # b_post = self.posterior_params_dict[a]['b_post']
      Lambda_inv_post = self.posterior_params_dict[a]['Lambda_inv_post']
      beta_post = self.posterior_params_dict[a]['beta_post']

      # sigma_sq_draw = np.random.gamma(a_post, 1.0 / b_post)
      # Just draw from the damn sampling dbn of sigmasqhat
      sigma_sq_hat = self.sigma_hat_list[a]**2
      n = self.X_list[a].shape[0]
      sigma_sq_draw = np.random.gamma(n / 2.0, n / (2 * sigma_sq_hat))
      beta_give_sigma_sq_draw = np.random.multivariate_normal(beta_post,
                                                              variance_shrinkage * sigma_sq_draw * Lambda_inv_post)
      draws_dict[a] = {'beta_draw': beta_give_sigma_sq_draw, 'var_draw': sigma_sq_draw}
    return draws_dict

  def initial_pulls(self):
    """
    Pull each arm twice so we can fit the model.
    :return:
    """
    self.initial_context()
    for a in range(self.number_of_actions):
      for rep in range(self.num_initial_pulls):
        self.step(a)

  def update_linear_model(self, a, x_new, y_new):
    linear_model_results = la.update_linear_model(self.X_list[a], self.y_list[a], self.Xprime_X_list[a],
                                                  self.Xprime_X_inv_list[a], x_new, self.X_dot_y_list[a], y_new)
    self.beta_hat_list[a] = linear_model_results['beta_hat']
    self.y_list[a] = linear_model_results['y']
    self.X_list[a] = linear_model_results['X']
    self.Xprime_X_inv_list[a] = linear_model_results['Xprime_X_inv']
    self.X_dot_y_list[a] = linear_model_results['X_dot_y']
    self.sampling_cov_list[a] = linear_model_results['sample_cov']
    self.sigma_hat_list[a] = linear_model_results['sigma_hat']
    self.Xprime_X_list[a] = linear_model_results['Xprime_X']
    self.max_X = np.max((self.max_X, np.max(x_new)))

  def reset(self):
    super(LinearCB, self).reset()
    # Reset linear model info
    self.beta_hat_list = self.beta_hat_list_initial[:]
    self.Xprime_X_inv_list = self.Xprime_X_inv_list_initial[:]
    self.Xprime_X_list = self.Xprime_X_list_initial[:]
    self.X_list = self.X_list_initial[:]
    self.y_list = self.y_list_initial[:]
    self.X_dot_y_list = self.X_dot_y_list_initial[:]
    self.sampling_cov_list = self.sampling_cov_list_initial[:]
    self.sigma_hat_list = self.sigma_hat_list_initial[:]
    self.X = copy.copy(self.X_initial)
    self.max_X = -float("inf")

  def step(self, a):
    u = super(LinearCB, self).step(a)
    self.X = np.vstack((self.X, self.curr_context))
    self.update_linear_model(a, self.curr_context, u)
    self.update_posterior(a, self.curr_context, u)
    x = self.next_context()
    self.estimated_context_mean = np.mean(self.X, axis=0)
    self.estimated_context_cov = np.cov(self.X, rowvar=False)
    return {'Utility': u, 'New context': x}

  def generate_mc_samples(self, mc_reps, time_horizon, n_patients=10, gen_model_params=None):
    """

    :param mc_reps:
    :param time_horizon:
    :param gen_model_params: None, or list of dictionary with keys reward_betas, reward_vars, context_max,
      with lists corresponding to parameter values for each mc rep.
    :return:
    """
    if gen_model_params is not None:
      gen_model_params_provided = True

      def reward_distribution(a, context, beta_list, var_list):
        beta = beta_list[a]
        var = var_list[a]
        mean = np.dot(beta, context)
        u = mean + np.random.normal(scale=np.sqrt(var))
        return u

      def context_distribution(context_mean_, context_var_):
        context = np.random.multivariate_normal(context_mean_, context_var_ * np.eye(len(context_mean)))
        return context
    else:
      gen_model_params_provided = False

      def reward_distribution(a, context_, beta_list, var_list):
        return self.expected_reward(a, context_) + self.reward_noise(a)

      def context_distribution(context_mean_, context_var_):
        return self.draw_context()

    results = []
    for rep in range(mc_reps):
      if gen_model_params_provided:
        beta_list = gen_model_params[rep]['reward_betas']
        var_list = gen_model_params[rep]['reward_vars']
        context_mean = gen_model_params[rep]['context_mean']
        context_var = gen_model_params[rep]['context_var']
      else:
        beta_list = None
        var_list = None
        context_mean = context_var = None

      each_rep_result = dict()

      # Initial pulls
      # For updating linear model estimates incrementally
      beta_hat_list = [None] * self.number_of_actions
      Xprime_X_list= [None] * self.number_of_actions
      Xprime_X_inv_list = [None] * self.number_of_actions
      X_list = [np.zeros((0, self.context_dimension)) for a in range(self.number_of_actions)]
      y_list = [np.zeros(0) for a in range(self.number_of_actions)]
      X_dot_y_list = [np.zeros(self.context_dimension) for a in range(self.number_of_actions)]
      sampling_cov_list = [None] * self.number_of_actions
      sigma_hat_list = [0.0] * self.number_of_actions

      for action in range(self.number_of_actions):
        for p in range(3):
          context = context_distribution(context_mean, context_var)
          reward = self.reward_dbn(action)
          linear_model_results = la.update_linear_model(X_list[action], y_list[action],
                                                        Xprime_X_list[action], Xprime_X_inv_list[action],
                                                        context, X_dot_y_list[action], reward)
          beta_hat_list[action] = linear_model_results['beta_hat']
          y_list[action] = linear_model_results['y']
          X_list[action] = linear_model_results['X']
          Xprime_X_inv_list[action] = linear_model_results['Xprime_X_inv']
          X_dot_y_list[action] = linear_model_results['X_dot_y']
          sampling_cov_list[action] = linear_model_results['sample_cov']
          sigma_hat_list[action] = linear_model_results['sigma_hat']
          Xprime_X_list[action] = linear_model_results['Xprime_X']

      initial_linear_model = {'beta_hat_list': beta_hat_list, 'y_list': y_list, 'X_list': X_list,
                              'Xprime_X_inv_list': Xprime_X_inv_list, 'X_dot_y_list': X_dot_y_list,
                              'sampling_cov_list': sampling_cov_list, 'sigma_hat_list': sigma_hat_list,
                              'Xprime_X_list': Xprime_X_list}

      rewards = []
      regrets = []
      contexts = []
      estimated_context_mean = np.zeros(self.context_dimension)
      estimated_context_var = np.diag(np.ones(self.context_dimension))

      for t in range(time_horizon):
        rewards_block = np.zeros((n_patients, self.number_of_actions))
        regrets_block = np.zeros((n_patients, self.number_of_actions))
        contexts_block = np.zeros((n_patients, self.context_dimension))

        for patient in range(n_patients):
          context = self.next_context()
          contexts_block[patient, :] = context
          opt_expected_reward = self.optimal_expected_reward(context)
          rewards_block[patient, :] = np.array([reward_distribution(a, context, beta_list, var_list)
                                      for a in range(self.number_of_actions)])
          regrets_block[patient, :] = np.array([self.regret(a, context) for a in range(self.number_of_actions)])

        rewards.append(rewards_block)
        regrets.append(regrets_block)
        contexts.append(contexts_block)
        # estimated_context_mean += (context - estimated_context_mean)/(t+1)
        # estimated_context_var = np.cov(contexts, rowvar=False)

      each_rep_result['contexts'] = contexts
      each_rep_result['rewards'] = rewards
      each_rep_result['regrets'] = regrets   
      each_rep_result['estimated_context_mean'] = estimated_context_mean
      each_rep_result['estimated_context_var'] = estimated_context_var
      each_rep_result['initial_linear_model'] = initial_linear_model
      results = np.append(results, each_rep_result)      
    return results
      
  @abstractmethod
  def draw_context(self):
    pass

  def initial_context(self):
    x0 = self.draw_context()
    self.curr_context = x0
    return x0

  def regret(self, a, context):
    return self.expected_reward(a, context) - self.optimal_expected_reward(context)

  def expected_reward(self, a, context):
#    pdb.set_trace()
    return np.dot(context, self.list_of_reward_betas[a])

  def optimal_expected_reward(self, context):
    return np.max([np.dot(context, beta) for beta in self.list_of_reward_betas])

  def reward_noise(self, a):
    var = self.list_of_reward_vars[a]
    noise = np.random.normal(0.0, np.sqrt(var))
    return noise

  def reward_dbn(self, a):
    mean = self.expected_reward(a, self.curr_context)
    noise = self.reward_noise(a)
    u = mean + noise
    return u

  def next_context(self):
    x = self.draw_context()
    self.curr_context = x
    return x


class NormalCB(LinearCB):
  def __init__(self, num_initial_pulls, feature_function=None, list_of_reward_betas=[[1, 1], [2, -2]],
               list_of_reward_thetas=[[0., 0.], [0., 0.]], list_of_reward_vars=[1, 1],
               context_mean=[1, 0], context_var=np.array([[1., 0.1], [0.1, 1.]])):
    self.context_var = context_var

    # Context prior params
    self.a0_context = 0.001
    self.b0_context = 0.001
    self.lambda0_context = 1.0 / 10.0
    self.posterior_context_params_dict = {'mu_post_context': np.zeros(len(context_mean)),
                                          'a_post_context': self.a0_context, 'b_post_context': self.b0_context,
                                          'lambda_post_context': self.lambda0_context}

    LinearCB.__init__(self, num_initial_pulls, feature_function, context_mean, list_of_reward_betas,
                      list_of_reward_thetas, list_of_reward_vars)

  def draw_context(self, context_mean=None, context_var=None):
    if context_mean is None:
      context_mean = self.context_mean
    if context_var is None:
      context_var = self.context_var
    context = np.append([1.0], np.random.multivariate_normal(context_mean, context_var))
    if self.feature_function is None:
      return context
    else:
      return self.feature_function(context)

  def update_posterior(self, a, x, y):
    super(NormalCB, self).update_posterior(a, x, y)

    # Update context posterior
    xbar = np.mean(self.X, axis=0)
    s_sq = np.mean(np.var(self.X, axis=0))
    n = self.X.shape[0]
    post_mean = (n * xbar) / (self.lambda0 + n)
    a_post_context = self.a0_context + n/2.0
    b_post_context = self.b0_context + (1/2.0) * (n*s_sq + (self.lambda0 * n * xbar**2) / (self.lambda0_context + n))
    post_lambda_context = self.lambda0_context + n

    # Update posterior params  (needed for sampling from posterior)
    self.posterior_context_params_dict['lambda_post_context'] = post_lambda_context
    self.posterior_context_params_dict['a_post_context'] = a_post_context
    self.posterior_context_params_dict['b_post_context'] = b_post_context
    self.posterior_context_params_dict['mu_post_context'] = post_mean

  def sample_from_posterior(self, variance_shrinkage=1.0):
    draws_dict = super(NormalCB, self).sample_from_posterior(variance_shrinkage=variance_shrinkage)

    # Get posterior parameters
    a_post_context = self.posterior_context_params_dict['a_post_context']
    b_post_context = self.posterior_context_params_dict['b_post_context']
    mu_post_context = self.posterior_context_params_dict['mu_post_context']
    lambda_post_context = self.posterior_context_params_dict['lambda_post_context']

    # Sample from posterior
    tau_draw = np.random.gamma(a_post_context, 1.0 / b_post_context)
    mu_var = variance_shrinkage / (lambda_post_context * tau_draw) * np.eye(self.context_dimension)
    mu_given_tau_draw = np.random.multivariate_normal(mu_post_context, mu_var)
    draws_dict['context_mu_draw'] = mu_given_tau_draw
    draws_dict['context_var_draw'] = 1.0 / tau_draw
    return draws_dict

  def sample_from_sampling_dist(self, variance_shrinkage=1.0):
    draws_dict = super(NormalCB, self).sample_from_posterior(variance_shrinkage=variance_shrinkage)
    sample_mean = np.mean(self.X, axis=0)
    sample_var = np.mean(np.std(self.X, axis=0)**2)
    n = self.X.shape[0]
    sample_se = np.sqrt(sample_var/n)*np.eye(self.context_dimension)
    context_mean = np.random.multivariate_normal(sample_mean, sample_se)
    draws_dict['context_mean'] = context_mean
    return draws_dict


class NormalUniformCB(LinearCB):
  def __init__(self, list_of_reward_betas=[[0.1, -0.1], [0.2, 0.1]], context_mean=[1.0, 0.5], list_of_reward_vars=[[4], [4]],
               context_bounds=(0.0, 1.0)):
    self.context_bounds = context_bounds
    # Hyperparameters for uniform-pareto model
    # for contexts https://tminka.github.io/papers/minka-uniform.pdf
    self.b_pareto_0 = 1.0
    self.K0 = 1.0
    self.posterior_context_params_dict = {'K_post': self.K0,
                                          'b_pareto_post': self.b_pareto_0}
    LinearCB.__init__(self, context_mean, list_of_reward_betas, list_of_reward_vars)

  def update_posterior(self, a, x, y):
    super(NormalUniformCB, self).update_posterior(a, x, y)
    self.max_X = np.max((np.max(x), self.max_X))
    self.posterior_context_params_dict['K_post'] += len(x)
    self.posterior_context_params_dict['b_pareto_post'] = np.max((self.max_X,
                                                                  self.posterior_context_params_dict['b_pareto_post']))

  def sample_from_posterior(self, variance_shrinkage=1.0):
    # Sample from condtional model
    draws_dict = super(NormalUniformCB, self).sample_from_posterior(variance_shrinkage=variance_shrinkage)

    # Sample from context model
    K_post = self.posterior_context_params_dict['K_post']
    b_pareto_post = self.posterior_context_params_dict['b_pareto_post']
    context_max = K_post * np.random.pareto(a=b_pareto_post)
    draws_dict['context_max'] = context_max

    return draws_dict

  def draw_context(self, context_mean=None, context_var=None):
    x = np.random.uniform(low=self.context_bounds[0], high=self.context_bounds[1], size=self.context_dimension)
    return x


def multiplier_bootstrap_mean_and_variance(x):
  """
  Helper for bootstrapping bandit estimators.

  :param x:
  :return:
  """
  n = len(x)
  weights = np.random.exponential(size=n)
  weights_sum = np.sum(weights)
  bootstrapped_mean = np.dot(weights, x) / weights_sum
  bootstrapped_var = np.dot(weights, (x - bootstrapped_mean)**2) / weights_sum
  return bootstrapped_mean, bootstrapped_var
