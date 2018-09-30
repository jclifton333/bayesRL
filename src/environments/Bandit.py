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

  def step(self, a):
    u = super(MAB, self).step(a)
    self.number_of_pulls[a] += 1
    # self.estimated_means[a] += (u - self.estimated_means[a]) / self.number_of_pulls[a]
    self.draws_from_each_arm[a] = np.append(self.draws_from_each_arm[a], u)
    std = np.std(self.draws_from_each_arm[a])
    # self.estimated_vars[a] = std ** 2
    # self.standard_errors[a] = std / np.sqrt(self.number_of_pulls[a])
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
    for rep in range(mc_reps):
      if normal_parameters_provided:
        mean_list = reward_means[rep]
        var_list = reward_vars[rep]
      else:
        mean_list = None
        var_list = None

      each_rep_result = dict()
      initial_model = {'sample_mean_list': copy.copy(self.estimated_means),
                       'number_of_pulls': copy.copy(self.number_of_pulls)}

      rewards = np.zeros((0, self.number_of_actions))
      regrets = np.zeros((0, self.number_of_actions))

      for t in range(time_horizon):
        rewards_t = np.array([reward_distribution(a, mean_list, var_list) for a in range(self.number_of_actions)])
        rewards = np.vstack((rewards, rewards_t))

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
    self.alpha0 = 10e-3
    self.beta0 = 10e-3
    self.posterior_params_dict = {a: {'lambda_post': self.lambda0, 'alpha_post': self.alpha0, 'beta_post': self.beta0,
                                      'mu_post': 0.0} for a in range(len(list_of_reward_mus))}
    MAB.__init__(self, list_of_reward_mus)

  def sample_from_posterior(self):
    """
    Sample from normal-gamma posterior.
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
      mu_given_tau_draw = np.random.normal(mu_post, np.sqrt(1.0 / (lambda_post * tau_draw)))
      draws_dict[a] = {'mu_draw': mu_given_tau_draw, 'var_draw': 1.0 / tau_draw}

    return draws_dict

  def conjugate_bayes_mean_and_variance(self, a):
    """
    Get bayes estimators of mean and variance from current list of rewards at each arm, and update.
    :return:
    """
    rewards_at_arm = self.draws_from_each_arm[a]
    xbar = np.mean(rewards_at_arm)
    s_sq = np.var(rewards_at_arm)
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
    self.posterior_params_dict[a]['post_mean'] = post_mean

  def reward_dbn(self, a):
    # utility is distributed as Normal(mu, var)
    return np.random.normal(self.list_of_reward_mus[a], np.sqrt(self.list_of_reward_vars[a]))

  def step(self, a):
    u = super(NormalMAB, self).step(a)
    self.conjugate_bayes_mean_and_variance(a)  # Update means and variances with new posterior expectations
    self.standard_errors[a] = np.sqrt(self.estimated_vars[a] / self.number_of_pulls[a])
    return {'Utility': u}


class BernoulliMAB(MAB):
  def __init__(self, list_of_reward_mus=[0.3, 0.7]):
    MAB.__init__(self, list_of_reward_mus)

  def reward_dbn(self, a):
    # utility is distributed as Bernoulli(p)
    prob = self.list_of_reward_mus[a]
    u = np.random.binomial(n=1, p=prob) 
    return u    
      

class LinearCB(Bandit):
  def __init__(self, context_mean, list_of_reward_betas=[[1, 1], [2, -2]], list_of_reward_vars=[1, 1]):
    Bandit.__init__(self)
    ## list_of_reward_vars: a list of scalars
    ## context_mean: the mean vetor, same length as each vector in the list "list_of_reward_betas";
    ## context_var: the covariance matrix
    self.context_mean = context_mean
    self.number_of_actions = len(list_of_reward_vars)
    self.context_dimension = len(context_mean)
    self.curr_context = None
    self.list_of_reward_betas = list_of_reward_betas
    self.list_of_reward_vars = list_of_reward_vars
    self.X = np.zeros((0, self.context_dimension))

    # For updating linear model estimates incrementally
    self.beta_hat_list = [None]*self.number_of_actions
    self.Xprime_X_inv_list = [None]*self.number_of_actions
    self.X_list = [np.zeros((0, self.context_dimension)) for a in range(self.number_of_actions)]
    self.y_list = [np.zeros(0) for a in range(self.number_of_actions)]
    self.X_dot_y_list = [np.zeros(self.context_dimension) for a in range(self.number_of_actions)]
    self.sampling_cov_list = [None]*self.number_of_actions
    self.sigma_hat_list = [0.0]*self.number_of_actions

    # Get initial pulls and model; do this at initialization so we don't have to re-fit on initial obs every time
    self.initial_pulls()
    self.beta_hat_list_initial = self.beta_hat_list[:]
    self.Xprime_X_inv_list_initial = self.Xprime_X_inv_list[:]
    self.X_list_initial = self.X_list[:]
    self.y_list_initial = self.y_list[:]
    self.X_dot_y_list_initial = self.X_dot_y_list[:]
    self.sampling_cov_list_initial = self.sampling_cov_list[:]
    self.sigma_hat_list_initial = self.sigma_hat_list[:]
    self.X_initial = copy.copy(self.X)

  def initial_pulls(self):
    """
    Pull each arm twice so we can fit the model.
    :return:
    """
    self.initial_context()
    for a in range(self.number_of_actions):
      for rep in range(2):
        self.step(a)

  def update_linear_model(self, a, x_new, y_new):
    linear_model_results = la.update_linear_model(self.X_list[a], self.y_list[a], self.Xprime_X_inv_list[a], x_new,
                                                  self.X_dot_y_list[a], y_new)
    self.beta_hat_list[a] = linear_model_results['beta_hat']
    self.y_list[a] = linear_model_results['y']
    self.X_list[a] = linear_model_results['X']
    self.Xprime_X_inv_list[a] = linear_model_results['Xprime_X_inv']
    self.X_dot_y_list[a] = linear_model_results['X_dot_y']
    self.sampling_cov_list[a] = linear_model_results['sample_cov']
    self.sigma_hat_list[a] = linear_model_results['sigma_hat']

  def reset(self):
    super(LinearCB, self).reset()
    # Reset linear model info
    self.beta_hat_list = self.beta_hat_list_initial[:]
    self.Xprime_X_inv_list = self.Xprime_X_inv_list_initial[:]
    self.X_list = self.X_list_initial[:]
    self.y_list = self.y_list_initial[:]
    self.X_dot_y_list = self.X_dot_y_list_initial[:]
    self.sampling_cov_list = self.sampling_cov_list_initial[:]
    self.sigma_hat_list = self.sigma_hat_list_initial[:]
    self.X = copy.copy(self.X_initial)

  def step(self, a):
    u = super(LinearCB, self).step(a)
    self.X = np.vstack((self.X, self.curr_context))
    self.update_linear_model(a, self.curr_context, u)
    x = self.next_context()
    return {'Utility': u, 'New context': x}

  def generate_mc_samples(self, mc_reps, time_horizon):
    results = []
    for rep in range(mc_reps):
      each_rep_result = dict()

      # Initial pulls
      # For updating linear model estimates incrementally
      beta_hat_list = [None] * self.number_of_actions
      Xprime_X_inv_list = [None] * self.number_of_actions
      X_list = [np.zeros((0, self.context_dimension)) for a in range(self.number_of_actions)]
      y_list = [np.zeros(0) for a in range(self.number_of_actions)]
      X_dot_y_list = [np.zeros(self.context_dimension) for a in range(self.number_of_actions)]
      sampling_cov_list = [None] * self.number_of_actions
      sigma_hat_list = [0.0] * self.number_of_actions

      for action in range(self.number_of_actions):
        for p in range(3):
          context = self.draw_context()
          reward = self.reward_dbn(action)
          linear_model_results = la.update_linear_model(X_list[action], y_list[action], Xprime_X_inv_list[action],
                                                        context, X_dot_y_list[action], reward)
          beta_hat_list[action] = linear_model_results['beta_hat']
          y_list[action] = linear_model_results['y']
          X_list[action] = linear_model_results['X']
          Xprime_X_inv_list[action] = linear_model_results['Xprime_X_inv']
          X_dot_y_list[action] = linear_model_results['X_dot_y']
          sampling_cov_list[action] = linear_model_results['sample_cov']
          sigma_hat_list[action] = linear_model_results['sigma_hat']

      initial_linear_model = {'beta_hat_list': beta_hat_list, 'y_list': y_list, 'X_list': X_list,
                              'Xprime_X_inv_list': Xprime_X_inv_list, 'X_dot_y_list': X_dot_y_list,
                              'sampling_cov_list': sampling_cov_list, 'sigma_hat_list': sigma_hat_list}

      rewards = np.zeros((0, self.number_of_actions))
      regrets = np.zeros((0, self.number_of_actions))
      contexts = np.zeros((0, self.context_dimension))
      estimated_context_mean = np.zeros(self.context_dimension)
      estimated_context_var = np.diag(np.ones(self.context_dimension))

      for t in range(time_horizon):
        context = self.next_context()
        contexts = np.vstack((contexts, context))
        opt_expected_reward = self.optimal_expected_reward(context)
        rewards_t = np.array([self.expected_reward(a, context) + self.reward_noise(a) for a in range(self.number_of_actions)])
        rewards = np.vstack((rewards, rewards_t))
        estimated_context_mean += (context - estimated_context_mean)/(t+1)
        estimated_context_var = np.cov(contexts, rowvar=False)

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
  def __init__(self, list_of_reward_betas=[[1, 1], [2, -2]], list_of_reward_vars=[1, 1],
               context_mean=[1, 0], context_var=np.array([[1., 0.1], [0.1, 1.]])):
    self.context_var = context_var
    LinearCB.__init__(self, context_mean, list_of_reward_betas, list_of_reward_vars)

  def draw_context(self, context_mean=None, context_var=None):
    if context_mean is None:
      context_mean = self.context_mean
    if context_var is None:
      context_var = self.context_var  
    return np.random.multivariate_normal(context_mean, context_var)


class NormalUniformCB(LinearCB):
  def __init__(self, list_of_reward_betas=[[0.1, -0.1], [0.2, 0.1]], context_mean=[1.0, 0.5], list_of_reward_vars=[[4], [4]],
               context_bounds=(0.0, 1.0)):
    self.context_bounds=context_bounds
    LinearCB.__init__(self, context_mean, list_of_reward_betas, list_of_reward_vars)
    # self.context_dimension = 2

  def draw_context(self, context_mean=None, context_var=None):
    x = np.random.uniform(low=self.context_bounds[0], high=self.context_bounds[1], size=self.context_dimension-1)
    context = np.append([1.0], x)
    return context


