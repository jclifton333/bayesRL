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

  def step(self, a):
    u = super(MAB, self).step(a)
    self.number_of_pulls[a] += 1
    self.estimated_means[a] += (u - self.estimated_means[a]) / self.number_of_pulls[a]
    self.draws_from_each_arm[a] = np.append(self.draws_from_each_arm[a], u)
    std = np.std(self.draws_from_each_arm[a])
    self.estimated_vars[a] = std ** 2
    self.standard_errors[a] = std / np.sqrt(self.number_of_pulls[a])
    return u

  @abstractmethod
  def reward_dbn(self, a):
    pass


class NormalMAB(MAB):
  def __init__(self, list_of_reward_mus=[[1], [2]], list_of_reward_vars=[[1], [1]]):
    MAB.__init__(self, list_of_reward_mus)
    self.list_of_reward_vars = list_of_reward_vars
  
  def reward_dbn(self, a):
    # utility is distributed as Normal(mu, var)
    return np.random.normal(self.list_of_reward_mus[a], np.sqrt(self.list_of_reward_vars[a]))
    

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
    x = self.next_context()
    self.X = np.vstack((self.X, x))
    self.update_linear_model(a, x, u)
    return {'Utility': u, 'New context': x}

  @abstractmethod
  def draw_context(self):
    pass

  def initial_context(self):
    x0 = self.draw_context()
    self.curr_context = x0
    return x0

  def expected_reward(self, a, context):
    return np.dot(context, self.list_of_reward_betas[a])

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

  def draw_context(self):
    return np.random.multivariate_normal(self.context_mean, self.context_var)


class NormalUniformCB(LinearCB):
  def __init__(self, list_of_reward_betas=[[-0.1], [0.1]], context_mean=[0.5], list_of_reward_vars=[[2], [2]]):
    LinearCB.__init__(self, context_mean, list_of_reward_betas, list_of_reward_vars)
    self.context_dimension = 1

  def draw_context(self):
    return np.random.random()


