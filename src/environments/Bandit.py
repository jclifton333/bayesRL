#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 20 15:27:56 2018

@author: lili
"""

import numpy as np
from abc import ABCMeta, abstractmethod

ABC = ABCMeta('ABC', (object, ), {'__slots__': ()})

class Bandit(ABC):   
  def __init__ (self):
      pass
      
  def reset(self):
    self.U = []
    self.X = self.initial_context()
    self.A = []
    self.number_of_pulls = np.zeros(self.number_of_actions)
    self.estimated_means = np.zeros(self.number_of_actions)
    self.estimated_vars = np.zeros(self.number_of_actions)
    self.standard_errors = np.zeros(self.number_of_actions)
    self.draws_from_each_arm = [[]]*self.number_of_actions
    
  def step(self, a):
    ''' return untility given an action and current context'''
    u = self.reward_dbn(a)
    x = self.next_context()
    self.U = np.append(self.U, u)
    self.A = np.append(self.A, a)
    self.X = np.vstack((self.X, x))
    self.number_of_pulls[a] += 1
    self.estimated_means[a] += (u - self.estimated_means[a])/self.number_of_pulls[a]
    self.draws_from_each_arm[a] = np.append(self.draws_from_each_arm[a], u)
    std = np.std(self.draws_from_each_arm[a] )
    self.estimated_vars[a] = std**2
    self.standard_errors[a] = std/np.sqrt(self.number_of_pulls[a])    
    return {"Utility":u, "New context":x}

  @abstractmethod
  def reward_dbn(self, a):
    pass
  
  @abstractmethod
  def initial_context(self):
    pass
  
  @abstractmethod
  def next_context(self):
    pass


class MAB(Bandit):
  def __init__ (self):
    Bandit.__init__(self)
  
  def initial_context(self):
    return None
    
  def next_context(self):
    return None
  
  @abstractmethod
  def reward_dbn(self, a):
    pass  


class NormalMAB(MAB):
  def __init__ (self, list_of_reward_mus=[[1],[2]], list_of_reward_vars=[[1],[1]]):
    MAB.__init__(self)
    self.number_of_actions = len(list_of_reward_vars)
    self.list_of_reward_mus = list_of_reward_mus
    self.list_of_reward_vars = list_of_reward_vars
  
  def reward_dbn(self, a):
    # utility is distributed as Normal(mu, var)
    return np.random.normal(self.list_of_reward_mus[a], np.sqrt(self.list_of_reward_vars[a]))
    

class BernoulliMAB(MAB):
  def __init__ (self, list_of_probs=[0.3,0.7]):
    MAB.__init__(self)
    self.list_of_probs = list_of_probs
    self.number_of_actions = len(list_of_probs)
  
  def reward_dbn(self, a):
    # utility is distributed as Bernoulli(p)
    prob = self.list_of_probs[a]
    u = np.random.binomial(n=1, p=prob) 
    return u    
      

class LinearCB(Bandit):
  def __init__ (self, list_of_reward_betas=[[1,1],[2,-2]], list_of_reward_vars=[[1],[1]]):
    Bandit.__init__(self)
    ## list_of_reward_vars: a list of scalars
    ## context_mean: the mean vetor, same length as each vector in the list "list_of_reward_betas";
    ## context_var: the covariance matrix
    self.number_of_actions = len(list_of_reward_vars)
    self.context_dimension = None
    self.curr_context = None
    self.list_of_reward_betas = list_of_reward_betas
    self.list_of_reward_vars = list_of_reward_vars

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
    return np.random.normal(0.0, np.sqrt(var))

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
  def __init__ (self, list_of_reward_betas=[[1,1],[2,-2]], list_of_reward_vars=[[1],[1]],
                context_mean=[0,0], context_var=np.array([[1., 0.1],[0.1, 1.]]) ):
    LinearCB.__init__(self, list_of_reward_betas, list_of_reward_vars)
    self.context_dimension = len(context_mean)
    self.context_mean = context_mean
    self.context_var = context_var

  def draw_context(self):
    return np.random.multivariate_normal(self.context_mean, self.context_var)


class NormalUniformCB(LinearCB):
  def __init__ (self, list_of_reward_betas=[[-0.1],[0.1]], list_of_reward_vars=[[2],[2]]):
    LinearCB.__init__(self, list_of_reward_betas, list_of_reward_vars)
    self.context_dimension = 1

  def draw_context(self):
    return np.random.random()


