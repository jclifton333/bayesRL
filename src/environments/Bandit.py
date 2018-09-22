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
    
  def step(self, a):
    ''' return untility given an action and current context'''
    u = self.reward_dbn(a)
    x = self.next_context()
    self.U = np.append(self.U, u)
    self.A = np.append(self.A, a)
    self.X = np.vstack((self.X, x))
    return {"Utility":u, "New context":x}

  @abstractmethod
  def reward_dbn(self, a):
    pass
  @abstractmethod
  def next_context():
    pass
  @abstractmethod
  def initial_context():
    pass
  
           
class BernoulliMAB(Bandit):
  def __init__ (self, list_of_probs=[0.3,0.7]):
    self.list_of_probs = list_of_probs
    self.number_of_actions = len(list_of_probs)
  
  def reward_dbn(self, a):
    prob = self.list_of_probs[a]
    u = np.random.binomial(n=1, p=prob) # utility is distributed as Bernoulli(p)
    return u
  
  def next_context(self):
    pass
   
     
class NormalCB(Bandit):
  def __init__ (self, list_of_reward_betas=[[1,1],[2,-2]], list_of_reward_vars=[[1],[1]],
                context_mean=[0,0], context_var=np.array([[1., 0.1],[0.1, 1.]]) ):
    ## list_of_reward_vars: a list of scalars
    ## context_mean: the mean vetor, same length as each vector in the list "list_of_reward_betas";
    ## context_var: the covariance matrix
    self.number_of_actions = len(list_of_reward_vars)
    self.context_dimension = len(context_mean)
    self.list_of_reward_betas = list_of_reward_betas
    self.list_of_reward_vars = list_of_reward_vars
    self.context_mean = context_mean
    self.context_var = context_var
  
  def initial_context(self):
    x0 = np.random.multivariate_normal(self.context_mean, self.context_var)  
    self.curr_context = x0
    return x0
    
  def reward_dbn(self, a):     
    mean = np.dot(self.curr_context, self.list_of_reward_betas[a])
    var = self.list_of_reward_vars[a]
    u = np.random.normal(mean, var)
    return u
  
  def next_context(self):
    x = np.random.multivariate_normal(self.context_mean, self.context_var)
    self.curr_context = x
    return x
        
class NormalUniformCB(Bandit):
  def __init__ (self, list_of_reward_betas=[[-0.1],[0.1]], list_of_reward_vars=[[2],[2]],
                context_mean=[0,0], context_var=np.array([[1., 0.1],[0.1, 1.]]) ):
    ## list_of_reward_vars: a list of scalars
    ## context_mean: the mean vetor, same length as each vector in the list "list_of_reward_betas";
    ## context_var: the covariance matrix
    self.number_of_actions = len(list_of_reward_vars)
    self.context_dimension = len(context_mean)
    self.list_of_reward_betas = list_of_reward_betas
    self.list_of_reward_vars = list_of_reward_vars
    self.context_mean = context_mean
    self.context_var = context_var
  
  def initial_context(self):
    x0 = np.random.rand()  
    self.curr_context = x0
    return x0
    
  def reward_dbn(self, a): 
    intercept = [0.1, 0.2]    
    mean = intercept[a] + np.dot(self.curr_context, self.list_of_reward_betas[a])
    var = self.list_of_reward_vars[a]
    u = np.random.normal(mean, var)
    return u
  
  def next_context(self):
    x = np.random.rand()  
    self.curr_context = x
    return x
        
        
        
        
        
        
