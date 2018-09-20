#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 20 15:27:56 2018

@author: lili
"""
import numpy as np
from abc import ABC

class Bandit(ABC):   
  def __init__ (self):
      pass
      
  def reset(self):
    self.U = []
    self.X = []
    self.A = []
    
  def step(self, a):
      ''' return untility given an action and current context'''
      u = self.reward_dbn(a)
      x = self.next_context()
      self.U = np.append(self.U, u)
      self.A = np.append(self.A, a)
      if len(self.A) == 1:
        self.X = x
      else:
        self.X = np.vstack((self.X, x))
      self.X = np.vstack((self.X, x))
      return u, x
  
  @abstractmethod
  def reward_dbn(self, a):
    pass
  @abstractmethod
  def next_context():
    pass
  
           
class BernoulliMAB(Bandit):
  def __init__ (self, list_of_probs):
    self.list_of_probs = list_of_probs
  
  def reward_dbn(self, a):
    p = self.list_of_probs[a]
    u = np.random.binomial(n=1, p) # utility is distributed as Bernoulli(p)
    return u
  
  def next_context(self):
    pass
   
     
class NormalCB(Bandit):
  def __init__ (self, list_of_reward_betas, list_of_reward_vars, 
                context_mean, contex_var):
    self.list_of_reward_betas = list_of_reward_betas
    self.list_of_reward_vars = list_of_reward_vars
    self.context_mean = context_mean
    self.context_var = context_var
    self.curr_context = np.random.multivariate_normal(self.context_mean, self.context_cov)  
    
  def reward_dbn(self, a):     
    mean = np.dot(self.curr_context, self.list_of_reward_betas[a])
    var = self.list_of_reward_vars[a]
    u = np.random.normal(mean, var)
    return u
  
  def next_context(self):
    x = np.random.multivariate_normal(self.context_mean, self.context_cov)
    self.curr_context = x
    return x
        
        
        
        
        
        
        
