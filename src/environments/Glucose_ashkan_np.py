#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 17 13:25:01 2020
## A1c distribution is estimated nonparametrically
@author: lwu9
"""


import sys
import pdb
import os
# this_dir = os.path.dirname(os.path.abspath(__file__))
# project_dir = os.path.join(this_dir, '..', '..')
# sys.path.append(project_dir)

import copy
import numpy as np
import math


class Glucose_np(object):
  def __init__(self,nPatients, regression, sd, sigma_eps = 0.5, mu0 = 7.7, 
               sigma_B0_X0_Y0 = 1, mu_B0_X0 = 0, prob_L_given_trts = np.array([0.2, 0.2, 0.2, 0.35]), # metf, sulf, glit, insu
               death_prob_coef = np.array([-10, 0.08, 0.5]), s_initials=None):
    # Generative model parameters
    self.sigma_eps = sigma_eps
    self.mu0 = mu0
    self.sigma_B0_X0_Y0 = sigma_B0_X0_Y0
    self.mu_B0_X0 = mu_B0_X0
    self.prob_L_given_trts = prob_L_given_trts
    self.death_prob_coef = death_prob_coef 
    
    self.R = [[]] * nPatients  # List of rewards at each time step
    self.A = [[]] * nPatients  # List of actions at each time step
    self.S = [[]] * nPatients  # List of states: < NAT, Discontinue, A1c, BP, Weight, C_t > at each time step
    self.X = [[]] * nPatients
    #    self.horizon = horizon
    self.current_state = [None] * nPatients
    self.last_state = [None] * nPatients
    self.last_action = [None] * nPatients
    self.nPatients = nPatients

    self.s_initials = s_initials
    self.t = None
    
    self.regression = regression
    self.sd = sd
    
  @staticmethod
  def reward_function(s):
    """

    :param s: state vector at current time step
    :return:
    """
#    eps = np.random.normal(scale=0.1, size=1)
    eps=0
    if s[5]==1:
      return -20.0+eps
    elif s[2]<7:  
      return 1.0+eps
    elif s[2]>7 and s[1]==1:
      return -2.0+eps
    else:
      return 0.0+eps

  @staticmethod
  def reward_funciton_mHealth(s_news):
    """

    :param s_news: an 2-d array of new s values for all patients (dim: (time steps * nPatients) by 6)
    :return:
    """
    r = np.zeros(s_news.shape[0])
    ind = (s_news[:,5] == 1)
    r[ind] = -20.0
    ind1 = s_news[~ind, 2] < 7
    r[~ind][ind1] = 1.0
    ind2 = s_news[~ind, :][~ind1, 1] == 1
    r[~ind][~ind1][ind2] = -2.0
    return np.mean(r, axis=0)

  def get_next_state(self,prev_state,action, i):
    bp_t=self.get_bp_t(prev_state[3],np.random.normal(0,self.sigma_eps,1))[0]
    w_t=self.get_w_t(prev_state[4],np.random.normal(0,self.sigma_eps,1))[0]
    c_t=self.get_c_t(prev_state[2],prev_state[0])[0]
    #Initialize NAT
    NAT=0
    if prev_state[0]<4:
      if action==1:
        NAT=prev_state[0]+1
      else:
        NAT=prev_state[0]
      d_t=self.get_d_t(action,prev_state[0])[0]
      A1c_t=self.get_A1c(prev_state,action,self.sd)[0]
      return [NAT,d_t,A1c_t,bp_t,w_t,c_t]
  # If NAT>4, make algorithm indiffernt between action 1 and 0 to enable better state space exploration
    else:
      d_t=self.get_d_t(0,prev_state[0])[0]
      A1c_t=self.get_A1c(prev_state,0,self.sd)[0]
      return [4,d_t,A1c_t,bp_t,w_t,c_t]

  def get_bp_t(self,prev_bp,eps):
    bp_t=(prev_bp+eps)/math.sqrt(1+float(self.sigma_eps)**2)
    return bp_t

  def get_w_t(self,prev_w,eps):
    w_t=(prev_w+eps)/math.sqrt(1+float(self.sigma_eps)**2)
    return w_t
    
  def get_c_t(self,prev_A1c,prev_NAT):
    ## death indicator
    A1c_indicator=int(prev_A1c>7)
    x=self.death_prob_coef[0]+self.death_prob_coef[1]*A1c_indicator*(prev_A1c**2)+self.death_prob_coef[2]*prev_NAT
    c_t=np.random.binomial(1,self.exp_helper(x),1)
    return c_t

  def get_d_t(self,action,prev_NAT):
    ## treatment discontinuation indicator
    p=0
    if action==0:
      return [0]
    else:
      p=self.prob_L_given_trts[int(prev_NAT)]
#       if prev_NAT==3:
#         p=Glucose.prob_L_given_trts[3]
#       else:
#         p=Glucose.prob_L_given_trts[0]
      d_t=np.random.binomial(1,p,1)
      return d_t

  def get_A1c(self,prev_state,action,sd):
      ## i: the ith patient
      s = np.append(prev_state, action)
      mean = self.regression.predict(s.reshape(1, len(s)))
      A1c_t = np.random.normal(mean, sd)
      return A1c_t


  def reset(self):
    """

    :return:
    """
    # Reset obs history
    self.t = -1
    self.R = [[]] * self.nPatients  # List of rewards at each time step
    self.A = [[]] * self.nPatients  # List of actions at each time step
    self.S = [[]] * self.nPatients  # state is 6 dim
    self.X = [[]] * self.nPatients  # state is 7 dim, state plus action
    self.current_s = [[]] * self.nPatients
    
    # Generate first states for nPatients
    if self.s_initials is None:
      for i in range(self.nPatients):
        bp_0=np.random.normal(self.mu_B0_X0, self.sigma_B0_X0_Y0, 1)[0]
        w_0=np.random.normal(self.mu_B0_X0, self.sigma_B0_X0_Y0, 1)[0]
        A1c_0=np.random.normal(self.mu0, self.sigma_B0_X0_Y0, 1)[0]
        s_0=[0,0,A1c_0,bp_0,w_0, 0]
        self.S[i] = np.append(self.S[i], [s_0])
        self.current_s[i] = s_0
    else:
        ## need to check this part, not correct
      self.S = [self.s_initials[i,] for i in range(self.nPatients)]
      for i in range(self.nPatients):
        self.S[i] = np.vstack((self.S[i], self.S[i]))

     # self.last_state = [self.sx_initials[i, 4:7] for i in range(self.nPatients)]
     # self.current_state = [self.sx_initials[i, 1:4] for i in range(self.nPatients)]
      self.last_action = [self.x_initials[i, 7] for i in range(self.nPatients)]
      self.A = [np.array(self.x_initials[i, 7]) for i in range(self.nPatients)]
      self.R = [np.array(self.reward_function(self.s_initials[i,:])) \
                for i in range(self.nPatients)]
    return

  def step(self, actions):
    '''
    actions: an array of actions for each patient
    '''
    self.t += 1
    #    done = self.t == self.horizon
    s_list = []
    mean_rewards_nPatients = []
    done = False
    for i in range(self.nPatients):
      #      pdb.set_trace()
      if self.current_s[i][-1] == 0: ## the ith patient is not dead
        if self.t > 0: 
          if self.S[i][-1,0] == 4:
            actions[i] = 0
          s = self.get_next_state(self.S[i][-1,:], actions[i], i)
          self.current_s[i] = s
          reward = self.reward_function(s)
          self.X[i][-1,-1] = actions[i]
          self.X[i] = np.vstack((self.X[i], np.concatenate((s, [-1]))))
        else:
          s = self.get_next_state(self.S[i], actions[i], i)
          self.current_s[i] = s
          reward = self.reward_function(s) 
          self.X[i] = np.append(self.X[i], np.concatenate((self.S[i], [actions[i]])))
          self.X[i] = np.vstack((self.X[i], np.concatenate((s, [-1]))))
        s_list.append(s)
          #      print(i, reward)
        self.R[i] = np.append(self.R[i], reward)
        self.A[i] = np.append(self.A[i], actions[i])
        self.S[i] = np.vstack((self.S[i], s))
        mean_rewards_nPatients = np.append(mean_rewards_nPatients, reward)
    if s_list==[]: ## this means all patients are dead, then stop running the policy
       s_list=[1] 
       mean_rewards_nPatients=[0]
       done = True
    return done, np.vstack(s_list), np.mean(mean_rewards_nPatients)  # , done

  def get_state_transitions_as_x_y_pair(self, new_state_only=True):
    """
    For estimating transition density.
    :return:
    """
    #    X = np.vstack(self.X[1:])
    #    Sp1 = np.vstack(self.S[1:])
    if new_state_only:
      X = np.vstack([self.X[j][1:] for j in range(self.nPatients)])
      y = np.vstack([self.S[j][2] for j in range(self.nPatients)])
    else:
      X = np.vstack([self.X[j] for j in range(self.nPatients)])
      y = np.vstack([self.S[j][:, 2] for j in range(self.nPatients)])
    return X, y
    
  def get_S_A_R_Sprime(self):
    """
    For estimating one step Q by doing regression R on S and A.
    :return:
    """  
    S = np.vstack([self.S[j][:-1][:,:-1] for j in range(self.nPatients)])
    R = np.hstack((self.R))
    A = np.hstack((self.A))
    S_prime = np.vstack([self.S[j][1:] for j in range(self.nPatients)])
    terminate_s = np.where(S_prime[:,5]==1)
    S_prime = S_prime[:,:-1] ## delete the last colunm: death indicators
    return S, A, R, S_prime, terminate_s

  def get_current_S(self):
      current_s = np.vstack([self.S[j][-1][:-1] for j in range(self.nPatients)])
      return current_s

  def exp_helper(self, x):
    return float(math.exp(x))/float(1+math.exp(x))

#   def next_state_and_reward(self, action, i):
#     """

#     :param action:
#     :return:
#     """

#     # Transition to next state
#     #    print(self.current_state[i], self.last_action[i])
#     get_next_state(self,prev_state,action)
#     sx = np.concatenate(([1], self.current_state[i], self.last_state[i],
#                          [self.last_action[i]]))
#     x = np.concatenate((sx, [action]))
#     glucose = np.random.normal(np.dot(x, self.COEF), self.SIGMA_NOISE)
#     food, activity = self.generate_food_and_activity()

#     # Update current and last state and action info
#     self.last_state[i] = copy.copy(self.current_state[i])
#     self.current_state[i] = np.array([glucose, food, activity]).reshape(1, 3)[0]
#     self.last_action[i] = action
#     reward = self.reward_function(self.last_state[i], self.current_state[i])
#     # current_x = np.concatenate()
#     return x, reward

  @staticmethod
  def get_state_at_action(action, x):
    """
    Replace current action entry in x with action.
    :param action:
    :param x:
    :return:
    """
    new_x = copy.copy(x)
    new_x[-1] = action
    return new_x

  def get_state_history_as_array(self):
    """
    :return:
    """
    X_as_array = np.vstack(self.X)
    S_as_array = np.vstack(self.S)
    return X_as_array, S_as_array

 