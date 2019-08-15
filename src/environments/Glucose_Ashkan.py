## import sys
import pdb
import os
# this_dir = os.path.dirname(os.path.abspath(__file__))
# project_dir = os.path.join(this_dir, '..', '..')
# sys.path.append(project_dir)

import copy
import numpy as np
from statistics import mean
import math


class Glucose(object):
  def __init__(self, nPatients=1, s_initials=None):
    # Generative model parameters
    self.metrformin_te=0.14
    self.solfonylurea_te=0.2
    self.glitazone_te=0.02
    self.insulin_te=0.14
    self.sd_eps=0.5
    self.prev_u_t=9
    self.d_t_prob = np.array([0.2, 0.2, 0.2, 0.35])

    self.R = [[]] * nPatients  # List of rewards at each time step
    self.A = [[]] * nPatients  # List of actions at each time step
#    self.X = [[]] * nPatients  # List of features (previous and current states) at each time step
    self.S = [[]] * nPatients
#    self.Xprime_X_inv = None
    self.t = -1
    #    self.horizon = horizon
    self.current_state = [None] * nPatients
#    self.last_state = [None] * nPatients
#    self.last_action = [None] * nPatients
    self.nPatients = nPatients

    self.s_initials = s_initials
#    self.sx_initials = sx_initials

  @staticmethod
  def reward_function(s_prev, s):
    """

    :param s_prev: state vector at previous time step
    :param s: state vector
    :return:
    """
    prev_A1c = s_prev[2]
    prev_NAT = s_prev[0]
    A1c_indicator = int(prev_A1c>7)
    x = -10+0.08*A1c_indicator*(prev_A1c**2)+0.5*prev_NAT
    C_t = np.random.binomial(1,0.5,1)
    if C_t==1:
      r = -10.0
    elif s[2]<7:
      r = 1.0
    elif s[2]>7 and s[1]==1:
      r = -2.0
    else:
      r = 0.0
    return r

  def reward_funciton_mHealth(self, s_prev_nPatients, s_nPatients):
    """

    :param s_prev_nPatients: an list of s_prev for all patients (dim: nPatients by 5, length: nPatients)
    :param s_nPatients: an array of s for all patients (dim: nPatients by 5, length: nPatients)
    :return:
    """
    rewards_list = list(map(self.reward_function, s_prev_nPatients, s_nPatients))
    return mean(rewards_list)

  def get_bp_t(self, prev_bp, eps):
    bp_t=(prev_bp+eps)/math.sqrt(1+float(self.sd_eps)**2)
    return bp_t

  def get_w_t(self, prev_w, eps):
    w_t=(prev_w+eps)/math.sqrt(1+float(self.sd_eps)**2)
    return w_t

  def get_d_t(self, action, prev_NAT):
    p=0
    if action==0:
    return np.array([0])
    else:
      p = self.d_t_prob[prev_NAT]
    d_t=np.random.binomial(1,p,1)
    return d_t

  def get_A1c(self, prev_state, action, d_t, eps):
    ## need to check, what if actions are inconsistent with d_t?
    if prev_state[2]>7 and prev_state[0]<4 and action!=0 and d_t!=1:
      new_u_t=0
      if prev_state[0]==0:
        new_u_t=self.prev_u_t*(1-self.metrformin_te)
      elif prev_state[0]==1:
        new_u_t=self.prev_u_t*(1-self.solfonylurea_te)
      elif prev_state[0]==2:
        new_u_t=self.prev_u_t*(1-self.glitazone_te)
      elif prev_state[0]==3:
        new_u_t=self.prev_u_t*(1-self.insulin_te)
      A1c_t=(prev_state[2]-self.prev_u_t+eps)/math.sqrt(1+self.sd_eps**2)+new_u_t
      self.prev_u_t=new_u_t
      return A1c_t
    else:
      A1c_t=(prev_state[2]-self.prev_u_t+eps)/math.sqrt(1+self.sd_eps**2)+self.prev_u_t
      return A1c_t

  # Return tuple of < NAT, D, A1c, BP, Weight>,C_t at beginning of trajectory
  def reset(self):
    # Reset obs history
    self.R = [[]] * self.nPatients
    self.A = [[]] * self.nPatients
    self.S = [[]] * self.nPatients
    # Generate first states for nPatients
    if self.s_initials is None:
      for i in range(self.nPatients):
        bp_0=np.random.normal(0,1,1)[0]
        w_0=np.random.normal(0,1,1)[0]
        A1c_0=np.random.normal(7.1,1,1)[0]
        s_0=[0,0,A1c_0,bp_0,w_0]
        action = np.random.choice(2)
        self.current_state[i] = np.array(s_0)
        self.S[i] = s_0
    else:
      self.S = [self.s_initials[i,] for i in range(self.nPatients)]
    return

  def get_next_state(self, prev_state, action):
    eps=np.random.normal(0, self.sd_eps, 1)
    bp_t=self.get_bp_t(prev_state[3],eps)[0]
    w_t=self.get_w_t(prev_state[4],eps)[0]
    #Initialize NAT
    NAT=0
    if prev_state[0]<4:
      if action==1:
        NAT=prev_state[0]+1
      else:
        NAT=prev_state[0]
      d_t=self.get_d_t(action,prev_state[0])[0]
      A1c_t=self.get_A1c(prev_state,action,d_t,eps)[0]
      s = [NAT,d_t,A1c_t,bp_t,w_t]
    else:
      d_t=self.get_d_t(0,prev_state[0])[0]
      A1c_t=self.get_A1c(prev_state,0,d_t,eps)[0]
      s = [4,d_t,A1c_t,bp_t,w_t]
    return s

  def step(self, actions):
    '''
    actions: an array of actions for each patient
    '''
    self.t += 1
    #    done = self.t == self.horizon
    s_list = []
    mean_rewards_nPatients = 0
    for i in range(self.nPatients):
      s = self.get_next_state(self.current_state[i], action[i])
      reward = reward_function(self.current_state[i], s)
      self.S[i] = np.vstack((self.S[i], s)) # need to check again
      self.R[i] = np.append(self.R[i], reward)
      self.A[i] = np.append(self.A[i], actions[i])
      s_list.append(s)

      #      pdb.set_trace()
      mean_rewards_nPatients += (reward - mean_rewards_nPatients) / (i + 1)
    return np.vstack(s_list), mean_rewards_nPatients  # , done

  #Helper function to calculate value of e^x/(1+e^x)
  def exp_helper(self,x):
    return float(math.exp(x))/float(1+math.exp(x))
