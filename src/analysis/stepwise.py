#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 25 12:21:41 2018

@author: lili
"""
import pdb
import os
import sys
this_dir = os.path.dirname(os.path.abspath(__file__))
project_dir = os.path.join(this_dir, '..', '..')
sys.path.append(project_dir)
import numpy as np
from src.environments.Bandit import NormalMAB, BernoulliMAB
import matplotlib.pyplot as plt

def rollout_epsilon_mab(env, mc_rep=50, T=100):
  best_score = -float('inf')
  for epsilon in np.linspace(0, 0.5, 25):
    mean_cum_reward = 0.0
    for rep in range(mc_rep):
      env.reset()
      for t in range(T):
        if np.random.rand() < epsilon:
          action = np.random.choice(2)
        else:
          action = np.argmax([env.estimated_means])
        env.step(action)
      cum_reward = sum(env.U)
      mean_cum_reward += (cum_reward - mean_cum_reward)/(rep+1)
    if mean_cum_reward > best_score:
      best_score = mean_cum_reward
      best_epsilon = epsilon
  return best_epsilon


def rollout_stepwise_linear_mab(zeta, env, J=10, mc_rep=10, T=100):
  mean_cum_reward = 0.0
  for rep in range(mc_rep):
    env.reset()
    for t in range(T):
      epsilon = stepwise_linear_epsilon(zeta, J, t, T=100)
      if np.random.rand() < epsilon:
        action = np.random.choice(2)
      else:
        action = np.argmax([env.estimated_means])
      env.step(action)
    cum_reward = sum(env.U)
    mean_cum_reward += (cum_reward - mean_cum_reward)/(rep+1)
  return mean_cum_reward
  

def stepwise_linear_epsilon(zeta, J, t, T=100):
  ## zeta: parameter vector with length J
  interval = int(T/J)
  if t == 0:
    j = J - 1
  else:
    j = int(np.floor((T-t)/interval))
  epsilon = sum(zeta[:j]) + ((T-t) - j*interval) * zeta[j] / interval
  return epsilon


def stochastic_approximation_step(zeta_k, lambda_k, env, J, mc_rep, T):
  z = np.random.normal(loc=0, size=J)
  v_of_zeta_k = rollout_stepwise_linear_mab(zeta_k, env, J=J, mc_rep=mc_rep, T=T)
  v_of_zeta_k_plus_z = rollout_stepwise_linear_mab(zeta_k + z, env, J=J, mc_rep=mc_rep, T=T)
  zeta_k_plus_one = zeta_k + lambda_k * z * (v_of_zeta_k_plus_z - v_of_zeta_k)
  return zeta_k_plus_one


def optimize_zeta(zeta_init, reward_mus, reward_vars, mc_rep=10, T=100):
  MAX_ITER = 100
  TOL = 1e-4
  it = 0
  diff = float('inf')
  J = zeta_init.size
  env = NormalMAB(list_of_reward_mus = reward_mus, list_of_reward_vars = reward_vars)
  zeta = zeta_init

  while it < MAX_ITER and diff > TOL:
    print(it)
    lambda_ = 0.01 / (it + 1)
    new_zeta = stochastic_approximation_step(zeta, lambda_, env, J, mc_rep, T)
    new_zeta = np.array([np.min((np.max((z, 0.0)), 0.10)) for z in new_zeta])
    diff = np.linalg.norm(new_zeta - zeta) / np.linalg.norm(zeta)
    zeta = new_zeta
    print(zeta)
    it += 1

  return zeta


if __name__ == "__main__":
  J = 10
  zeta_init = 0.1 * np.ones(J)
  replicates = 1
  mc_rep = 100
  method = "stochastic-gradient"
  
#  for mu in [1.1, 2, 5, 10]:
  for mu in [2]:
#    for var in [1]:
    for var in [1, 10, 100]:
      for rep in range(replicates):
        reward_mus = [[1],[mu]]
        reward_vars = [[1], [var]]
        if method == "stochastic-gradient":
          zeta_opt = optimize_zeta(zeta_init, reward_mus, reward_vars, 
                                   mc_rep=mc_rep)
        elif method == "random":
          zeta_opt = np.random.uniform(low=0.0, high=0.1, size=J)
        times = np.linspace(0, 100, 100)
        vals = [stepwise_linear_epsilon(zeta_opt, J, t) for t in times]  
        plt.plot(times, vals)
        plt.title("mus{}; vars {}".format(reward_mus, reward_vars))
        # plt.show()
        plt.savefig("method_{}_MC{}_samePara_mu{}_var{}.png".format(method, mc_rep, mu, var))
        
#        rollout_epsilon_mab()




