#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 25 12:21:41 2018

@author: lili

See https://www.ncbi.nlm.nih.gov/pmc/articles/PMC4769712/.
Functions called 'spall' are recommendations of Spall (1998a as cited in above).
Also trying version on pdf pg 21 of https://arxiv.org/pdf/1804.05589.pdf.
"""
import pdb
import os
import copy
import sys
this_dir = os.path.dirname(os.path.abspath(__file__))
project_dir = os.path.join(this_dir, '..', '..')
sys.path.append(project_dir)
import numpy as np
from src.environments.Bandit import NormalMAB, BernoulliMAB
import matplotlib.pyplot as plt


def rademacher(n):
  """
  For simultaenous perturbation.
  :param n:
  :return:
  """
  return np.random.choice([-1, 1], size=n, replace=True)


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


def adaptive_initial_step(v_init, v_of_zeta_k_plus_z, v_of_zeta_k_minus_k, best_zeta, zeta_k, a):
  """
  From paper linked at top of script.

  :param v_init:
  :param v_of_zeta_k_plus_z:
  :param v_of_zeta_k_minus_k:
  :param best_zeta:
  :param zeta_k:
  :param a:
  :return:
  """
  if min((v_of_zeta_k_plus_z, v_of_zeta_k_minus_k)) - v_init >= 0:
    zeta_kp1 = best_zeta
    a = 0.5 * a
  else:
    zeta_kp1 = zeta_k
  return zeta_kp1, a


def spall_a_k(k, initial_diff, a=None):
  """

  :param k:
  :param initial_diff: one element of finite difference evaluated at initial parameter iterate.
  :return:
  """
  A = 0.1 * 100  # less than or equal to max number iterations
  alpha = 0.602
  if a is None:
    a = 0.01 * np.power(A + 1, alpha) / np.abs(initial_diff)
  return a / np.power(A + k + 1, alpha)


def spall_c_k(k):
  c = 1.0  # Standard deviation of measurement noise
  gamma = 0.101
  return c / np.power(k + 1, gamma)


def bb_step_size_proposal(zeta_kp1, zeta_k, diff_kp1, diff_k):
  delta_zeta = zeta_kp1 - zeta_k
  delta_diff = diff_kp1 - diff_k
  num_ = np.dot(delta_zeta, delta_diff)
  denom_ = np.dot(delta_diff, delta_diff)
  return num_ / denom_


def bb_step_size(zeta_kp1, zeta_k, diff_kp1, diff_k, list_of_previous_step_sizes):
  a_k = bb_step_size_proposal(zeta_kp1, zeta_k, diff_kp1, diff_k)
  if a_k < 0:
    max_a = np.max(list_of_previous_step_sizes)
    min_a = np.min(list_of_previous_step_sizes)
    a_k = np.max((min_a, np.min((a_k, max_a))))
  k = len(list_of_previous_step_sizes)
  t = min((2, k))
  list_of_previous_step_sizes.append(a_k)
  final_a_k = (1 / (t + 1)) * np.sum(list_of_previous_step_sizes[-t:])
  return final_a_k, list_of_previous_step_sizes


def stochastic_approximation_step(zeta_k, zeta_km1, c_k, a_k, env, J, mc_rep, T, last_few_gradients,
                                  previous_gradient, list_of_previous_step_sizes, gradients_to_keep=3):
  z = rademacher(J)
  perturbation = c_k * z
  v_of_zeta_k_minus_z = rollout_stepwise_linear_mab(zeta_k - perturbation, env, J=J, mc_rep=mc_rep, T=T)
  v_of_zeta_k_plus_z = rollout_stepwise_linear_mab(zeta_k + perturbation, env, J=J, mc_rep=mc_rep, T=T)
  diff = (v_of_zeta_k_plus_z - v_of_zeta_k_minus_z) / (2 * perturbation)

  # Gradient smoothing
  if len(last_few_gradients) >= gradients_to_keep:
    last_few_gradients.pop(0)
  last_few_gradients.append(diff)
  smoothed_diff = np.mean(last_few_gradients, axis=0)
  a_k, list_of_previous_step_sizes = \
    bb_step_size(zeta_k, zeta_km1, diff, previous_gradient, list_of_previous_step_sizes)

  # Take step and evaluate
  zeta_kp1 = zeta_k + a_k * smoothed_diff
  v_of_zeta_kp1 = rollout_stepwise_linear_mab(zeta_kp1, env, J=J, mc_rep=mc_rep, T=T)

  return zeta_kp1, v_of_zeta_kp1, v_of_zeta_k_plus_z, v_of_zeta_k_minus_z, diff, last_few_gradients, \
         list_of_previous_step_sizes


def optimize_zeta(zeta_init, reward_mus, reward_vars, mc_rep=10, T=100):
  MAX_ITER = 100
  TOL = 1e-4
  it = 0
  diff = float('inf')
  J = zeta_init.size
  env = NormalMAB(list_of_reward_mus = reward_mus, list_of_reward_vars = reward_vars)
  zeta = zeta_init
  previous_zeta = np.zeros(zeta.size)
  last_few_gradients = []

  # Needed for setting hyperparams
  # Initial diff
  c_0 = spall_c_k(0)
  z = rademacher(J)
  perturbation = c_0 * z
  v_of_zeta_k_minus_z = rollout_stepwise_linear_mab(zeta - perturbation, env, J=J, mc_rep=mc_rep, T=T)
  v_of_zeta_k_plus_z = rollout_stepwise_linear_mab(zeta + perturbation, env, J=J, mc_rep=mc_rep, T=T)
  initial_diff = (v_of_zeta_k_plus_z - v_of_zeta_k_minus_z) / (2 * perturbation)
  last_few_gradients.append(initial_diff)  # For gradient smoothing
  previous_gradient = initial_diff

  # Get v_init
  v_init = rollout_stepwise_linear_mab(zeta, env, J=J, mc_rep=mc_rep, T=T)
  best_zeta = zeta
  best_v = v_init

  # a_k hyperparameter (from Spall)
  a = 0.01 * np.power(0.1 * 100 + 1, 0.602) / np.abs(initial_diff[0])
  list_of_previous_step_sizes = [a]

  while it < MAX_ITER and diff > TOL:
    print(it)

    # Get tuning parameters
    c_k = spall_c_k(it)
    # a_k = spall_a_k(it, initial_diff[0], a)

    # Do stochastic perturbation
    new_zeta, new_v_zeta, v_zeta_plus_z, v_zeta_minus_z, previous_gradient, last_few_gradients, \
      list_of_previous_step_sizes = stochastic_approximation_step(zeta, previous_zeta, c_k, None, env, J, mc_rep, T,
                                                                  last_few_gradients, previous_gradient,
                                                                  list_of_previous_step_sizes)

    # if new_v_zeta > best_v:
    #   best_zeta = new_zeta
    #   best_v = new_v_zeta

    # Adaptive
    # new_zeta, a = adaptive_initial_step(v_init, v_zeta_plus_z, v_zeta_minus_z, best_zeta, new_zeta, a)

    # diff = np.linalg.norm(new_zeta - zeta) / np.linalg.norm(zeta)
    previous_zeta = copy.copy(zeta)
    zeta = new_zeta
    print(zeta)
    print(new_v_zeta)
    it += 1

  return zeta


if __name__ == "__main__":
  J = 10
  zeta_init = 0.1 * np.ones(J)
  replicates = 10
  mc_rep = 100
  method = "stochastic-gradient"
  
  
#  for mu in [1.1, 2, 5, 10]:
  for mu in [2]:
    for var in [1]:
#    for var in [1, 10, 100]:
      for rep in range(replicates):
        reward_mus = [[1],[mu]]
        reward_vars = [[1], [var]]
#        env = NormalMAB(list_of_reward_mus = reward_mus, list_of_reward_vars = reward_vars)
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
#        plt.savefig("method_{}_MC{}_samePara_mu{}_var{}_rep{}_lambda001.png".format(method, mc_rep, mu, var, replicates))
        
#        rollout_epsilon_mab()


