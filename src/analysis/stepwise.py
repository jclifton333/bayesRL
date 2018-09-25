#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 25 12:21:41 2018

@author: lili
"""




#best_zeta2 = None
#best_score = -float('inf')
#print('p0p1 {}'.format(p0p1))
#for zeta2 in np.linspace(0, 2, zeta_grid_dimension):
#  zeta = np.array([0.3, -5, zeta2])
def rollout_stepwise_linear_mab(zeta, env, J=10, mc_rep=500, T=100):
  mean_cum_reward = 0.0
  for rep in range(mc_rep):
    env.reset()
    for t in range(T):
      epsilon = tune_stepwise_linear_mab(zeta, J, t, T=100)
      if np.random.rand() < epsilon:
        action = np.random.choice(2)
      else:
        action = np.argmax([env.estimated_means])
      env.step(action)
    cum_reward = sum(env.U)
    mean_cum_reward += (cum_reward - mean_cum_reward)/(rep+1)
  return mean_cum_reward
  

def tune_stepwise_linear_mab(zeta, J, t, T=100):
  ## zeta: parameter vector with length J
  interval = int(T/J)
  j = int((T-t)/interval)
  epsilon = sum(zeta[:j]) + ((T-t) - j*interval) * zeta[j]
  return epsilon
  