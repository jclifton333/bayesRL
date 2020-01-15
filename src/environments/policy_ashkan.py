#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 11 11:58:05 2019

@author: lwu9

Policy for Glucose-Ashkan model
"""

from Glucose_ashkan import *
from ashkan_mle import *

import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
#import matplotlib.pyplot as plt
from functools import partial
from multiprocessing import Pool
import random 
import pdb


def fitted_q_iteration_mHealth(env, gamma, K=1, regressor=RandomForestRegressor):
  S, A, target, S_prime, terminate_s = env.get_S_A_R_Sprime()
  current_s = env.get_current_S()
  l = len(A)
  n,_ = current_s.shape
  # Fit one-step q fn
  X = np.hstack((S, A.reshape(l,1)))
  A0 = np.zeros(l)
  A1 = np.ones(l)
  regr = regressor(n_estimators=50, min_samples_leaf=1)
  regr.fit(X, target)
  # Fitted Q iteration with K rollouts
  for k in range(K):
      Q0 = regr.predict(np.hstack((S_prime, A0.reshape(l,1))))
      Q1 = regr.predict(np.hstack((S_prime, A1.reshape(l,1))))
      ## check again, do we have terminate state?
      if len(terminate_s) > 0:
          Q0[terminate_s] = -20 
          Q1[terminate_s] = -20 
      y = target + gamma*np.maximum(Q0, Q1)
      regr.fit(X, y)
  
  q0 = regr.predict(np.hstack((current_s, np.zeros(shape=(n,1)))))
  q1 = regr.predict(np.hstack((current_s, np.ones(shape=(n,1)))))
  # Last entry of list gives optimal action at current state
  optimal_actions = np.argmax(np.vstack((q0, q1)), axis=0)
  return optimal_actions


def linear_rule(env, gamma, K=1):
    S, A, target, S_prime, terminate_s = env.get_S_A_R_Sprime()
    l, p = S.shape
    current_s = env.get_current_S()
    A0 = np.zeros(l)
    A1 = np.ones(l)
    # Fit one-step q fn with linear regression
    X = np.hstack((np.hstack((S,A.reshape(l,1))), np.transpose(np.multiply(np.transpose(S),A))))
    regr = LinearRegression().fit(X, target)
    for k in range(K):
        Q0 = regr.predict(np.hstack((np.hstack((S_prime,A0.reshape(l,1))), np.transpose(np.multiply(np.transpose(S_prime),A0)))))
        Q1 = regr.predict(np.hstack((np.hstack((S_prime,A1.reshape(l,1))), np.transpose(np.multiply(np.transpose(S_prime),A1)))))
        ## check again, do we have terminate state?
        if len(terminate_s) > 0:
            Q0[terminate_s] = -20 
            Q1[terminate_s] = -20 
#      pdb.set_trace()
        y = target + gamma*np.maximum(Q0, Q1)
        regr.fit(X, y)
      
    coef = regr.coef_[-(p+1):]
    new_X = np.hstack((np.ones(shape=(current_s.shape[0],1)), current_s))
    optimal_actions = np.sum(new_X*coef, axis=1)>0
    return optimal_actions
  

def rewards_policy(rep, gamma, gamma_type, K, env, T, policy="eps-greedy", epsilon=0.05, fixed_decay=0, regr="RF",
            rollouts=100, gammas=np.arange(0,1,0.1), is_rollout=False):
    np.random.seed(rep)
  ## need to check this again
#  if gamma_type=="not_tune":
#    env = Glucose(nPatients=env.nPatients)
    env.reset()
    mean_rewards = 0
    actions = np.random.binomial(1,0.5,env.nPatients)
    done, _, rewards = env.step(actions)
    gammas_used = []; rewards_gammas_means_each_t = []
    for t in range(T):
        if done:
            break
        else:
            if policy=="eps-greedy":
                if fixed_decay == 1:
                    epsilon_t = 1/(t+1)
                elif fixed_decay == 2:
                    epsilon_t = 0.5/(t+1)
                elif fixed_decay == 3:
                    epsilon_t = 0.8**t
                else:
                    ## fixed epsilon
                    epsilon_t = epsilon
                if gamma_type=="increase":
            #        gamma = np.exp((t+1-T)/10)
                    gamma = np.exp((t+1-T)/5)
                elif gamma_type=="prespecify":
                    gamma = gammas[t] ## gammas: an array of pre-specified gammas at each time step
#                    print(t, rewards, gamma)
                elif gamma_type=="tune":
                    tune_gamma = policy_tune_gamma(env, gammas, K, T, rollouts=rollouts, regr=regr)
                    gamma = tune_gamma["gamma"]
                    rewards_gammas_means = tune_gamma["rewards_gammas_means"]
                    rewards_gammas_means_each_t = np.append(rewards_gammas_means_each_t,rewards_gammas_means)
                    print(t, rewards, gamma)
                elif gamma_type=="not_tune":
                    gamma = gamma
                elif gamma_type=="random":
                    gamma = np.random.uniform()
                else:
                    print("Wrong gamma_type")
                gammas_used = np.append(gammas_used, gamma)
                if regr=="RF":
                    est_opt_actions = fitted_q_iteration_mHealth(env, gamma, K)
                elif regr=="linear":
                    est_opt_actions = linear_rule(env, gamma, K)
                actions_random = np.random.binomial(1,0.5,env.nPatients)
                random_prob = np.random.rand(env.nPatients) 
                est_opt_actions[random_prob < epsilon_t] = actions_random[random_prob < epsilon_t]
#                for i in range(env.nPatients):
#                    if np.random.rand() >= epsilon_t:
#                        actions[i] = est_opt_actions[i]
#                    else:
#                        actions[i] = np.random.binomial(1,0.5,1)[0]
                done,_,rewards = env.step(est_opt_actions)
                mean_rewards += (rewards - mean_rewards) / (t+1)
#                if not is_rollout:
#                    pdb.set_trace()
    if gamma_type == "tune":
        return({'mean_rewards':mean_rewards, 'gammas_used':gammas_used, 
                'rewards_gammas_means_each_t':rewards_gammas_means_each_t})
    else:
        return({'mean_rewards':mean_rewards, 'gammas_used':gammas_used})
            

def policy_tune_gamma(env, gammas, K, T, rollouts, regr):
  rollout_gamma_type = "not_tune"
  rewards_gammas = np.zeros(shape=(rollouts, len(gammas)))
  for i in range(rollouts):
    print('rollout {}'.format(i))
    rep = i
    model_boot = GlucoseTransitionModel()
    model_boot.bootstrap_and_fit_conditional_densities(env.X)
    for j in range(len(gammas)):
      env_rollout = Glucose(nPatients=env.nPatients, sigma_eps = model_boot.sigma_eps, 
                            mu0 = model_boot.mu0, sigma_B0_X0_Y0 = model_boot.sigma_B0_X0_Y0, 
                            mu_B0_X0 = model_boot.mu_B0_X0, prob_L_given_trts = model_boot.prob_L_given_trts,
                            tau_trts=model_boot.tau_trts, death_prob_coef=model_boot.death_prob_coef)
      gamma = gammas[j]
      rewards_gammas[i,j] = rewards_policy(rep, gamma, rollout_gamma_type, K, env_rollout, 
                    T, regr=regr, is_rollout=True)["mean_rewards"] 
#  pdb.set_trace()
  rewards_gammas_means = np.mean(rewards_gammas, axis=0)
#  print(rewards_gammas_means)
  return({"gamma":gammas[np.argmax(rewards_gammas_means)], "rewards_gammas_means":rewards_gammas_means})

    
#
  
#reg = regressor(n_estimators=30, n_jobs=7, min_samples_leaf=1)
#reg.fit(np.hstack((S,A.reshape(len(A), 1))), target)
#m,_ = current_s.shape
#q1 = reg.predict(np.hstack((current_s, np.ones(shape=(m,1)))))
#q0 = reg.predict(np.hstack((current_s, np.zeros(shape=(m,1)))))
#len(np.where(q0-q1==0)[0])
#reg.feature_importances_
#
#ind = np.where(q0-q1==0)[0]
#reg.decision_path(np.hstack((current_s[ind,:], np.ones(shape=(len(ind),1)) )) )
#reg.decision_path(np.hstack((current_s[ind,:], np.zeros(shape=(len(ind),1)) )) )
#reg.apply(np.hstack((current_s[ind,:], np.ones(shape=(len(ind),1)) )) )
#reg.apply(np.hstack((current_s[ind,:], np.zeros(shape=(len(ind),1)) )) )
#
#reg0.apply(current_s[ind[43]].reshape(1,-1))
#reg0.predict(current_s[ind[43]].reshape(1,-1))
#reg1.apply(current_s[ind[43]].reshape(1,-1))
#reg1.predict(current_s[ind[43]].reshape(1,-1))
