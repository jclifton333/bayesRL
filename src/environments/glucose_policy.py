import sys
import pdb
import os
this_dir = os.path.dirname(os.path.abspath(__file__))
project_dir = os.path.join(this_dir, '..', '..')
sys.path.append(project_dir)


import matplotlib.pyplot as plt
from src.environments.Bandit import NormalCB, NormalUniformCB
from src.policies import tuned_bandit_policies as tuned_bandit
from src.policies import rollout
from src.environments.Glucose import Glucose
from src.environments.Glucose_example import Glucose_Approx
  
import copy
import numpy as np
import src.policies.linear_algebra as la
from scipy.linalg import block_diag
from functools import partial
import datetime
import yaml
import multiprocessing as mp

from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from bayes_opt import BayesianOptimization
from scipy.stats import wishart
from scipy.stats import chi2
import time


def maximize_q_function_at_x(q_fn, x, env):
  """

  :param q_fn: Function from feature vectors x to q-values.
  :param x: Feature vector from Glucose environment at which to maximize Q.
  :param env: Glucose environment
  :return:
  """
  x0, x1 = env.get_state_at_action(0, x), env.get_state_at_action(1, x)
  q0, q1 = q_fn(x0.reshape(1, -1)), q_fn(x1.reshape(1, -1))
  return np.max([q0, q1]), np.argmax([q0, q1])

 
def maximize_q_function_at_block(q_fn, X, env):
  """

  :param q_fn:
  :param X: Nx9 array of Glucose features.
  :param env:
  :return:
  """
  array_of_q_maxes = []
  list_of_optimal_actions = []
  for x in X:
    q_max, optimal_action = maximize_q_function_at_x(q_fn, x, env)
    array_of_q_maxes.append(q_max)
    list_of_optimal_actions.append(optimal_action)
  return np.array(array_of_q_maxes), list_of_optimal_actions


def fitted_q(env, gamma, regressor, number_of_value_iterations):
  X = env.get_state_history_as_array()
  X, Xp1 = X[:-1, :], X[1:, :]
  target = np.hstack(env.R)

  # Fit one-step q fn
  reg = regressor(n_estimators=10)
  reg.fit(X, target)

  # Fit longer-horizon q fns
  for k in range(number_of_value_iterations):
    q_max, _ = maximize_q_function_at_block(reg.predict, Xp1, env)
    target += gamma * q_max
    reg.fit(X, target)

  # Maximize final q iterate to get next action
  _, list_of_optimal_actions = maximize_q_function_at_block(reg.predict, X, env)

  # Last entry of list gives optimal action at current state
  optimal_action = list_of_optimal_actions[-1]
  
  return optimal_action


def fitted_q_step1(env, gamma, regressor, number_of_value_iterations):
  X, SX, _ = env.get_state_history_as_array()
#  X, Xp1 = X[:-1, :], X[1:, :]
  Xp, SXp = X[1:, :], SX[1:, :] ## Xp include actions, SXp doesn't 
  target = np.hstack(env.R)

  # Fit one-step q fn
  reg = regressor(n_estimators=10)
  reg.fit(Xp, target)
#  k=0
#  x0 = np.concatenate(([1], list(env.S[k][-1,:]),list( env.S[k][-2,:]), 
#                       [env.last_action[k]],[0]))
#  x1 = np.concatenate(([1], list(env.S[k][-1,:]),list( env.S[k][-2,:]), 
#                       [env.last_action[k]],[1]))  
#  reg.fit(Xp1, target)
#  np.append(1, env.S[-1,:], env.S[-2,:], 0, 
  
#  np.concatenate(([1], list(env.S[k][-1,:]),list( env.S[k][-2,:]), 
#                  [env.last_action[k]],[0])) # curent x for the kth patients
  
#  reg.fit(X, target)
#
#  x0, x1 = env.get_state_at_action(0, Xp1[-1,]), env.get_state_at_action(1, Xp1[-1,])
  x0, x1 = np.append(SXp[-1,], 0), np.append(SXp[-1,], 1)
  q0, q1 = reg.predict(x0.reshape(1, -1)), reg.predict(x1.reshape(1, -1))
  print(q0- q1)
  # Last entry of list gives optimal action at current state
  optimal_action = np.argmax([q0, q1])
  
  return optimal_action


def fitted_q_step1_mHealth(env, gamma, regressor, number_of_value_iterations):
  X, SX, _ = env.get_state_history_as_array()
  Xp = np.delete(X, np.arange(0, X.shape[0], X.shape[0]/env.nPatients), axis=0)
#  SXp = np.delete(SX, np.arange(0, SX.shape[0], env.nPatients), axis=0)
  target = np.hstack(env.R)
  current_sx = env.get_current_SX()
  # Fit one-step q fn
  reg = regressor()
  reg.fit(Xp[:,1:], target)

  x0 = np.hstack((current_sx[:,1:], np.zeros((env.nPatients,1))))
  x1 = np.hstack((current_sx[:,1:], np.ones((env.nPatients,1))))
  q0, q1 = reg.predict(x0), reg.predict(x1)
  # Last entry of list gives optimal action at current state
  optimal_actions = np.argmax(np.vstack((q0, q1)), axis=0)
  return optimal_actions


def fitted_q_step1_mHealth_vanilla(env, times=100, sampling=False, AR1=True):
  '''
  :param env:
  :param times:
  :param AR1: True means use AR(1) model to do FQI
  :return: average rewards based on the new glucose draw from its distribution
  '''
  X, Sp1 = env.get_state_transitions_as_x_y_pair()
  if AR1:
    ## Model Misspecification: Only model the current states and action, instead of two time steps
    X = np.delete(X, [4,5,6,7], axis=1) 
  regr = condition_dist_of_next_state2(X, Sp1, sampling=sampling)
  beta_hat = regr['beta_hat']
  sigma_hat = regr['sigma_hat']
  current_sx = env.get_current_SX()
  x0 = np.hstack((current_sx, np.zeros((env.nPatients, 1))))
  x1 = np.hstack((current_sx, np.ones((env.nPatients, 1))))
  if len(beta_hat) == 5:
    x0 = np.delete(x0, [4,5,6,7], axis=1)
    x1 = np.delete(x1, [4,5,6,7], axis=1)
  mean0 = np.dot(beta_hat, x0.T)
  mean1 = np.dot(beta_hat, x1.T)
  cov = sigma_hat**2*np.eye(env.nPatients)
  ## why do the following??? instead of just use mean1 and mean0 to compare???? ## 
  sample_glucose_new0 = np.random.multivariate_normal(mean0, cov, size=times)
  sample_glucose_new1 = np.random.multivariate_normal(mean1, cov, size=times)
  q0 = env.reward_funciton_mHealth(sample_glucose_new0)
  q1 = env.reward_funciton_mHealth(sample_glucose_new1)
  optimal_actions = np.argmax(np.vstack((q0, q1)), axis=0)
  return optimal_actions


def mdp_epsilon_policy(optimal_actions, tuning_function, tuning_function_parameter, time_horizon,
                      t):
  epsilon = tuning_function(time_horizon, t, tuning_function_parameter)
  random_action_index = (np.random.rand(len(optimal_actions)) < epsilon)
  optimal_actions[random_action_index] = np.random.choice(2, size=sum(random_action_index))
  return optimal_actions


def condition_dist_of_next_state(X, Y, env):
  '''
  Y: dependent variables matrix
  '''
  n = X.shape[0]
#  reg = LinearRegression(fit_intercept=False).fit(X, Y)
  
#  R_sq = reg.score(X, Y)
  # Maximum Likelihood Estimate of Error Covariance matrix (slide 61 on http://users.stat.umn.edu/~helwig/notes/mvlr-Notes.pdf)
#  Y_hat = np.matmul(X, reg.coef_.T)
#  Xprime_X_inv = np.linalg.inv(np.matmul(X.T, X))
  env.get_Xprime_X_inv()
  Xprime_X_inv = env.Xprime_X_inv
  beta_hat = np.matmul(Xprime_X_inv, np.matmul(X.T, Y))
  Y_hat = np.matmul(X, beta_hat)

  Sigma_hat = (np.matmul(Y.T, Y) - np.matmul(Y_hat.T, Y_hat))/n
#  beta_hat = np.hstack((reg.intercept_.reshape(-1,1), reg.coef_))  
  # matrix H: X(X'X)^{-1}X'
#  X_include_intercept  =  np.X
  # estimated sampling covariance matrix of beta_hat
  sampling_cov = np.kron(Sigma_hat, Xprime_X_inv)
  
  return {'Sigma_hat': Sigma_hat, 'sampling_cov':sampling_cov,# 'R_sq': R_sq, 'reg': reg, 
          'Xprime_X_inv': Xprime_X_inv, 'beta_hat': beta_hat.T}


def condition_dist_of_next_state2(X, Y, sampling=True):
  n, p = X.shape
  prob_food = 1 - sum(X[:,2]==0)/float(n)
  prob_activity = 1 - sum(X[:,3]==0)/float(n)
  X_food = X[X[:,2]!=0, 2]
  n_effect_food = X_food.shape[0]
  X_activity = X[X[:,3]!=0, 3]
  n_effect_activity = X_activity.shape[0]
  mu_food, sigma_food = np.mean(X_food), np.std(X_food)
  mu_activity, sigma_activity = np.mean(X_activity), np.std(X_activity)

  y = Y[:,0]
  reg = LinearRegression(fit_intercept=False).fit(X, y)
  beta_hat = reg.coef_
  #  print(abs(beta_hat-beta_hat1))
  y_hat = np.dot(X, beta_hat)
  var_hat = np.sum((y - y_hat) ** 2) / (n - p)
  if sampling == True:
  #  env.get_Xprime_X_inv()
#    pdb.set_trace()
    Xprime_X_inv = np.linalg.inv(np.matmul(X.T, X))
#  beta_hat1 = np.dot(Xprime_X_inv, np.dot(X.T, y))

  ### Sampling distributions for each estimator
    sampling_cov = var_hat * Xprime_X_inv
    sampling_sigma_food = sigma_food/np.sqrt(n_effect_food)
    sampling_sigma_activity = sigma_activity/np.sqrt(n_effect_activity)
    sampling_prob_food = np.sqrt(prob_food*(1-prob_food)/n)
    sampling_prob_activity = np.sqrt(prob_activity*(1-prob_activity)/n)
#  R_sq = reg.score(X, y)
#  env.get_Xprime_X_inv()
#  
#  beta_hat = np.dot(Xprime_X_inv, np.dot(X.T, y))
#  yhat = np.dot(X, beta_hat)
#  sigma_hat = np.sum((yhat - y)**2) / (n - p)
  # estimated sampling covariance matrix of beta_hat
#  sampling_cov = np.kron(Sigma_hat, Xprime_X_inv)
  
    return {'sigma_hat': np.sqrt(var_hat), #'R_sq': R_sq, #'Xprime_X_inv': Xprime_X_inv,  'reg': reg,
          'beta_hat':beta_hat, 'mu_food': mu_food, 'sigma_food':sigma_food, 'prob_food':prob_food, 
          'mu_activity':mu_activity, 'sigma_activity':sigma_activity, 'prob_activity':prob_activity,
          'sampling_cov':sampling_cov, 'sampling_sigma_food':sampling_sigma_food,
          'sampling_sigma_activity':sampling_sigma_activity,'sampling_prob_food':sampling_prob_food,
          'sampling_prob_activity':sampling_prob_activity,'n_effect_food':n_effect_food,'n_effect_activity':n_effect_activity}
  return {'sigma_hat': np.sqrt(var_hat), 'beta_hat':beta_hat}


def check_coef_converge(nPatients=15, T=100):
  env = Glucose(nPatients)
  env.reset()
  for t in range(T):
    env.step(np.random.choice(2, size=nPatients))
    X, Sp1 = env.get_state_transitions_as_x_y_pair()
    regr = condition_dist_of_next_state2(X, Sp1)
    beta_hat = regr['beta_hat']
    sigma_hat, mu_food, sigma_food, prob_food, \
    mu_activity, sigma_activity, prob_activity = regr['sigma_hat'], \
    regr['mu_food'], regr['sigma_food'], regr['prob_food'],    \
    regr['mu_activity'], regr['sigma_activity'], regr['prob_activity']
#    est_conditional_mean = np.dot(beta_hat, x_initial)
#    true_conditional_mean = np.dot(np.array([10, 0.9, 0.1, -0.01, 0.0, 0.1, -0.01, -10, -4]), x_initial)
#    print(np.abs(est_conditional_mean - true_conditional_mean))
#    print('time {}'.format(t))
    print(beta_hat, sigma_hat, mu_food, sigma_food, prob_food, \
  mu_activity, sigma_activity, prob_activity)
  print(np.array([10, 0.9, 0.1, -0.01, 0.0, 0.1, -0.01, -10, -4]), 5, np.array([0,10,0.2]), np.array([0,10,0.2]))
  return


def mdp_glucose_rollout(tuning_function_parameter, mdp_epsilon_policy, 
                             time_horizon, x_initial, tuning_function, env, env_approx, 
                             gamma, number_of_value_iterations, mc_replicates):
  X, Sp1 = env.get_state_transitions_as_x_y_pair()
  n, X_dim = X.shape
  S_dim = Sp1.shape[1]
  multi_regr = condition_dist_of_next_state(X, Sp1, env)
  beta_hat = multi_regr['beta_hat']
  Sigma_hat = multi_regr['Sigma_hat']
  sample_beta_hat = np.zeros((S_dim, X_dim))
  mean_cumulative_reward = 0
  for rep in range(mc_replicates):
    ## Sample beta_hat and Sigma_hat from their corresponding sampling distributions.
    sample_beta_hat = (np.random.multivariate_normal(np.hstack(beta_hat),
                     multi_regr['sampling_cov'])).reshape(S_dim, X_dim)
    sample_Sigma_hat = wishart.rvs(df = n-X_dim, scale = Sigma_hat)/n
    sim_env = env_approx(time_horizon, x_initial, sample_beta_hat, sample_Sigma_hat)
    sim_env.reset()
    r = 0
    for t in range(time_horizon):
#      if t==30:
#      print(t, sim_env.current_state, sample_beta_hat)
      optimal_action = fitted_q_step1(sim_env, gamma, RandomForestRegressor, number_of_value_iterations=number_of_value_iterations)
      action = mdp_epsilon_policy(optimal_action, tuning_function, tuning_function_parameter, time_horizon, t)
#      print(t, action, sim_env.current_state[0])
      _, reward = sim_env.step(action)
      r += reward
    
    mean_cumulative_reward += (r - mean_cumulative_reward)/(rep+1)
  return mean_cumulative_reward
 

def mdp_glucose_mHealth_rollout(tuning_function_parameter, mdp_epsilon_policy, 
                               time_horizon, x_initials, sx_initials, tuning_function, env, 
                               gamma, number_of_value_iterations, mc_replicates, AR1):
  X, Sp1 = env.get_state_transitions_as_x_y_pair()
  if AR1:
    X = np.delete(X, [4,5,6,7], axis=1)
  n, X_dim = X.shape
#  S_dim = Sp1.shape[1]
  regr = condition_dist_of_next_state2(X, Sp1)
  beta_hat = regr['beta_hat']
  sigma_hat = regr['sigma_hat']
  mu_food = regr['mu_food']
  sigma_food = regr['sigma_food']
  prob_food = regr['prob_food']
  mu_activity = regr['mu_activity']
  sigma_activity = regr['sigma_activity']
  prob_activity = regr['prob_activity']
  sampling_cov = regr['sampling_cov']
  sampling_sigma_food = regr['sampling_sigma_food']
  sampling_sigma_activity = regr['sampling_sigma_activity']
  sampling_prob_food = regr['sampling_prob_food']
  sampling_prob_activity = regr['sampling_prob_activity']
  n_effect_activity = regr['n_effect_activity']
  n_effect_food = regr['n_effect_food']

  mean_cumulative_reward = 0
  for rep in range(mc_replicates):
    '''
    # check again the following sample sigma's
    '''
    ### Sample beta_hat and Sigma_hat from their corresponding sampling distributions.
    sample_beta_hat = np.random.multivariate_normal(beta_hat, sampling_cov)
    sample_sigma_hat = np.sqrt(chi2.rvs(df=n-X_dim, size=1)/(n-X_dim)) * (sigma_hat)
    sample_prob_food = np.random.normal(prob_food, sampling_prob_food)
    sample_mu_food = np.random.normal(mu_food, sampling_sigma_food)
    sample_sigma_food  = np.sqrt(chi2.rvs(df=n_effect_food-1, size=1)/(n_effect_food-1) )* sigma_food
    sample_prob_activity = np.random.normal(prob_activity, sampling_prob_activity)
    sample_mu_activity = np.random.normal(mu_activity, sampling_sigma_activity)
    sample_sigma_activity = np.sqrt(chi2.rvs(df=n_effect_activity-1, size=1)/(n_effect_activity-1) )* sigma_activity
    '''
    # check again the x_initials and sx_initials
    '''
    sim_env = Glucose(env.nPatients, COEF = sample_beta_hat, SIGMA_NOISE = sample_sigma_hat, 
               prob_food =sample_prob_food, MU_FOOD = sample_mu_food, SIGMA_FOOD = sample_sigma_food, 
               prob_activity = sample_prob_activity, MU_ACTIVITY = sample_mu_activity, 
               SIGMA_ACTIVITY = sample_sigma_activity, x_initials=x_initials, sx_initials=sx_initials)
    sim_env.reset()
    r = 0
    for t in range(time_horizon):
#      if t==30:
#      print(t, sim_env.current_state, sample_beta_hat)
      optimal_actions = fitted_q_step1_mHealth_vanilla(sim_env, AR1=AR1)
#      optimal_actions = fitted_q_step1_mHealth(sim_env, gamma, RandomForestRegressor, number_of_value_iterations)
      actions = mdp_epsilon_policy(optimal_actions, tuning_function, tuning_function_parameter, time_horizon, t)
#      print(t, action, sim_env.current_state[0])
      _, reward = sim_env.step(actions)
      r += reward
    
    mean_cumulative_reward += (r - mean_cumulative_reward)/(rep+1)
  return mean_cumulative_reward
     
  
def bayesopt(rollout_function, policy, tuning_function, zeta_prev, time_horizon, env, mc_replicates,
             bounds, explore_, x_initial, sx_initial, number_of_value_iterations, gamma, AR1):

  # Assuming 10 params!
  # def objective(zeta0, zeta1, zeta2, zeta3, zeta4, zeta5, zeta6, zeta7, zeta8, zeta9):
  # zeta = np.array([zeta0, zeta1, zeta2, zeta3, zeta4, zeta5, zeta6, zeta7, zeta8, zeta9])

  def objective(zeta0, zeta1, zeta2):
    zeta = np.array([zeta0, zeta1, zeta2])
    return rollout_function(zeta, policy, time_horizon, x_initial, sx_initial, tuning_function, 
                            env, gamma, number_of_value_iterations, mc_replicates, AR1)

  # bounds = {'zeta{}'.format(i): (lower_bound, upper_bound) for i in range(10)}
  explore_.update({'zeta{}'.format(i): [zeta_prev[i]] for i in range(len(zeta_prev))})
  bo = BayesianOptimization(objective, bounds)
  bo.explore(explore_)
  bo.maximize(init_points=10, n_iter=15, alpha=1e-4)
  best_param = bo.res['max']['max_params']
  best_param = np.array([best_param['zeta{}'.format(i)] for i in range(len(bounds))])
  return best_param


def rollout_under_true_model(tuning_function_parameter, mdp_epsilon_policy, 
                             tuning_function, time_horizon=50, mc_replicates=10,
                             gamma=0.9, number_of_value_iterations=0):
  env = Glucose(time_horizon)
  mean_cumulative_reward = 0
  for rep in range(mc_replicates):
    env.reset()
    r = 0
    for t in range(time_horizon):
      optimal_action = fitted_q_step1_mHealth_vanilla(env)
#      optimal_action = fitted_q_step1(env, gamma, RandomForestRegressor, number_of_value_iterations=number_of_value_iterations)
      action = mdp_epsilon_policy(optimal_action, tuning_function, tuning_function_parameter, time_horizon, t)
      _, reward = env.step(action)
      r += reward
    
    mean_cumulative_reward += (r - mean_cumulative_reward)/(rep+1)
  return mean_cumulative_reward


def rollout_under_true_model_mHealth(tuning_function_parameter, mdp_epsilon_policy, 
                             tuning_function, time_horizon=50, mc_replicates=10,
                             gamma=0.9, number_of_value_iterations=0, nPatients=15):
  env = Glucose(nPatients)
  mean_cumulative_reward = 0
  for rep in range(mc_replicates):
    env.reset()
    r = 0
    for t in range(time_horizon):
      optimal_action = fitted_q_step1_mHealth_vanilla(env)
#      optimal_action = fitted_q_step1_mHealth(env, gamma, RandomForestRegressor, number_of_value_iterations=number_of_value_iterations)
      action = mdp_epsilon_policy(optimal_action, tuning_function, tuning_function_parameter, time_horizon, t)
      _, reward = env.step(action)
      r += reward
    
    mean_cumulative_reward += (r - mean_cumulative_reward)/(rep+1)
  return mean_cumulative_reward


def bayesopt_under_true_model(T):
  rollout_function = rollout_under_true_model_mHealth
  policy = mdp_epsilon_policy    
  bounds = {'zeta0': (0.8, 2.0), 'zeta1': (1.0, 49.0), 'zeta2': (0.01, 2.5)}
  tuning_function = tuned_bandit.expit_epsilon_decay
  
  def objective(zeta0, zeta1, zeta2):
    zeta = np.array([zeta0, zeta1, zeta2])
    return rollout_function(zeta, policy, tuning_function, time_horizon=T)
  
  # bounds = {'zeta{}'.format(i): (lower_bound, upper_bound) for i in range(10)}
  explore_ = {'zeta0': [1.0, 0.05, 1.0, 0.1], 'zeta1': [50.0, 49.0, 1.0, 49.0], 'zeta2': [0.1, 2.5, 1.0, 2.5]}
  bo = BayesianOptimization(objective, bounds)
  bo.explore(explore_)
  bo.maximize(init_points=10, n_iter=15, alpha=1e-4)
  best_param = bo.res['max']['max_params']
  best_param = np.array([best_param['zeta{}'.format(i)] for i in range(len(bounds))])
  return best_param


def episode(policy_name, label, mc_replicates=10, T=50, nPatients=15, AR1=False):
  np.random.seed(label)
  if policy_name == 'eps':
    tuning_function = lambda a, b, c: 0.05  # Constant epsilon
    tune = False
    tuning_function_parameter = None
  elif policy_name == "eps-decay":
    tuning_function = tuned_bandit.expit_epsilon_decay
    tune = True
    tuning_function_parameter = np.array([0.05, 1.0, 0.01]) 
    bounds = {'zeta0': (0.8, 2.0), 'zeta1': (1.0, 49.0), 'zeta2': (0.01, 2.5)}
    explore_ = {'zeta0': [1.0, 0.05, 1.0, 0.1], 'zeta1': [50.0, 49.0, 1.0, 49.0], 'zeta2': [0.1, 2.5, 1.0, 2.5]}
    rollout_function = mdp_glucose_mHealth_rollout
  elif policy_name == 'eps-fixed-decay':
    if T==25:
      tuning_function = lambda a, b, c: 0.7**b
    elif T==50:
      tuning_function = lambda a, b, c: 0.8**b
#    tuning_function = lambda a, b, c: 0.3/float(b+1)
    #tune = False
    #tuning_function_parameter = None
#    tuning_function = tuned_bandit.expit_epsilon_decay
    tune = False
    tuning_function_parameter = np.array([ 2. ,17.40051652, 2.5])
#    tuning_function_parameter = np.array([ 2., 41.68182633, 2.5])

  policy = mdp_epsilon_policy    
  gamma = 0.9
  number_of_value_iterations = 0
  env = Glucose(nPatients)
  env.reset()
  _,_ = env.step(np.random.choice(2, size=nPatients))
#  print(env.get_state_history_as_array())
#  for i in range(11):
#    x_initial, reward, done = env.step(np.array([0]))
#    print('time {}: reward {}; glucose {}'.format(i, reward, x_initial[1]))
  
  ### current state features (not include current actions, since they are unknown) for all patients
  X, SX, _ = env.get_state_history_as_array()
  nx = X.shape[0]
  x_initials = X[np.arange(1, nx, int(nx/env.nPatients)), ] # an array
  sx_initials = SX[np.arange(1, nx, int(nx/env.nPatients)), ] # an array
#  print(x_initial)
  rewards = np.zeros(T)
  actions_array = np.zeros((nPatients, T))
  tuning_parameter_sequence = []
  for t in range(T):
#    print("******** replicate: ", label, ", timestep: ", t, " ********")
    if tune:
      tuning_function_parameter = bayesopt(rollout_function, policy, tuning_function, tuning_function_parameter, 
                                           T, env, mc_replicates, bounds, explore_, x_initials, sx_initials,
                                           number_of_value_iterations, gamma, AR1)
#      print("epsilon {}".format(tuning_function(T, t, tuning_function_parameter)))
      tuning_parameter_sequence.append([float(z) for z in tuning_function_parameter]) 
#    print('time {}, tuning_function_parameter {}'.format(t, tuning_function_parameter))
    optimal_actions = fitted_q_step1_mHealth_vanilla(env, AR1=AR1)
#    optimal_actions = fitted_q_step1_mHealth(env, gamma, RandomForestRegressor, number_of_value_iterations=number_of_value_iterations)
#    print(optimal_actions)
    actions = policy(optimal_actions, tuning_function, tuning_function_parameter, T, t)
    x_initials, reward = env.step(actions)
    sx_initials = env.get_current_SX()
#    print(mc_replicates, t, sx_initials[:,1])
#    print('time {}: reward {}; action {}, glucose {}'.format(time, reward, action, x_initial[1]))
    rewards[t] = reward
    actions_array[:, t] = actions
#    pdb.set_trace()
    results = {'T':T, 'mc_replicates': mc_replicates, 't':t, 'cum_rewards': float(sum(rewards)), 
               'reward':float(reward)}
             #  'rewards': float(np.mean(cum_rewards)), 'se_rewards':float(np.std(cum_rewards)/np.sqrt(replicates)),
               #'zeta_sequences': list(tuning_function_parameter)}#, 'actions': actions}#, 'rewards':rewards}

#    base_name = 'mdp-glucose-{}'.format(policy_name)
#    prefix = os.path.join(project_dir, 'src', 'environments', base_name)
#    suffix = nPatients
#    filename = '{}_{}_T{}_nMC().yml'.format(prefix, suffix, T, mc_replicates)
#    #np.save('{}_{}'.format(prefix, suffix), results)
#    with open(filename, 'a') as outfile:
#      yaml.dump(results, outfile)
    
#  print('cum_rewards {}, rewards {}'.format( sum(rewards), rewards) )
  print('replicate {} cum_rewards {}'.format(label, sum(rewards)))
#  print(type(rewards))
#  plt.plot(rewards)
#  plt.title(policy_name)
#  plt.show()
  return {'rewards':rewards, 'cum_rewards': sum(rewards), 'zeta_sequence': tuning_parameter_sequence,
          'actions': actions_array}
    
      
def run(policy_name, save=True, mc_replicates=10, T=50, AR1=True):
  """

  :return:
  """

#  replicates = 48
#  num_cpus = int(mp.cpu_count())
  num_cpus = 4
  replicates = 96
  results = []
  pool = mp.Pool(processes=num_cpus)

  episode_partial = partial(episode, policy_name, mc_replicates=mc_replicates, T=T, AR1=AR1)

  results = pool.map(episode_partial, range(replicates))
#  pdb.set_trace()
#  cumulative_regrets = [np.float(d['cumulative_regret']) for d in results]
  zeta_sequences = [list(d['zeta_sequence']) for d in results]
  actions = [list(d['actions']) for d in results]
  cum_rewards = [float(d['cum_rewards']) for d in results]
#  rewards = [list(d['rewards'].astype(float)) for d in results]
  print(policy_name, 'rewards', float(np.mean(cum_rewards)), 'se_rewards',float(np.std(cum_rewards))/np.sqrt(replicates))
  # Save results
  if save:
    results = {'T':T, 'mc_replicates': mc_replicates, 'AR1': AR1, 'cum_rewards': cum_rewards, 
               'rewards': float(np.mean(cum_rewards)), 'se_rewards':float(np.std(cum_rewards)/np.sqrt(replicates)),
               'zeta_sequences': zeta_sequences, 'actions': actions}#, 'rewards':rewards}

    base_name = 'mdp-glucose-{}-AR1_{}-T_{}'.format(policy_name, AR1, T)
    prefix = os.path.join(project_dir, 'src', 'environments', base_name)
    suffix = datetime.datetime.now().strftime("%y%m%d_%H%M%S")
    filename = '{}_{}.yml'.format(prefix, suffix)
    np.save('{}_{}'.format(prefix, suffix), results)
    with open(filename, 'w') as outfile:
      yaml.dump(results, outfile)

  return


if __name__ == '__main__':
  start_time = time.time()
#  check_coef_converge()
#  episode('eps-decay', 0, T=5)
#  episode('eps-fixed-decay', 0, T=5)
#  run('eps-decay', T= 25, mc_replicates=10, AR1=False)
  run('eps-fixed-decay', T=50, mc_replicates=10, AR1=False)
#  run('eps',save=False, T=25, mc_replicates=10, AR1=False)
#  episode('eps', 0, T=25)
#  result = episode('eps', 0, T=50)
  # print(result['actions'])
 # episode('eps-fixed-decay', 1, T=50)
#  num_processes = 4
#  num_replicates = num_processes
#  pool = mp.Pool(num_processes)
#  params = pool.map(bayesopt_under_true_model, range(num_processes))
#  params_dict = {str(i): params[i].tolist() for i in range(len(params))}
#  with open('bayes-opt-glucose.yml', 'w') as handle:
#    yaml.dump(params_dict, handle)
#  print(bayesopt_under_true_model(T=25))
  elapsed_time = time.time() - start_time
  print("time {}".format(elapsed_time))
  # episode('ts-decay-posterior-sample', 0, T=10, mc_replicates=100)
  # episode('ucb-tune-posterior-sample', 0, T=10, mc_replicates=100)
  # run('ts-decay-posterior-sample', T=10, mc_replicates=100)
  # run('ucb-tune-posterior-sample', T=10, mc_replicates=100)
  
  
  
  
  
  
  
  
  
