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
  target = np.hstack(sim_env.R)

  # Fit one-step q fn
  reg = regressor()
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


def mdp_epsilon_policy(optimal_action, tuning_function, tuning_function_parameter, time_horizon,
                      time):
  epsilon = tuning_function(time_horizon, time, tuning_function_parameter)
  if np.random.rand() < epsilon:
    action = np.random.choice(2)
  else:
    action = optimal_action
  return action


def condition_dist_of_next_state(X, Y):
  '''
  Y: dependent variables matrix
  '''
  n = X.shape[0]
  reg = LinearRegression(fit_intercept=False).fit(X, Y)
  
  R_sq = reg.score(X, Y)
  # Maximum Likelihood Estimate of Error Covariance matrix (slide 61 on http://users.stat.umn.edu/~helwig/notes/mvlr-Notes.pdf)
  Y_hat  = np.matmul(X, reg.coef_.T)
  Xprime_X_inv = np.linalg.inv(np.matmul(X.T, X))
  Var_hat = (np.matmul(Y.T, Y) - np.matmul(Y_hat.T, Y_hat))/n
#  beta_hat = np.hstack((reg.intercept_.reshape(-1,1), reg.coef_))  
#  pdb.set_trace()
  # matrix H: X(X'X)^{-1}X'
#  X_include_intercept  =  np.X
  # estimated sampling covariance matrix of beta_hat
  sampling_cov = np.kron(Sigma_hat, Xprime_X_inv)
  
  return {'Var_hat': Var_hat, 'beta_hat': reg.coef_, 'sampling_cov':sampling_cov, 
          'R_sq': R_sq, 'reg': reg, 'Xprime_X_inv': Xprime_X_inv}

  
def mdp_glucose_rollout(tuning_function_parameter, mdp_epsilon_policy, 
                             time_horizon, x_initial, tuning_function, env, env_approx, 
                             gamma, number_of_value_iterations, mc_replicates):
  X, Sp1 = env.get_state_transitions_as_x_y_pair()
  n, X_dim = X.shape
  S_dim = Sp1.shape[1]
  multi_regr = condition_dist_of_next_state(X, Sp1)
  beta_hat = multi_regr['beta_hat']
  Var_hat = multi_regr['Var_hat']
  sample_beta_hat = np.zeros((S_dim, X_dim))
  mean_cumulative_reward = 0
  for rep in range(mc_replicates):
    ## Sample beta_hat and Sigma_hat from their corresponding sampling distribution
    sample_beta_hat = np.random.multivariate_normal(np.hstack(beta_hat),
                     np.sqrt(multi_regr['sampling_cov'])).reshape(S_dim, X_dim)
    sample_Sigma_hat = wishart.rvs(df = n-X_dim, scale = Var_hat)/n
#    pdb.set_trace()
    sim_env = env_approx(time_horizon, x_initial, sample_beta_hat, sample_Sigma_hat)
    sim_env.reset()
    r = 0
    for time in range(time_horizon):
      optimal_action = fitted_q(sim_env, gamma, RandomForestRegressor, number_of_value_iterations=number_of_value_iterations)
      action = mdp_epsilon_policy(optimal_action, tuning_function, tuning_function_parameter, time_horizon, time)
      _, reward, done = sim_env.step(action)
      r += reward
    
    mean_cumulative_reward += (r - mean_cumulative_reward)/(rep+1)
  return mean_cumulative_reward
    
  
def bayesopt(rollout_function, policy, tuning_function, zeta_prev, time_horizon, env, mc_replicates,
             bounds, explore_, env_approx, x_initial, number_of_value_iterations, gamma):

  # Assuming 10 params!
  # def objective(zeta0, zeta1, zeta2, zeta3, zeta4, zeta5, zeta6, zeta7, zeta8, zeta9):
  # zeta = np.array([zeta0, zeta1, zeta2, zeta3, zeta4, zeta5, zeta6, zeta7, zeta8, zeta9])

  def objective(zeta0, zeta1, zeta2):
    zeta = np.array([zeta0, zeta1, zeta2])
    return rollout_function(zeta, policy, time_horizon, x_initial, tuning_function, 
                            env, env_approx, gamma, number_of_value_iterations, mc_replicates)

  # bounds = {'zeta{}'.format(i): (lower_bound, upper_bound) for i in range(10)}
  explore_.update({'zeta{}'.format(i): [zeta_prev[i]] for i in range(len(zeta_prev))})
  bo = BayesianOptimization(objective, bounds)
  bo.explore(explore_)
  bo.maximize(init_points=10, n_iter=15)
  best_param = bo.res['max']['max_params']
  best_param = np.array([best_param['zeta{}'.format(i)] for i in range(len(bounds))])
  return best_param


def episode(policy_name, label, mc_replicates=1, T=50):
  np.random.seed(label)
  tuning_function = tuned_bandit.expit_epsilon_decay
  tuning_function_parameter = np.array([0.05, 1.0, 0.01]) 
  bounds = {'zeta0': (0.05, 1.0), 'zeta1': (1.0, 49.0), 'zeta2': (0.01, 2.5)}
  explore_ = {'zeta0': [1.0, 0.05, 1.0, 0.1], 'zeta1': [50.0, 49.0, 1.0, 49.0], 'zeta2': [0.1, 2.5, 1.0, 2.5]}
  
  rollout_function = mdp_glucose_rollout
  policy = mdp_epsilon_policy
  env_approx = Glucose_Approx
  gamma = 0.9
  number_of_value_iterations = 8
  env = Glucose(T)
  env.reset()
  for i in range(20):
    env.step(np.random.choice(2))
  x_initial = env.get_state_history_as_array()[-1,:]
  rewards = []
  actions = []
  tuning_parameter_sequence = []
  for time in range(T):
    tuning_function_parameter = bayesopt(rollout_function, policy, tuning_function, tuning_function_parameter, 
                                         T, env, mc_replicates, bounds, explore_, env_approx, x_initial, 
                                         number_of_value_iterations, gamma)
    tuning_parameter_sequence.append([float(z) for z in tuning_function_parameter])
    optimal_action = fitted_q(env, gamma, RandomForestRegressor, number_of_value_iterations=number_of_value_iterations)
    action = policy(optimal_action, tuning_function, tuning_function_parameter, T, time)
    x_initial, reward, done = env.step(action)
    rewards.append(reward)
    actions.append(action)
    
  return {'rewards':rewards, 'zeta_sequence': tuning_parameter_sequence,'actions': actions}
    
      
def run(policy_name='eps-decay', save=True, mc_replicates=1000, T=50):
  """

  :return:
  """

  replicates = 96
  num_cpus = int(mp.cpu_count())
  results = []
  pool = mp.Pool(processes=num_cpus)

  episode_partial = partial(episode, policy_name, mc_replicates=mc_replicates, T=T)

  results = pool.map(episode_partial, range(replicates))
#  cumulative_regrets = [np.float(d['cumulative_regret']) for d in results]
  zeta_sequences = [d['zeta_sequence'] for d in results]
  actions = [d['actions'] for d in results]
  rewards = [d['rewards'] for d in results]
  print(policy_name, 'rewards', float(rewards), 'se_rewards',float(np.std(rewards))/np.sqrt(replicates))
  # Save results
  if save:
    results = {'actions': actions, 'rewards': rewards,
               'zeta_sequences': zeta_sequences}

    base_name = 'mdp-glucose-{}'.format(policy_name)
    prefix = os.path.join(project_dir, 'src', 'run', base_name)
    suffix = datetime.datetime.now().strftime("%y%m%d_%H%M%S")
    filename = '{}_{}.yml'.format(prefix, suffix)
    with open(filename, 'w') as outfile:
      yaml.dump(results, outfile)

  return


if __name__ == '__main__':
  # episode('eps', 50)
  episode('eps-decay', 0, T=50)
  # run('eps-decay', T=50)
  # run('eps', T=50)
  # episode('ts-decay-posterior-sample', 0, T=10, mc_replicates=100)
  # episode('ucb-tune-posterior-sample', 0, T=10, mc_replicates=100)
  # run('ts-decay-posterior-sample', T=10, mc_replicates=100)
  # run('ucb-tune-posterior-sample', T=10, mc_replicates=100)

      
      
  
