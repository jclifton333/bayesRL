import sys
import pdb
import os
this_dir = os.path.dirname(os.path.abspath(__file__))
project_dir = os.path.join(this_dir, '..', '..')
sys.path.append(project_dir)


import matplotlib.pyplot as plt
from src.environments.Bandit import NormalCB
from src.policies import tuned_bandit_policies as tuned_bandit
import copy
import numpy as np
from scipy.linalg import block_diag
from functools import partial
import datetime
import yaml
import multiprocessing as mp


def episode(policy_name, label, save=True, points_per_grid_dimension=1, monte_carlo_reps=1):
  if save:
    base_name = 'mhealth-{}-{}'.format(label, policy_name)
    prefix = os.path.join(project_dir, 'src', 'run', 'results', base_name)
    suffix = datetime.datetime.now().strftime("%y%m%d_%H%M%S")
    filename = '{}_{}.yml'.format(prefix, suffix)

  np.random.seed(label)
  T = 10

  # ToDo: Create policy class that encapsulates this behavior
  if policy_name == 'eps':
    tuning_function = lambda a, b, c: 0.05  # Constant epsilon
    policy = tuned_bandit.epsilon_greedy_policy
    tune = False
    tuning_function_parameter = None
  elif policy_name == 'eps-decay':
    tuning_function = tuned_bandit.expit_epsilon_decay
    policy = tuned_bandit.epsilon_greedy_policy
    tune = True
    tuning_function_parameter = np.array([0.2, -2, 1])
  elif policy_name == 'ts':
    tuning_function = lambda a, b, c: 1.0  # No shrinkage
    policy = tuned_bandit.thompson_sampling_policy
    tune = False
    tuning_function_parameter = None
  elif policy_name == 'ts-shrink':
    tuning_function = tuned_bandit.expit_truncate
    policy = tuned_bandit.thompson_sampling_policy
    tune = True
    tuning_function_parameter = np.array([-2, 1])
  else:
    raise ValueError('Incorrect policy name')

  env = NormalCB(list_of_reward_betas=[np.array([1.0, 1.0]), np.array([2.0, -2.0])])
  cumulative_regret = 0.0
  nPatients = 10
  env.reset()

  # Initial pulls (so we can fit the models)  
  for a in range(env.number_of_actions):
    for p in range(5):
        for j in range(nPatients): 
          env.step(a)

  # Get initial linear model
  X, A, U = env.X, env.A, env.U
  linear_model_results = {'beta_hat_list': [], 'Xprime_X_inv_list': [], 'X_list': [], 'X_dot_y_list': [],
                          'sample_cov_list': [], 'sigma_hat_list': [], 'y_list': []}
  for a in range(env.number_of_actions):  # Fit linear model on data from each actions
    # Get observations where action == a
    indices_for_a = np.where(A == a)
    X_a = X[indices_for_a]
    U_a = U[indices_for_a]

    Xprime_X_inv_a = np.linalg.inv(np.dot(X_a.T, X_a))
    X_dot_y_a = np.dot(X_a.T, U_a)
    beta_hat_a = np.dot(Xprime_X_inv_a, X_dot_y_a)
    yhat_a = np.dot(X_a, beta_hat_a)
    sigma_hat_a = np.sum((U_a - yhat_a) ** 2)
    sample_cov_a = sigma_hat_a * Xprime_X_inv_a

    # Add to linear model results
    linear_model_results['beta_hat_list'].append(beta_hat_a)
    linear_model_results['Xprime_X_inv_list'].append(Xprime_X_inv_a)
    linear_model_results['X_list'].append(X_a)
    linear_model_results['X_dot_y_list'].append(X_dot_y_a)
    linear_model_results['sample_cov_list'].append(sample_cov_a)
    linear_model_results['sigma_hat_list'].append(sigma_hat_a)
    linear_model_results['y_list'].append(U_a)

  for t in range(T):
    X = env.X
    estimated_context_mean = np.mean(X, axis=0)
    estimated_context_variance = np.cov(X, rowvar=False)
    if tune:
      tuning_function_parameter = tuned_bandit.grid_search(tuned_bandit.mHealth_rollout, policy, tuning_function,
                                                           tuning_function_parameter,
                                                           linear_model_results, T, t, estimated_context_mean,
                                                           estimated_context_variance, env, nPatients,
                                                           points_per_grid_dimension, monte_carlo_reps)
    # print('time {} epsilon {}'.format(t, tuning_function(T,t,tuning_function_parameter)))
    for j in range(nPatients):
        # tuning_function_parameter = tuned_bandit.epsilon_greedy_policy_gradient(linear_model_results, T, t,
        #                                                                         estimated_context_mean,
        #                                                                         estimated_context_variance,
        #                                                                         None, None, tuning_function_parameter)
  
      x = copy.copy(env.curr_context)

      beta_hat = np.array(linear_model_results['beta_hat_list'])
      action = policy(beta_hat, linear_model_results['sample_cov_list'], x, tuning_function,
                      tuning_function_parameter, T, t)
      step_results = env.step(action)
      reward = step_results['Utility']
  
      # Compute regret
      oracle_expected_reward = np.max((np.dot(x, env.list_of_reward_betas[0]), np.dot(x, env.list_of_reward_betas[1])))
      regret = oracle_expected_reward - np.dot(x, env.list_of_reward_betas[a])
      cumulative_regret += regret
  
      # Update linear model
      linear_model_results = tuned_bandit.update_linear_model_at_action(a, linear_model_results, x, reward)

    # Save results
    if save:
      results = {'t': float(t), 'regret': float(cumulative_regret)}
      with open(filename, 'w') as outfile:
        yaml.dump(results, outfile)

  return cumulative_regret


def run(policy_name, save=True, points_per_grid_dimension=100, monte_carlo_reps=1000):
  """

  :return:
  """

  num_cpus = int(mp.cpu_count())
  pool = mp.Pool(processes=num_cpus)

  episode_partial = partial(episode, policy_name, save=save, points_per_grid_dimension=points_per_grid_dimension,
                            monte_carlo_reps=monte_carlo_reps)

  results = pool.map(episode_partial, range(num_cpus))

  # Save results
  if save:
    results = {'T': float(10), 'mean_regret': float(np.mean(results)), 'std_regret': float(np.std(results)),
               'regret list': [float(r) for r in results],
               'points_per_grid_dimension': points_per_grid_dimension, 'monte_carlo_reps': monte_carlo_reps}

    base_name = 'mhealth-{}'.format(policy_name)
    prefix = os.path.join(project_dir, 'src', 'run', 'results', base_name)
    suffix = datetime.datetime.now().strftime("%y%m%d_%H%M%S")
    filename = '{}_{}.yml'.format(prefix, suffix)
    with open(filename, 'w') as outfile:
      yaml.dump(results, outfile)

  return


if __name__ == '__main__':
  # episode('eps-decay', np.random.randint(low=1, high=1000))
  run('eps-decay')

