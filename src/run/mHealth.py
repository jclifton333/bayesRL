"""
Mimic M-health by normal linear contextual bandit for multiple ''patients'' at each time step.
"""

import sys
import pdb
import os
this_dir = os.path.dirname(os.path.abspath(__file__))
project_dir = os.path.join(this_dir, '..', '..')
sys.path.append(project_dir)


import matplotlib.pyplot as plt
from src.environments.Bandit import NormalCB
from src.policies import rollout
import src.policies.global_optimization as opt
from src.policies import tuned_bandit_policies as tuned_bandit
from src.policies import reference_policies as ref
import copy
import numpy as np
from scipy.linalg import block_diag
from functools import partial
import datetime
import yaml
import multiprocessing as mp


def episode(policy_name, label, save=False, points_per_grid_dimension=50, monte_carlo_reps=1000):
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
    policy = tuned_bandit.linear_cb_epsilon_greedy_policy
    tune = False
    tuning_function_parameter = None
  elif policy_name == 'eps-decay':
    tuning_function = tuned_bandit.expit_epsilon_decay
    policy = tuned_bandit.linear_cb_epsilon_greedy_policy
    tune = True
    tuning_function_parameter = np.array([0.2, -2, 1])
  elif policy_name == 'greedy':
    tuning_function = lambda a, b, c: 0.00  # Constant epsilon
    policy = tuned_bandit.linear_cb_epsilon_greedy_policy
    tune = False
    tuning_function_parameter = None
  elif policy_name == 'worst':
    tuning_function = lambda a, b, c: 0.00
    policy = ref.linear_cb_worst_policy
    tune = False
    tuning_function_parameter = None
  # elif policy_name == 'ts':
  #   tuning_function = lambda a, b, c: 1.0  # No shrinkage
  #   policy = tuned_bandit.thompson_sampling_policy
  #   tune = False
  #   tuning_function_parameter = None
  # elif policy_name == 'ts-shrink':
  #   tuning_function = tuned_bandit.expit_truncate
  #   policy = tuned_bandit.thompson_sampling_policy
  #   tune = True
  #   tuning_function_parameter = np.array([-2, 1])
  else:
    raise ValueError('Incorrect policy name')

  env = NormalCB(list_of_reward_betas=[np.array([1.0, 1.0]), np.array([2.0, -2.0])])
  cumulative_regret = 0.0
  nPatients = 10
  env.reset()

  # Initial assignments
  for t in range(10):
    for j in range(5):
      env.step(0)
    for j in range(5):
      env.step(1)

  for t in range(T):
    X = env.X
    estimated_context_mean = np.mean(X, axis=0)
    estimated_context_variance = np.cov(X, rowvar=False)
    if tune:
      tuning_function_parameter = opt.grid_search(rollout.mHealth_rollout, policy, tuning_function,
                                                  tuning_function_parameter,
                                                  T, t, estimated_context_mean,
                                                  estimated_context_variance, env, nPatients,
                                                  points_per_grid_dimension, monte_carlo_reps)
    # print('time {} epsilon {}'.format(t, tuning_function(T,t,tuning_function_parameter)))
    for j in range(nPatients):
      x = copy.copy(env.curr_context)

      beta_hat = env.beta_hat_list
      action = policy(beta_hat, env.sampling_cov_list, x, tuning_function, tuning_function_parameter, T, t, env)
      env.step(action)

      # Compute regret
      expected_rewards = np.max([env.expected_reward(a, env.curr_context)
                                        for a in range(env.number_of_actions)])
      expected_reward_at_action = expected_rewards[action]
      optimal_expected_reward = np.max(expected_rewards)
      regret = optimal_expected_reward - expected_reward_at_action
      cumulative_regret += regret
  
    # Save results
    if save:
      results = {'t': float(t), 'regret': float(cumulative_regret)}
      with open(filename, 'w') as outfile:
        yaml.dump(results, outfile)

  return cumulative_regret


def run(policy_name, save=True, points_per_grid_dimension=10, monte_carlo_reps=1000):
  """

  :return:
  """

  num_cpus = int(mp.cpu_count())
  pool = mp.Pool(processes=num_cpus)

  episode_partial = partial(episode, policy_name, save=False, points_per_grid_dimension=points_per_grid_dimension,
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
  # episode('worst', np.random.randint(low=1, high=1000))
  run('worst')

