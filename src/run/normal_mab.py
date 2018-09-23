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


def episode(policy_name, label):
  np.random.seed(label)
  T = 1

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

  env = NormalMAB(list_of_reward_mus=[[1],[2]], list_of_reward_vars=[[1],[1]])
  cumulative_regret = 0.0
  env.reset()

  # Initial pulls (so we can fit the models)
  for a in range(env.number_of_actions):
    env.step(a)

  # Get initial linear model
  A, U = env.A, env.U
  for a in range(env.number_of_actions):  # Fit linear model on data from each actions
    # Get observations where action == a
    indices_for_a = np.where(A == a)
    U_a = U[indices_for_a]

  for t in range(T):
    if tune:
      tuning_function_parameter = tuned_bandit.random_search(tuned_bandit.oracle_rollout, policy, tuning_function,
                                                             tuning_function_parameter, T, t, env)
      # tuning_function_parameter = tuned_bandit.epsilon_greedy_policy_gradient(linear_model_results, T, t,
      #                                                                         estimated_context_mean,
      #                                                                         estimated_context_variance,
      #                                                                         None, None, tuning_function_parameter)

    print('time {} epsilon {}'.format(t, tuning_function(T,t,tuning_function_parameter)))
    action = policy(tuning_function, tuning_function_parameter, T, t)
    step_results = env.step(action)
    reward = step_results['Utility']

    # Compute regret
    oracle_expected_reward = np.max( env.list_of_reward_mus[0], env.list_of_reward_mus[1])
    regret = oracle_expected_reward - env.list_of_reward_mus[action]
    cumulative_regret += regret

  return cumulative_regret


def run(policy_name, save=True):
  """

  :return:
  """

  replicates = 80
  num_cpus = int(mp.cpu_count() / 2)
  results = []
  pool = mp.Pool(processes=num_cpus)

  episode_partial = partial(episode, policy_name)

  num_batches = int(replicates / num_cpus)
  for batch in range(num_batches):
    results_for_batch = pool.map(episode_partial, range(batch*num_cpus, (batch+1)*num_cpus))
    results += results_for_batch

  # Save results
  if save:
    results = {'T': float(25), 'mean_regret': float(np.mean(results)), 'std_regret': float(np.std(results)),
                'beta': [[1.0, 1.0], [2.0, -2.0]], 'regret list': [float(r) for r in results]}

    base_name = 'normalcb-25-{}'.format(policy_name)
    prefix = os.path.join(project_dir, 'src', 'run', base_name)
    suffix = datetime.datetime.now().strftime("%y%m%d_%H%M%S")
    filename = '{}_{}.yml'.format(prefix, suffix)
    with open(filename, 'w') as outfile:
      yaml.dump(results, outfile)

  return


if __name__ == '__main__':
  episode('eps-decay', np.random.randint(low=1, high=1000))
  # run('eps-decay')


