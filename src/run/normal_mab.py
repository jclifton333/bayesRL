import sys
import pdb
import os
this_dir = os.path.dirname(os.path.abspath(__file__))
project_dir = os.path.join(this_dir, '..', '..')
sys.path.append(project_dir)


from src.policies import tuned_bandit_policies as tuned_bandit
from src.policies import rollout
from src.environments.Bandit import NormalMAB
import src.policies.global_optimization as opt
import numpy as np
from functools import partial
import datetime
import yaml
import multiprocessing as mp


def episode(policy_name, label, save=False, points_per_grid_dimension=50, monte_carlo_reps=1000):
  if save:
    base_name = 'normal-mab-{}-{}'.format(label, policy_name)
    prefix = os.path.join(project_dir, 'src', 'run', 'results', base_name)
    suffix = datetime.datetime.now().strftime("%y%m%d_%H%M%S")
    filename = '{}_{}.yml'.format(prefix, suffix)

  np.random.seed(label)
  nPatients = 10
  T = 10

  # ToDo: Create policy class that encapsulates this behavior
  if policy_name == 'eps':
    tuning_function = lambda a, b, c: 0.05  # Constant epsilon
    policy = tuned_bandit.mab_epsilon_greedy_policy
    tune = False
    tuning_function_parameter = None
  elif policy_name == 'eps-decay':
    tuning_function = tuned_bandit.expit_epsilon_decay
    policy = tuned_bandit.mab_epsilon_greedy_policy
    tune = True
    tuning_function_parameter = np.array([0.2, -2, 1])
  else:
    raise ValueError('Incorrect policy name')

  env = NormalMAB(list_of_reward_mus=[[1], [2]], list_of_reward_vars=[[1], [1]])
  cumulative_regret = 0.0
  mu_opt = np.max(env.list_of_reward_mus)
  env.reset()

  # Initial pulls (so we can fit the models)
  for a in range(env.number_of_actions):
    env.step(a)

  for t in range(T):
    print(t)
    if tune:
      tuning_function_parameter = opt.mab_grid_search(rollout.normal_mab_rollout, policy, tuning_function,
                                                      tuning_function_parameter, T, t, env, nPatients,
                                                      points_per_grid_dimension, monte_carlo_reps)

    print('time {} epsilon {}'.format(t, tuning_function(T, t, tuning_function_parameter)))
    action = policy(env, tuning_function, tuning_function_parameter, T, t)
    env.step(action)

    # Compute regret
    regret = mu_opt - env.list_of_reward_mus[action]
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


