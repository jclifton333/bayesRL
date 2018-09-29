import sys
import pdb
import os
this_dir = os.path.dirname(os.path.abspath(__file__))
project_dir = os.path.join(this_dir, '..', '..')
sys.path.append(project_dir)


from src.policies import tuned_bandit_policies as tuned_bandit
from src.policies import gittins_index_policies as gittins
from src.policies import rollout
from src.environments.Bandit import NormalMAB
import src.policies.global_optimization as opt
import numpy as np
from functools import partial
import datetime
import yaml
import multiprocessing as mp


def episode(policy_name, label, save=True, monte_carlo_reps=30):
  if save:
    base_name = 'normal-mab-{}-{}'.format(label, policy_name)
    prefix = os.path.join(project_dir, 'src', 'run', 'results', base_name)
    suffix = datetime.datetime.now().strftime("%y%m%d_%H%M%S")
    filename = '{}_{}.yml'.format(prefix, suffix)

  np.random.seed(label)
  nPatients = 1
  T = 100

  # ToDo: Create policy class that encapsulates this behavior
  if policy_name == 'eps':
    tuning_function = lambda a, b, c: 0.05  # Constant epsilon
    policy = tuned_bandit.mab_epsilon_greedy_policy
    tune = False
    tuning_function_parameter = None
  elif policy_name == 'greedy':
    tuning_function = lambda a, b, c: 0.00  # Constant epsilon
    policy = tuned_bandit.mab_epsilon_greedy_policy
    tune = False
    tuning_function_parameter = None
  elif policy_name == 'random':
    tuning_function = lambda a, b, c: 1.0  # Constant epsilon
    policy = tuned_bandit.mab_epsilon_greedy_policy
    tune = False
    tuning_function_parameter = None
  elif policy_name == 'eps-decay-fixed':
    tuning_function = lambda a, t, c: 0.5 / (t + 1)
    policy = tuned_bandit.mab_epsilon_greedy_policy
    tune = False
    tuning_function_parameter = None
  elif policy_name == 'eps-decay':
    tuning_function = tuned_bandit.stepwise_linear_epsilon
    policy = tuned_bandit.mab_epsilon_greedy_policy
    tune = True
    tuning_function_parameter = np.ones(10)*0.05
  elif policy_name == 'gittins':
    tuning_function = lambda a, b, c: None
    tuning_function_parameter = None
    tune = False
    policy = gittins.normal_mab_gittins_index_policy
  else:
    raise ValueError('Incorrect policy name')

#  env = NormalMAB(list_of_reward_mus=[[1], [1.1]], list_of_reward_vars=[[1], [1]])
  env = NormalMAB(list_of_reward_mus=[[0], [1]], list_of_reward_vars=[[1], [600]])

  cumulative_regret = 0.0
  mu_opt = np.max(env.list_of_reward_mus)
  env.reset()
  tuning_parameter_sequence = []
  # Initial pulls
  for a in range(env.number_of_actions):
    env.step(a)

  for t in range(T):
    print(t)
    if tune:
      sim_env = NormalMAB(list_of_reward_mus=env.estimated_means, list_of_reward_vars=env.estimated_vars)
      pre_simulated_data = sim_env.generate_mc_samples(monte_carlo_reps, T)
      tuning_function_parameter = opt.bayesopt(rollout.mab_rollout_with_fixed_simulations, policy, tuning_function,
                                               tuning_function_parameter, T, env,
                                               {'pre_simulated_data': pre_simulated_data})
      tuning_parameter_sequence.append([float(z) for z in tuning_function_parameter])

    print('time {} epsilon {}'.format(t, tuning_function(T, t, tuning_function_parameter)))
    for j in range(nPatients):
      action = policy(env.estimated_means, env.standard_errors, env.number_of_pulls, tuning_function,
                      tuning_function_parameter, T, t, env)
      env.step(action)

      # Compute regret
      regret = mu_opt - env.list_of_reward_mus[action]
      cumulative_regret += regret

  return {'cumulative_regret': cumulative_regret, 'zeta_sequence': tuning_parameter_sequence, 'list_of_reward_mus': env.list_of_reward_mus}


def run(policy_name, save=True, monte_carlo_reps=30):
  """

  :return:
  """
  replicates = 48
  num_cpus = int(mp.cpu_count())
  pool = mp.Pool(processes=num_cpus)
  results = []
  episode_partial = partial(episode, policy_name, save=False,
                            monte_carlo_reps=monte_carlo_reps)
  num_batches = int(replicates / num_cpus)
  for batch in range(num_batches):
    results_for_batch = pool.map(episode_partial, range(batch*num_cpus, (batch+1)*num_cpus))
    results += results_for_batch

  rewards = [np.float(d['cumulative_regret']) for d in results]
  zeta_sequences = [d['zeta_sequence'] for d in results]
  list_of_reward_mus_sequences = [d['list_of_reward_mus'] for d in results]
  
  # Save results
  if save:
    results = {'T': float(100), 'mean_regret': float(np.mean(rewards)), 'std_regret': float(np.std(rewards)),
               'mus': [[0], [1]], 'vars':[[1], [600]], 'regret list': [float(r) for r in rewards],
               'list_of_reward_mus_sequences': list_of_reward_mus_sequences, 'zeta_sequences': zeta_sequences}

    base_name = 'normalmab-10-{}'.format(policy_name)
    prefix = os.path.join(project_dir, 'src', 'run', base_name)
    suffix = datetime.datetime.now().strftime("%y%m%d_%H%M%S")
    filename = '{}_{}.yml'.format(prefix, suffix)
    with open(filename, 'w') as outfile:
      yaml.dump(results, outfile)

  return


if __name__ == '__main__':
#  episode('eps-decay')
  run('eps-decay-fixed')
  run('eps')
  run('greedy')
  run('random')
#  run('eps-decay')



