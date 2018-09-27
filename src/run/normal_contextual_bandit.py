import sys
import pdb
import os
this_dir = os.path.dirname(os.path.abspath(__file__))
project_dir = os.path.join(this_dir, '..', '..')
sys.path.append(project_dir)


import matplotlib.pyplot as plt
from src.environments.Bandit import NormalCB
from src.policies import tuned_bandit_policies as tuned_bandit
from src.policies import global_optimization as opt
from src.policies import rollout
import copy
import numpy as np
from scipy.linalg import block_diag
from functools import partial
import datetime
import yaml
import multiprocessing as mp


def episode(policy_name, label, beta_hat_list=[[1.0, 1.0], [2.0, -2.0]], context_mean=np.array([0.0, 0.0]),
            context_variance=np.array([[1.0, -0.2], [-0.2, 1.]]), pre_simulate=True):
  np.random.seed(label)
  T = 100
  mc_replicates = 100

  # ToDo: Create policy class that encapsulates this behavior
  if policy_name == 'eps':
    tuning_function = lambda a, b, c: 0.05  # Constant epsilon
    policy = tuned_bandit.linear_cb_epsilon_greedy_policy
    tune = False
    tuning_function_parameter = None
  elif policy_name == 'eps-decay-fixed':
    tuning_function = lambda a, t, c: 0.5 / (t + 1)
    policy = tuned_bandit.linear_cb_epsilon_greedy_policy
    tune = False
    tuning_function_parameter = None
  elif policy_name == 'eps-decay':
    tuning_function = tuned_bandit.stepwise_linear_epsilon
    policy = tuned_bandit.linear_cb_epsilon_greedy_policy
    tune = True
    tuning_function_parameter = np.ones(10) * 0.025
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
  elif policy_name == 'ts':
    tuning_function = lambda a, b, c: 1.0  # No shrinkage
    policy = tuned_bandit.linear_cb_thompson_sampling_policy
    tune = False
    tuning_function_parameter = None
  # elif policy_name == 'ts-shrink':
  #   tuning_function = tuned_bandit.expit_truncate
  #   policy = tuned_bandit.thompson_sampling_policy
  #   tune = True
  #   tuning_function_parameter = np.array([-2, 1])
  else:
    raise ValueError('Incorrect policy name')

  env = NormalCB(list_of_reward_betas=beta_hat_list, context_mean=context_mean, context_var=context_variance,
                 list_of_reward_vars=[1, 1000])
  cumulative_regret = 0.0
  env.reset()

  for t in range(T):
    X = env.X
    estimated_context_mean = np.mean(X, axis=0)
    estimated_context_variance = np.cov(X, rowvar=False)

    if tune:
      if pre_simulate:
        sim_env = NormalCB(list_of_reward_betas=env.beta_hat_list, list_of_reward_vars=env.sigma_hat_list,
                           context_mean=estimated_context_mean, context_var=estimated_context_variance)
        pre_simulated_data = sim_env.generate_mc_samples(mc_replicates, T)
        tuning_function_parameter = opt.bayesopt(rollout.normal_cb_rollout_with_fixed_simulations, policy,
                                                 tuning_function, tuning_function_parameter, T, estimated_context_mean,
                                                 estimated_context_variance, sim_env, mc_replicates,
                                                 {'pre_simulated_data': pre_simulated_data})
      else:
        tuning_function_parameter = tuned_bandit.random_search(tuned_bandit.oracle_rollout, policy, tuning_function,
                                                               tuning_function_parameter,
                                                               linear_model_results, T, t, estimated_context_mean,
                                                               estimated_context_variance, env)

    x = copy.copy(env.curr_context)
    print('time {} epsilon {}'.format(t, tuning_function(T,t,tuning_function_parameter)))
    beta_hat = np.array(env.beta_hat_list)
    action = policy(beta_hat, env.sampling_cov_list, x, tuning_function, tuning_function_parameter, T, t, env)
    env.step(action)
    cumulative_regret += env.regret(action, x)

  return cumulative_regret


def run(policy_name, save=True):
  """

  :return:
  """
  np.random.seed(3)
  beta_hat_list = [np.random.normal(scale=0.5, size=10) for act in range(2)]
  context_mean = np.zeros(10)
  context_variance = np.diag(np.random.random(size=10))


  replicates = 96
  num_cpus = int(mp.cpu_count())
  results = []
  pool = mp.Pool(processes=num_cpus)

  episode_partial = partial(episode, policy_name, beta_hat_list=beta_hat_list, context_mean=context_mean, 
                            context_variance=context_variance)

  num_batches = int(replicates / num_cpus)
  for batch in range(num_batches):
    results_for_batch = pool.map(episode_partial, range(batch*num_cpus, (batch+1)*num_cpus))
    results += results_for_batch

  # Save results
  if save:
    results = {'mean_regret': float(np.mean(results)), 'std_regret': float(np.std(results)),
               'beta_hat_list': [beta.tolist() for beta in beta_hat_list], 
               'context_mean': [float(c) for c in context_mean], 'regret list': [float(r) for r in results],
               'context_variance': [[float(context_variance[i, j]) for j in range(context_variance.shape[1])]
                                    for i in range(context_variance.shape[0])]}

    base_name = 'normalcb-{}'.format(policy_name)
    prefix = os.path.join(project_dir, 'src', 'run', base_name)
    suffix = datetime.datetime.now().strftime("%y%m%d_%H%M%S")
    filename = '{}_{}.yml'.format(prefix, suffix)
    with open(filename, 'w') as outfile:
      yaml.dump(results, outfile)

  return


if __name__ == '__main__':
  # episode('eps-decay', np.random.randint(low=1, high=1000))
  run('eps')
  run('greedy')
  run('eps-decay-fixed')

