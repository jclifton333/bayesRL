import sys
import pdb
import os
this_dir = os.path.dirname(os.path.abspath(__file__))
project_dir = os.path.join(this_dir, '..', '..')
sys.path.append(project_dir)


import matplotlib.pyplot as plt
from src.environments.Bandit import NormalCB, NormalUniformCB
from src.policies import tuned_bandit_policies as tuned_bandit
from src.policies import global_optimization as opt
from src.policies import rollout
import copy
import numpy as np
import src.policies.linear_algebra as la
from scipy.linalg import block_diag
from functools import partial
import datetime
import yaml
import multiprocessing as mp


def episode(label, tuning_function_parameter, n_patients=1,
            list_of_reward_betas=[[-10, 0.4, 0.4, -0.4], [-9.8, 0.6, 0.6, -0.4]], context_mean=np.array([0.0, 0.0, 0.0]),
            context_var=np.array([[1.0,0,0], [0,1.,0], [0, 0, 1.]]), list_of_reward_vars=[1, 1], T=30):

  np.random.seed(label)
  tuning_function = tuned_bandit.expit_epsilon_decay
  policy = tuned_bandit.linear_cb_epsilon_greedy_policy

  env = NormalCB(list_of_reward_betas=list_of_reward_betas, context_mean=context_mean, context_var=context_var,
                 list_of_reward_vars=list_of_reward_vars)
#  env = NormalUniformCB(list_of_reward_betas=[np.ones(10) + 0.05, np.ones(10)], list_of_reward_vars=[0.01, 25])
  cumulative_regret = 0.0
  # env.reset()
  tuning_parameter_sequence = []
  rewards = []
  actions = []

  # Using pre-simulated data
  # data_for_episode = env.generate_mc_samples(1, T)
  # rep_dict = data_for_episode[0]
  # initial_linear_model = rep_dict['initial_linear_model']
  # beta_hat_list = initial_linear_model['beta_hat_list']
  # Xprime_X_list = initial_linear_model['Xprime_X_list']
  # Xprime_X_inv_list = initial_linear_model['Xprime_X_inv_list']
  # X_list = initial_linear_model['X_list']
  # y_list = initial_linear_model['y_list']
  # X_dot_y_list = initial_linear_model['X_dot_y_list']
  # sampling_cov_list = initial_linear_model['sampling_cov_list']
  # sigma_hat_list = initial_linear_model['sigma_hat_list']

  # context_sequence = rep_dict['contexts']
  # regrets_sequence = rep_dict['regrets']
  # rewards_sequence = rep_dict['rewards']

  for t in range(T):
    X = env.X

    for patient in range(n_patients):
      x = copy.copy(env.curr_context)
      beta_hat = np.array([env.posterior_params_dict[a]['beta_post'] for a in range(env.number_of_actions)])
      # print(env.posterior_params_dict)
      action = policy(beta_hat, env.sampling_cov_list, x, tuning_function, tuning_function_parameter, T, t, env)
      res = env.step(action)
      cumulative_regret += -env.regret(action, x)
      actions.append(action)
      u = res['Utility']
      rewards.append(u)
    print(beta_hat)

    if t == 0:
      break
  return {'cumulative_regret': cumulative_regret, 'zeta_sequence': tuning_parameter_sequence,
          'rewards': rewards, 'actions': actions}


def run(number_of_initial_pulls):
  """

  :return:
  """

  parameters = yaml.load(open('../analysis/initial-pulls-params.yml'))
  parameters_for_num_initial_pulls = parameters[number_of_initial_pulls]
  replicates = 96

  regrets = []

  for param in parameters_for_num_initial_pulls:
    regrets_for_episode = []
    episode_partial = partial(episode, param)
    num_cpus = int(mp.cpu_count())
    pool = mp.Pool(processes=num_cpus)
    results = pool.map(episode_partial, range(replicates))
    regrets_for_episode = [d['cumulative_regret'] for d in results]
    regrets.append(regrets_for_episode)

  base_name = 'regrets-for-{}-initial-pulls'.format(number_of_initial_pulls)
  prefix = os.path.join(project_dir, 'src', 'run', base_name)
  suffix = datetime.datetime.now().strftime("%y%m%d_%H%M%S")
  filename = '{}_{}.yml'.format(prefix, suffix)
  with open(filename, 'w') as outfile:
    yaml.dump(results, outfile)

  return


if __name__ == '__main__':
  run(5)
  print('5 done')
  run(15)
  print('15 done')
  run(25)
  print('25 done')
  run(35)
  print('35 done')
