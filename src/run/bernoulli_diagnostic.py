import sys
import pdb
import os
this_dir = os.path.dirname(os.path.abspath(__file__))
project_dir = os.path.join(this_dir, '..', '..')
sys.path.append(project_dir)


from src.policies import tuned_bandit_policies as tuned_bandit
from src.policies import gittins_index_policies as gittins
from src.policies import rollout
from src.environments.Bandit import BernoulliMAB
import src.policies.global_optimization as opt
import numpy as np
from functools import partial
import datetime
import yaml
import multiprocessing as mp
import matplotlib.pyplot as plt
from matplotlib import pyplot


def episode(tune, list_of_reward_mus=[0.3, 0.6], T=50, monte_carlo_reps=1000):
  N_REPLICATES = 1000

  policy = tuned_bandit.mab_epsilon_greedy_policy
  if tune:
    # Tune under true model
    tuning_function = tuned_bandit.expit_epsilon_decay
    tuning_function_parameter = np.array([0.05, -45, 2.5])
    bounds = {'zeta0': (0.05, 2.0), 'zeta1': (1.0, 49.0), 'zeta2': (0.01, 2.5)}
    explore_ = {'zeta0': [1.0, 1.0, 1.0, 1.90980867, 5.848, 0.4466, 10.177],
                'zeta1': [25.0, 49.0, 1.0, 49.94980088, 88.9, 50, 87.55],
                 'zeta2': [0.1, 2.5, 2.0, 1.88292034, 0.08, 0.1037, 0.094]}
    sim_env = BernoulliMAB(list_of_reward_mus=list_of_reward_mus)
    pre_simulated_data = sim_env.generate_mc_samples_bernoulli(monte_carlo_reps, T)
    tuning_function_parameter, tuning_val = opt.bayesopt(rollout.bernoulli_mab_rollout_with_fixed_simulations, policy,
                                                         tuning_function,
                                                         tuning_function_parameter, T, sim_env, monte_carlo_reps,
                                                         {'pre_simulated_data': pre_simulated_data},
                                                         bounds, explore_)
    print('estimate of optimal value: {}'.format(tuning_val))
    # tuning_function_parameter = np.array([0.17, 26.1, 1.99])

  else:
    tuning_function = lambda x, y, z: 0.05
    tuning_function_parameter = None

  # Rollouts to evaluate exploration policy
  env = BernoulliMAB(list_of_reward_mus=list_of_reward_mus)
  mu_opt = np.max(env.list_of_reward_mus)
  cumulative_rewards = []
  for rep in range(N_REPLICATES):
    cumulative_reward = 0.0
    env.reset()

    # Initial pulls: each arm is pulled once
    # for a in range(env.number_of_actions):
    #   env.step(a)

    for t in range(T):
      action = policy(env.estimated_means, env.standard_errors, env.number_of_pulls, tuning_function,
                      tuning_function_parameter, T, t, env)
      cumulative_reward += env.list_of_reward_mus[action]

    cumulative_rewards.append(cumulative_reward)

  mean_regret = np.mean(cumulative_rewards)
  std_regret = np.std(cumulative_rewards) / np.sqrt(N_REPLICATES)
  return mean_regret, std_regret


if __name__ == '__main__':
  tuned_performance = episode(True)
  not_tuned_performance = episode(False)
  print('Tuned reward: {}\nNot tuned reward: {}'.format(tuned_performance, not_tuned_performance))
