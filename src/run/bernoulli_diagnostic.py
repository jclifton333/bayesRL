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


def episode(list_of_reward_mus=[0.3, 0.6], T=50, monte_carlo_reps=1000):
  np.random.seed(label)

  # Tune under true model
  tuning_function = tuned_bandit.expit_epsilon_decay
  policy = tuned_bandit.mab_epsilon_greedy_policy
  tuning_function_parameter = np.array([0.05, 45, 2.5])
  bounds = {'zeta0': (0.05, 2.0), 'zeta1': (1.0, 49.0), 'zeta2': (0.01, 2.5)}
  explore_ = {'zeta0': [1.0, 1.0, 1.0, 1.90980867, 5.848, 0.4466, 10.177],
              'zeta1': [25.0, 49.0, 1.0, 49.94980088, 88.9, 50, 87.55],
              'zeta2': [0.1, 2.5, 2.0, 1.88292034, 0.08, 0.1037, 0.094]}
  sim_env = BernoulliMAB(list_of_reward_mus=list_of_reward_mus)
  pre_simulated_data = sim_env.generate_mc_samples_bernoulli(monte_carlo_reps, T)
  tuning_function_parameter, tuning_val = opt.bayesopt(rollout.bernoulli_mab_rollout_with_fixed_simulations, policy,
                                                       tuning_function,
                                                       tuning_function_parameter, T, env, monte_carlo_reps,
                                                       {'pre_simulated_data': pre_simulated_data},
                                                       bounds, explore_)

  # Rollout using parameter obtained by tuning under true model
  env = BernoulliMAB(list_of_reward_mus=list_of_reward_mus)
  mu_opt = np.max(env.list_of_reward_mus)
  cumulative_regrets = []
  for rep in range(100):
    cumulative_regret = 0.0
    env.reset()

    # Initial pulls: each arm is pulled once
    for a in range(env.number_of_actions):
      env.step(a)
    estimated_means_list = []
    estimated_vars_list = []
    actions_list = []
    rewards_list = []

    for t in range(T):
      action = policy(env.estimated_means, env.standard_errors, env.number_of_pulls, tuning_function,
                      tuning_function_parameter, T, t, env)
      cumulative_regret += mu_opt - env.list_of_reward_mus[action]

    cumulative_regrets.append(cumulative_regret)

  mean_regret = np.mean(cumulative_regrets)
  std_regret = np.std(cumulative_regrets)
  return mean_regret, std_regret


if __name__ == '__main__':
  episode()
