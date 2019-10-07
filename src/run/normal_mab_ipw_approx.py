import sys
import pdb
import os
this_dir = os.path.dirname(os.path.abspath(__file__))
project_dir = os.path.join(this_dir, '..', '..')
sys.path.append(project_dir)


from src.policies import tuned_bandit_policies as tuned_bandit
from src.policies import gittins_index_policies as gittins
import src.policies.ipw_regret_estimate as ipw
from src.policies import rollout
from src.environments.Bandit import NormalMAB
import src.policies.global_optimization as opt
import numpy as np
from functools import partial
import datetime
import yaml
import multiprocessing as mp


def episode(label, std=0.1, list_of_reward_mus=[0.0,0.25], T=50, monte_carlo_reps=1000, in_sample_size=100,
            out_of_sample_size=10000, posterior_sample=False):
  np.random.seed(label)
  env = NormalMAB(list_of_reward_mus=list_of_reward_mus, list_of_reward_vars=[std**2]*len(list_of_reward_mus))

  cumulative_regret = 0.0
  mu_opt = np.max(env.list_of_reward_mus)
  env.reset()
  tuning_parameter_sequence = []
  # Initial pulls
  for a in range(env.number_of_actions):
    env.step(a)

  estimated_means_list = []
  estimated_vars_list = []
  actions_list = []
  rewards_list = []
  for t in range(T):
    estimated_means_list.append([float(xbar) for xbar in env.estimated_means])
    estimated_vars_list.append([float(s) for s in env.estimated_vars])

    # Get best epsilon using ipw estimator
    # Get confidence intervals to form range for minimax

    mu_1_upper_conf = env.estimated_means[0] + 1.96*env.standard_errors[0]
    mu_1_lower_conf = env.estimated_means[0] - 1.96*env.standard_errors[0]
    mu_2_upper_conf = env.estimated_means[1] + 1.96*env.standard_errors[1]
    mu_2_lower_conf = env.estimated_means[1] - 1.96*env.standard_errors[1]

    min_range_ = np.max((0.25, mu_2_lower_conf - mu_1_upper_conf))
    max_range_ = np.min((0.75, mu_2_upper_conf - mu_1_lower_conf))
    best_epsilon = ipw.minimax_epsilon(in_sample_size, out_of_sample_size, min_range_, max_range_, propensity_sums)
    action = policy(estimated_means, env.standard_errors, env.number_of_pulls, tuning_function,
                    tuning_function_parameter, T, t, env)
    res = env.step(action)
    u = res['Utility']
    actions_list.append(int(action))
    rewards_list.append(float(u))

    # Compute regret
    regret = mu_opt - env.list_of_reward_mus[action]
    cumulative_regret += regret

  return {'cumulative_regret': cumulative_regret, 'zeta_sequence': tuning_parameter_sequence,
          'estimated_means': estimated_means_list, 'estimated_vars': estimated_vars_list,
          'rewards_list': rewards_list, 'actions_list': actions_list}