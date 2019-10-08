import sys
import pdb
import os
this_dir = os.path.dirname(os.path.abspath(__file__))
project_dir = os.path.join(this_dir, '..', '..')
sys.path.append(project_dir)


import src.policies.ipw_regret_estimate as ipw
from src.environments.Bandit import NormalMAB
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
  # Initial pulls
  for a in range(env.number_of_actions):
    env.step(a)

  # Initial propensities
  pi_tilde = 0.5
  pi_inv_sum = (1 / pi_tilde + 1 / pi_tilde)  # Propensity sums, used for IPW estimates of regret
  m_pi_inv_sum = (1 / (1 - pi_tilde) + 1 / (1 - pi_tilde))

  for t in range(T):
    # Get best epsilon using ipw estimator
    # Get confidence intervals to form range for minimax
    mu_1_upper_conf = env.estimated_means[0] + 1.96*env.standard_errors[0]
    mu_1_lower_conf = env.estimated_means[0] - 1.96*env.standard_errors[0]
    mu_2_upper_conf = env.estimated_means[1] + 1.96*env.standard_errors[1]
    mu_2_lower_conf = env.estimated_means[1] - 1.96*env.standard_errors[1]

    min_range_ = np.max((0.25, mu_2_lower_conf - mu_1_upper_conf))
    max_range_ = np.min((0.75, mu_2_upper_conf - mu_1_lower_conf))
    best_epsilon = ipw.minimax_epsilon(in_sample_size, out_of_sample_size, min_range_, max_range_, propensity_sums)

    # Get action probabilities (also used for propensities)
    arm0_prob = (env.estimated_means[0] > env.estimated_means[1])*((1 - best_epsilon) + best_epsilon/2) \
                + (env.estimated_means[0] <= env.estimated_means[1])*(best_epsilon/2)

    pi_inv_sum += 1 / arm0_prob  # ToDo: Check that this is the correct arm
    m_pi_inv_sum += 1 / (1 - arm0_prob)

    # Get eps-greedy action
    if np.random.uniform() < arm0_prob:
      action = 0
    else:
      action = 1

    # Take action and update regret
    env.step(action)
    regret = mu_opt - env.list_of_reward_mus[action]
    cumulative_regret += regret

  return {'cumulative_regret': cumulative_regret}
