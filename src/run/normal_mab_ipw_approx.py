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


def episode(label, tune=True, std=0.1, list_of_reward_mus=[0.0,0.1], T=50, out_of_sample_size=100):
  np.random.seed(label)
  env = NormalMAB(list_of_reward_mus=list_of_reward_mus, list_of_reward_vars=[std**2]*len(list_of_reward_mus))
  lower_bound = 0.1
  upper_bound = 0.55

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
  in_sample_size = T

  epsilon_sequence = []
  # Get initial epsilon
  best_epsilon = ipw.minimax_epsilon(in_sample_size, out_of_sample_size, lower_bound, upper_bound,
                                       (pi_inv_sum, m_pi_inv_sum), t)
  epsilon_sequence.append(best_epsilon)
  for t in range(T):
    # Get action probabilities (also used for propensities)
    arm0_prob = (env.estimated_means[0] > env.estimated_means[1])*((1 - best_epsilon) + best_epsilon/2) \
                + (env.estimated_means[0] <= env.estimated_means[1])*(best_epsilon/2)

    if tune:
      # Get best epsilon using ipw estimator
      # Get confidence intervals to form range for minimax
      mu_1_upper_conf = env.estimated_means[0] + 1.96*env.standard_errors[0]
      mu_1_lower_conf = env.estimated_means[0] - 1.96*env.standard_errors[0]
      mu_2_upper_conf = env.estimated_means[1] + 1.96*env.standard_errors[1]
      mu_2_lower_conf = env.estimated_means[1] - 1.96*env.standard_errors[1]

      min_range_ = np.max((lower_bound, mu_2_lower_conf - mu_1_upper_conf))
      max_range_ = np.min((upper_bound, mu_2_upper_conf - mu_1_lower_conf))
      best_epsilon = ipw.minimax_epsilon(in_sample_size, out_of_sample_size, min_range_, max_range_,
                                         (pi_inv_sum, m_pi_inv_sum), t)
      epsilon_sequence.append(best_epsilon)

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

  return {'cumulative_regret': cumulative_regret, 'epsilon_sequence': epsilon_sequence}


def run(replicates=48, tune=True):
  # Partial function to distribute
  episode_partial = partial(episode, tune=tune)

  # Run episodes in parallel
  pool = mp.Pool(processes=replicates)
  res = pool.map(episode_partial, range(replicates))

  # Get regrets
  mean_regret = float(np.mean([d['cumulative_regret'] for d in res]))


if __name__ == "__main__":
  print(episode(0))
