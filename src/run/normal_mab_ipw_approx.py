import sys
import pdb
import os
this_dir = os.path.dirname(os.path.abspath(__file__))
project_dir = os.path.join(this_dir, '..', '..')
sys.path.append(project_dir)


import src.policies.ipw_regret_estimate as ipw
import src.hypothesis_test.mab_hypothesis_test as ht
from src.environments.Bandit import NormalMAB
import numpy as np
from functools import partial
import datetime
import yaml
import multiprocessing as mp
from scipy.stats import norm


def episode(label, tune=True, std=1., list_of_reward_mus=[0.0,0.1], T=50, out_of_sample_size=10000):
  np.random.seed(label)
  env = NormalMAB(list_of_reward_mus=list_of_reward_mus, list_of_reward_vars=[std**2]*len(list_of_reward_mus))
  lower_bound = 0.1
  upper_bound = 0.55

  cumulative_regret = 0.0
  mu_opt = np.max(env.list_of_reward_mus)
  env.reset()

  # Initialize lists that will be needed to compute IPWEs
  actions = []
  action_probs = []
  rewards = []

  # Initial pulls
  for a in range(env.number_of_actions):
    res_ = env.step(a)
    rewards.append(res_['Utility'])
    actions.append(a)
    action_probs.append(0.5)

  # Initial propensities
  pi_tilde = 0.5
  pi_sum = pi_tilde
  pi_inv_sum = (1 / pi_tilde + 1 / pi_tilde)  # Propensity sums, used for IPW estimates of regret
  m_pi_inv_sum = (1 / (1 - pi_tilde) + 1 / (1 - pi_tilde))
  in_sample_size = T

  epsilon_sequence = []
  # Get initial epsilon
  best_epsilon = ipw.minimax_epsilon(in_sample_size, out_of_sample_size, lower_bound, upper_bound,
                                       (pi_sum, pi_inv_sum, m_pi_inv_sum), 0)
  epsilon_sequence.append(float(best_epsilon))
  for t in range(T):
    # Get action probabilities (also used for propensities)
    arm0_prob = (env.estimated_means[0] > env.estimated_means[1])*((1 - best_epsilon) + best_epsilon/2) \
                + (env.estimated_means[0] <= env.estimated_means[1])*(best_epsilon/2)

    if tune:
      # Get best epsilon using ipw estimator
      mu_ipw, std_ipw = ht.ipw(env.number_of_actions, np.array(actions), np.array(action_probs), np.array(rewards))
      se_ipw = std_ipw / np.sqrt(t + 1)  # Pooled estimate

      # Get confidence intervals to form range for minimax
      v0 = pi_inv_sum / (t+1)**2
      v1 = m_pi_inv_sum / (t+1)**2
      se_ipw = np.sqrt((v0 + v1) / 2)
      diff = mu_ipw[1] - mu_ipw[0]
      diff_lower_conf = diff - 1.96*se_ipw
      diff_upper_conf = diff + 1.96*se_ipw

      best_epsilon = ipw.minimax_epsilon(in_sample_size, out_of_sample_size, diff_lower_conf, diff_upper_conf,
                                         (pi_sum, pi_inv_sum, m_pi_inv_sum), t)
      epsilon_sequence.append(float(best_epsilon))

      propensity = 1 - norm.cdf(diff / se_ipw)
      pi_sum += propensity
      pi_inv_sum += 1 / propensity
      m_pi_inv_sum += 1 / (1 - propensity)

    # Get eps-greedy action
    if np.random.uniform() < arm0_prob:
      action = 0
    else:
      action = 1

    # Take action and update regret
    env.step(action)
    regret = mu_opt - env.list_of_reward_mus[action]
    cumulative_regret += regret

  cumulative_regret += out_of_sample_size*(env.estimated_means[0] > env.estimated_means[1])
  return {'cumulative_regret': cumulative_regret, 'epsilon_sequence': epsilon_sequence}


def run(replicates, tune=True, save=True, out_of_sample_size=1000):
  # Partial function to distribute
  episode_partial = partial(episode, tune=tune, out_of_sample_size=out_of_sample_size)

  # Run episodes in parallel
  num_batches = int(np.floor(replicates / 48))
  pool = mp.Pool(processes=48)
  res = []
  for batch in range(num_batches):
    res_batch = pool.map(episode_partial, range(replicates*batch, replicates*(batch+1)))
    res += res_batch

  # Collect results
  regrets_for_episode = [d['cumulative_regret'] for d in res]
  se_regret = float(np.std(regrets_for_episode) / replicates)
  mean_regret = float(np.mean([d['cumulative_regret'] for d in res]))
  epsilon_sequences = [d['epsilon_sequence'] for d in res]
  if save:
    results = {'mean_regret': mean_regret, 'epsilon_sequence': epsilon_sequences, 'se_regret': se_regret}

    # Make filename and save to yaml
    base_name = 'ipw-estimate-N={}-tune={}'.format(out_of_sample_size,tune)
    prefix = os.path.join(project_dir, 'src', 'run', base_name)
    suffix = datetime.datetime.now().strftime("%y%m%d_%H%M%S")
    filename = '{}_{}.yml'.format(prefix, suffix)
    with open(filename, 'w') as outfile:
      yaml.dump(results, outfile)


if __name__ == "__main__":
  np.random.seed(3)
  # episode(0, tune=True, std=0.1, list_of_reward_mus=[0.0, 0.1], T=50, out_of_sample_size=1000)
  run(48*4, tune=False, out_of_sample_size=100)
  run(48*4, tune=True, out_of_sample_size=100)
  run(48*4, tune=False, out_of_sample_size=1000)
  run(48*4, tune=True, out_of_sample_size=1000)
  run(48*4, tune=False, out_of_sample_size=10000)
  run(48*4, tune=True, out_of_sample_size=10000)
