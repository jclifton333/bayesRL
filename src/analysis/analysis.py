import sys
"""
Functions for sensitive analysis the optimal value of \zeta is to the parameters of the model
"""
import os
this_dir = os.path.dirname(os.path.abspath(__file__))
project_dir = os.path.join(this_dir, '..', '..')
sys.path.append(project_dir)

import pdb
import numpy as np
from src.environments.Bandit import NormalMAB, BernoulliMAB
import src.policies.tuned_bandit_policies as tuned_bandit
import yaml
import multiprocessing as mp
from functools import partial


def episode(p0p1, env_name, zeta_grid_dimension, mc_rep=500, T=10):
  if env_name == "BernoulliMAB":
    env = BernoulliMAB(list_of_reward_mus=[p0p1[0], p0p1[1]])
  elif env_name == "NormalMAB":
    env = NormalMAB(list_of_reward_mus=[p0p1[0], p0p1[1]])

  best_zeta2 = None
  best_score = -float('inf')
  print('p0p1 {}'.format(p0p1))
  for zeta2 in np.linspace(0, 2, zeta_grid_dimension):
    zeta = np.array([0.3, -5, zeta2])
    mean_cum_reward = 0.0
    for rep in range(mc_rep):
      env.reset()
      for t in range(T):
        epsilon = tuned_bandit.expit_epsilon_decay(T, t, zeta)
        if np.random.rand() < epsilon:
          action = np.random.choice(2)
        else:
          action = np.argmax([env.estimated_means])
        env.step(action)
      cum_reward = sum(env.U)
      mean_cum_reward += (cum_reward - mean_cum_reward)/(rep+1)
    if mean_cum_reward > best_score:
      best_score = mean_cum_reward
      best_zeta2 = zeta2
  return {'p0': float(env.list_of_reward_mus[0]), 'p1': float(env.list_of_reward_mus[1]),
          'best_zeta2': float(best_zeta2), 'best_score': float(best_score)}


def run(env_name, p_grid_dimension=10, zeta_grid_dimension=10, mc_rep=500, T=10):

  if env_name == "BernoulliMAB":
    p0_low = 0
    p0_up = 1
  elif env_name == "NormalMAB":
    p0_low = -2
    p0_up = 2

  param_grid = np.linspace(p0_low, p0_up, p_grid_dimension)
  params = [(param_grid[i], param_grid[j]) for i in range(len(param_grid)) for j in range(i, len(param_grid))]

  episode_partial = partial(episode, env_name=env_name, zeta_grid_dimension=zeta_grid_dimension, mc_rep=mc_rep, T=T)

  pool = mp.Pool(mp.cpu_count())
  results = pool.map(episode_partial, params)

  # Collect results
  combined_results = {'p0': [], 'p1': [], 'best_zeta2': [], 'best_score': []}
  for d in results:
    for k, v in d.items():
      combined_results[k].append(v)

  with open('{}-opt-zeta-vs-model.yml'.format(env_name), 'w') as handle:
    yaml.dump(combined_results, handle)


if __name__ == '__main__':
  run("NormalMAB", p_grid_dimension=10, zeta_grid_dimension=10, mc_rep=500, T=10)
  run("BernoulliMAB", p_grid_dimension=10, zeta_grid_dimension=10, mc_rep=500, T=10)
