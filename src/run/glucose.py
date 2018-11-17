import sys
import pdb
import os
this_dir = os.path.dirname(os.path.abspath(__file__))
project_dir = os.path.join(this_dir, '..', '..')
sys.path.append(project_dir)

import datetime
import time
import multiprocessing as mp
import numpy as np
import src.policies.rollout as rollout
import src.estimation.dependent_density as dd
from src.estimation.TransitionModel import GlucoseTransitionModel
import src.policies.global_optimization as opt
from src.environments.Glucose import Glucose
import src.policies.tuned_bandit_policies as policies
import yaml
import pandas as pd
import pymc3 as pm
import matplotlib.pyplot as plt
from theano import shared, tensor as tt
from functools import partial


def npb_diagnostics():
  np.random.seed(3)
  n_patients = 1
  T = 5
  env = Glucose(nPatients=n_patients)
  cumulative_reward = 0.0
  env.reset()

  # Collect data with random policy
  for t in range(T):
    # Get posterior
    # X, Sp1 = env.get_state_transitions_as_x_y_pair()
    # X = shared(X)
    # y = Sp1[:, 0]
    # model_, trace_ = dd.dependent_density_regression(X, y)
    action = np.random.binomial(1, 0.3, n_patients)
    env.step(action)

  # Get posterior
  X, Sp1 = env.get_state_transitions_as_x_y_pair()
  X = shared(X)
  y = Sp1[:, 0]
  model_, trace_, compare_ = dd.dependent_density_regression(X, y, stack=True)
  print(compare_)

  # Test states
  hypoglycemic_0 = np.array([[1.0, 50, 0, 33, 50, 0, 0, 0, 0]])
  hypoglycemic_1 = np.array([[1.0, 50, 0, 33, 50, 0, 0, 1, 0]])
  hyperglycemic_0 = np.array([[1.0, 200, 0, 30, 200, 0, 0, 0, 0]])
  hyperglycemic_1 = np.array([[1.0, 200, 0, 30, 200, 0, 0, 1, 0]])

  # Histograms
  PPD_SAMPLES = 500
  # X.set_value(hyperglycemic_0)
  # pp_sample_0 = pm.sample_ppc(trace_, model=model_, samples=PPD_SAMPLES)['obs']
  # X.set_value(hyperglycemic_1)
  # pp_sample_1 = pm.sample_ppc(trace_, model=model_, samples=PPD_SAMPLES)['obs']
  # plt.hist(pp_sample_0)
  # plt.hist(pp_sample_1)
  # plt.show()

  x_vals = np.array([[1.0, g, 0, 33, g, 0, 0, 0, 0] for g in np.linspace(50, 200, 20)])
  X.set_value(x_vals)
  pp_sample = \
    pm.sample_ppc_w(trace_, PPD_SAMPLES, model_, weights=compare_.weight.sort_index(ascending=True))
  plt.plot(np.linspace(50, 200, 20), np.mean(pp_sample['obs'], axis=0))
  plt.show()
  return


def episode(label, policy_name, save=False, monte_carlo_reps=10):
  if policy_name in ['np', 'p', 'averaged']:
    tune = True
    explore_ = {'zeta0': [1.0, 0.05, 1.0, 0.1], 'zeta1': [30.0, 0.0, 1.0, 0.0], 'zeta2': [0.1, 1.0, 0.01, 1.0]}
    bounds = {'zeta0': (0.025, 2.0), 'zeta1': (0.0, 30.0), 'zeta2': (0.01, 2)}
    tuning_function_parameter = np.array([0.05, 1.0, 0.01])
    fixed_eps = None
  else:
    tune = False
    fixed_eps = 0.05

  if policy_name == 'averaged':
    stack = True
  else:
    stack = False

  # if save:
  #   base_name = 'glucose-stacked={}-{}'.format(stacked, label)
  #   prefix = os.path.join(project_dir, 'src', 'run', 'results', base_name)
  #   suffix = datetime.datetime.now().strftime("%y%m%d_%H%M%S")
  #   filename = '{}_{}.yml'.format(prefix, suffix)

  np.random.seed(label)
  n_patients = 10
  T = 20

  tuning_function = policies.expit_epsilon_decay
  policy = policies.glucose_one_step_policy
  # ToDo: check these
  explore_ = {'zeta0': [1.0, 0.05, 1.0, 0.1], 'zeta1': [30.0, 0.0, 1.0, 0.0], 'zeta2': [0.1, 1.0, 0.01, 1.0]}
  bounds = {'zeta0': (0.025, 2.0), 'zeta1': (0.0, 30.0), 'zeta2': (0.01, 2)}
  tuning_function_parameter = np.array([0.05, 1.0, 0.01])
  env = Glucose(nPatients=n_patients)
  estimator = GlucoseTransitionModel(method=policy_name)
  cumulative_reward = 0.0
  env.reset()
  env.step(np.random.binomial(1, 0.3, n_patients))

  for t in range(T):
    if tune:
      # Get posterior
      X, Sp1 = env.get_state_transitions_as_x_y_pair()
      # X_np = shared(X)
      # X_p = shared(X[:, :3])
      y = Sp1[:, 0]
      # model_, trace_, compare_ = dd.dependent_density_regression(X_np, y, X_p=X_p)
      # kwargs = {'n_rep': monte_carlo_reps, 'X_np': X_np, 'model': model_, 'trace': trace_, 'compare': compare_,
      #           'X_p': X_p}
      estimator.fit(X, y)
      kwargs = {'n_rep': monte_carlo_reps, 'estimator': estimator}

      tuning_function_parameter = opt.bayesopt(rollout.glucose_npb_rollout, policy, tuning_function,
                                               tuning_function_parameter, T, env, None, kwargs, bounds, explore_)

    X = [x[:-1, :] for x in env.X]
    action = policy(env, X, env.R, tuning_function, tuning_function_parameter, T, t, fixed_eps=fixed_eps)
    _, r = env.step(action)
    cumulative_reward += r

    # Save results
    if save:
      results = {'t': float(t), 'regret': float(cumulative_reward)}
      with open(filename, 'w') as outfile:
        yaml.dump(results, outfile)

  return {'cumulative_reward': float(cumulative_reward)}


def run(policy_name):
  replicates = 24
  num_cpus = replicates
  pool = mp.Pool(processes=num_cpus)

  episode_partial = partial(episode, policy_name=policy_name)
  results = pool.map(episode_partial, range(replicates))

  base_name = 'glucose-{}'.format(policy_name)
  prefix = os.path.join(project_dir, 'src', 'run', base_name)
  suffix = datetime.datetime.now().strftime("%y%m%d_%H%M%S")
  filename = '{}_{}.yml'.format(prefix, suffix)
  results = [d['cumulative_reward'] for d in results]
  results_to_save = {'mean': float(np.mean(results)),
                     'se': float(np.std(results) / np.sqrt(len(results)))}
  with open(filename, 'w') as outfile:
    yaml.dump(results_to_save, outfile)

  return


if __name__ == '__main__':
  # t0 = time.time()
  # reward = episode(0, 'averaged')
  # t1 = time.time()
  # print('time: {} reward: {}'.format(t1 - t0, reward))
  # npb_diagnostics()
  # episode(0, 'averaged')
  run('np')
  # run('p')
  run('averaged')
