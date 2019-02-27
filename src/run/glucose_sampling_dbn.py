import sys
import pdb
import os
this_dir = os.path.dirname(os.path.abspath(__file__))
project_dir = os.path.join(this_dir, '..', '..')
sys.path.append(project_dir)

import datetime
import multiprocessing as mp
import numpy as np
import src.policies.rollout as rollout
import src.estimation.TransitionModel as transition
import src.policies.global_optimization as opt
from src.environments.Glucose import Glucose
import src.policies.tuned_bandit_policies as policies
import yaml
from functools import partial


def rollout_to_get_parameter(policy_name, num_to_collect, T, decay_function=None, save=False, monte_carlo_reps=10):
  if policy_name == 'kde':
    estimator = transition.KdeGlucoseModel()
  else:
    estimator = transition.LinearGlucoseModel(ar1=(policy_name == 'ar1'))

  np.random.seed(label)
  n_patients = 15

  tuning_function = policies.expit_epsilon_decay
  policy = policies.glucose_one_step_policy
  if T < 30:
    explore_ = {'zeta0': [1.0, 0.05, 1.0, 0.1, 300.3, 18.34, 387.0],
                'zeta1': [30.0, 0.0, 1.0, 0.0, 51.6, 52.58, 72.4],
                'zeta2': [0.1, 1.0, 0.01, 1.0, 0.22, 0.16, 0.14]}
  else:
    explore_ = {'zeta0': [1.0, 0.05, 1.0, 0.1, 3.97, 33.74],
                'zeta1': [30.0, 0.0, 1.0, 0.0, 84.88, 66.53],
                'zeta2': [0.1, 1.0, 0.01, 1.0, 0.09, 0.23]}
  bounds = {'zeta0': (0.025, 2.0), 'zeta1': (0.0, T), 'zeta2': (0.01, 2)}
  tuning_function_parameter = np.array([0.05, 1.0, 0.01])
  env = Glucose(nPatients=n_patients)
  cumulative_reward = 0.0
  env.reset()
  env.step(np.random.binomial(1, 0.3, n_patients))
  env.step(np.random.binomial(1, 0.3, n_patients))
  epsilon_list = []

  # Collect data using fixed eps-greedy
  for t in range(num_to_collect):
    eps = decay_function(t)
    action = policy(env, tuning_function, tuning_function_parameter, T, t, fixed_eps=eps)
    _, r = env.step(action)
    cumulative_reward += r
    epsilon_list.append(float(eps))

  # Get tuned parameter
  X, Sp1 = env.get_state_transitions_as_x_y_pair()
  y = Sp1[:, 0]
  estimator.fit(X, y)
  kwargs = {'n_rep': monte_carlo_reps, 'estimator': estimator}
  tuning_function_parameter = opt.bayesopt(rollout.glucose_npb_rollout, policy, tuning_function,
                                           tuning_function_parameter, T, env, None, kwargs, bounds, explore_)
  return tuning_function_parameter, estimator


def get_sampling_dbn_of_tuned_policy(tuning_function_parameter, T, monte_carlo_reps):
  pass

