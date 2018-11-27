import sys
import pdb
import numpy as np
import os
this_dir = os.path.dirname(os.path.abspath(__file__))
project_dir = os.path.join(this_dir, '..', '..')
sys.path.append(project_dir)

from src.environments.Glucose import Glucose
from src.estimation.TransitionModel import GlucoseTransitionModel
import src.policies.simulation_optimization_policies as opt
import matplotlib.pyplot as plt


def evaluate_policy(transition_model, time_horizon, policy, feature_function, initial_state_and_x=None):
  """
  :param initial_state:
  :param initial_x: initial features
  :param transition_model:
  :param time_horizon:
  :param policy:
  :param feature_function:
  :return:
  """
  MC_REPLICATES = 100
  returns = []
  env_ = Glucose(nPatients=100)
  env_.reset()
  for t in range(time_horizon):
    actions = []
    for patient in range(MC_REPLICATES):
      s = env_.S[patient][-1, :]
      x = env_.X[patient][-1, :]
      a = policy(s, x)
      actions.append()
    env_.step(actions)
  return np.mean(env_.R)


  if initial_state_and_x is not None:
    s = initial_state_and_x['initial_state']
    x = initial_state_and_x['initial_x']
  else:
    env_ = Glucose()
  for _ in range(MC_REPLICATES):
    if initial_state_and_x is None:
      env_.reset()
      s = env_.S[-1][-1, :]
      x = env_.X[-1][-1, :]
    return_ = 0.0
    for t in range(time_horizon):
      a = policy(s, x)
      x = feature_function(s, a, x)
      s, r = transition_model(np.array([x]))
      return_ += r
    returns.append(return_)
  return np.mean(returns)


def evaluate_glucose_mb_policy(method):
  # Roll out to get data
  n_patients = 20
  T = 20
  env = Glucose(n_patients)
  env.reset()
  env.step(np.random.binomial(1, 0.3, n_patients))

  for t in range(T):
    env.step(np.random.binomial(1, 0.3, n_patients))

  if method in ['np', 'p', 'averaged']:
    # Fit model on data
    estimator = GlucoseTransitionModel(method='np')
    X, Sp1 = env.get_state_transitions_as_x_y_pair()
    S = env.S
    y = Sp1[:, 0]
    estimator.fit(X, y)

    # Get optimal policy under model
    def rollout_policy(s_, x_):
      return np.random.binomial(1, 0.3)

    initial_x = X[-1, :]
    initial_state = S[0][-1, :]
    transition_model = estimator.draw_from_ppd
    feature_function = opt.glucose_feature_function
    pi = opt.solve_for_pi_opt(initial_state, initial_x, transition_model, T, 2, rollout_policy, feature_function)

  elif method == 'random':
    def pi(s_, x_):
      return np.random.binomial(1, 0.3)

  # Evaluate policy
  # v = None
  v = evaluate_policy(transition_model, T, pi, feature_function)

  return v


if __name__ == "__main__":
  v_ = evaluate_glucose_mb_policy()
