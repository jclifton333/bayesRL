import sys
import pdb
import numpy as np
import os
this_dir = os.path.dirname(os.path.abspath(__file__))
project_dir = os.path.join(this_dir, '..', '..')
sys.path.append(project_dir)

from src.environments.Glucose import Glucose
from src.estimation.TransitionModel import GlucoseTransitionModel
from src.policies.simulation_optimization_policies import solve_for_pi_opt


def evaluate_policy(initial_state, transition_model, time_horizon, policy, feature_function):
  """

  :param initial_state:
  :param transition_model:
  :param time_horizon:
  :param policy:
  :param feature_function:
  :return:
  """
  MC_REPLICATES = 100
  returns = []

  for _ in range(MC_REPLICATES):
    s = initial_state
    return_ = 0.0
    for t in range(time_horizon):
      a = policy(s)
      x = feature_function(s, a)
      s, r = transition_model(x)
      return_ += r
    returns.append(return_)
  return np.mean(returns)


def evaluate_glucose_mb_policy():
  # Roll out to get data
  n_patients = 10
  T = 20
  env = Glucose(n_patients)
  env.reset()
  env.step(np.random.binomial(1, 0.3, n_patients))

  for t in range(T):
    env.step(np.random.binomial(1, 0.3, n_patients))

  # Fit model on data
  estimator = GlucoseTransitionModel()
  X, Sp1 = env.get_state_transitions_as_x_y_pair()
  y = Sp1[:, 0]
  estimator.fit(X, y)

  # Get optimal policy under model
  def rollout_policy(s):
    return np.random.binomial(1, 0.3)

  initial_state = X[-1, :]
  transition_model = estimator.draw_from_ppd
  pi = solve_for_pi_opt(initial_state, transition_model, T, 2, rollout_policy, feature_function)

  # Evaluate policy
  v = evaluate_policy(initial_state, transition_model, T, pi, feature_function)

  return v
