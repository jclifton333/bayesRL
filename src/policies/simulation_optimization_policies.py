import numpy as np
from sklearn.ensemble import RandomForestRegressor
import pdb


def glucose_feature_function(s, a, x):
  """

  :param s: array [glucose, food, activity]
  :param a:
  :param x: previous features
  :return:
  """

  x_new = np.array([1.0, s[0], s[1], s[2], x[1], x[2], x[3], a, x[-2]])
  return x_new


def simulate_from_transition_model(initial_state, initial_x, transition_model, time_horizon, number_of_actions,
                                   rollout_policy, feature_function, mc_rollouts=1000):
  # Generate data for fqi
  x_dim = len(initial_x)
  s_dim = len(initial_state)
  X = np.zeros((0, x_dim))
  S = np.zeros((0, s_dim))
  R = np.zeros(0)
  for rollout in range(mc_rollouts):
    s = initial_state
    x = initial_x
    for t in range(time_horizon):
      a = rollout_policy(s, x)
      x = feature_function(s, a, x)
      X = np.vstack((X, x))
      S = np.vstack((S, s))
      s, r = transition_model(np.array([x]))
      R = np.append(R, r)
  return X, S, R


def solve_for_pi_opt(initial_state, initial_x, transition_model, time_horizon, number_of_actions, rollout_policy,
                     feature_function, mc_rollouts=1000, number_of_dp_iterations=0):
  """
  Solve for optimal policy using dynamic programming.

  :param initial_x: initial features
  :param transition_model:
  :param time_horizon:
  :param rollout_policy: policy for generating data only
  :param number_of_dp_iterations:
  :return:
  """
  # Generate data for fqi
  X, S, R = simulate_from_transition_model(initial_state, initial_x, transition_model, time_horizon, number_of_actions,
                                           rollout_policy, feature_function, mc_rollouts=mc_rollouts)
  # Do FQI
  reg = RandomForestRegressor()
  reg.fit(X, R)
  q_ = lambda x_: reg.predict(x_.reshape(1, -1))
  for _ in range(number_of_dp_iterations):
    Q_ = R[:-1] + np.array([
      np.max([q_(feature_function(s, a, x)) for a in range(number_of_actions)]) for s, x in zip(S[1:], X[:1])])
    reg.fit(X[:-1], Q_)
    q_ = lambda x_: reg.predict(x_.reshape(1, -1))

  def pi_opt(s_, x_):
    return np.argmax([q_(feature_function(s_, a_, x_)) for a_ in range(number_of_actions)])

  return pi_opt