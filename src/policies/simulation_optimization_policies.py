import numpy as np
import logging
from sklearn.ensemble import RandomForestRegressor
import pdb


def glucose_feature_function(s, a, x):
  """

  :param s: array [glucose, food, activity]
  :param a:
  :param x: previous features
  :return:
  """

  x_new = np.array([1.0, s[0], s[1], s[2], x[1], x[2], x[3], a, x[-1]])
  return x_new


def simulate_from_transition_model(initial_state, initial_x, transition_model, time_horizon, number_of_actions,
                                   rollout_policy, feature_function, mc_rollouts=100,
                                   reference_distribution_for_truncation=None):
  """
  :param reference_distribution_for_truncation: if array of feature vectors provided, then reject samples that
  are outside bounds of those vectors.
  """
  if reference_distribution_for_truncation is not None:
    max_ = np.max(reference_distribution_for_truncation, axis=0)
    min_ = np.min(reference_distribution_for_truncation, axis=0)

    def sample_from_transition_model(x):
      s, r = transition_model(np.array([x]))
      out_of_bounds = np.any(s > max_)*np.any(s < min_)
      counter = 0.0
      max_reject = 10
      while counter < max_reject and out_of_bounds:
        s, r = transition_model(np.array([x]))
        out_of_bounds = np.any(s > max_)*np.any(s < min_)
        counter += 1
      if counter == max_reject:
        logging.warning('Max rejections reached!')
      return s, r

  else:
    def sample_from_transition_model(x):
      s, r = transition_model([x])
      return s, r

  # Generate data for fqi
  x_dim = len(initial_x)
  s_dim = len(initial_state)
  # X = np.zeros((0, x_dim))
  # S = np.zeros((0, s_dim))
  S = []
  X = []
  R = []
  for rollout in range(mc_rollouts):
    print(rollout)
    s = initial_state
    x = initial_x
    S_rep = np.zeros((0, s_dim))
    X_rep = np.zeros((0, x_dim))
    R_rep = np.zeros(0)
    for t in range(time_horizon):
      a = rollout_policy(s, x)
      x = feature_function(s, a, x)
      X_rep = np.vstack((X_rep, x))
      S_rep = np.vstack((S_rep, s))
      s, r = sample_from_transition_model(x)
      R_rep = np.append(R_rep, r)
    S.append(S_rep)
    X.append(X_rep)
    R.append(R_rep)
  return X, S, R


def solve_for_pi_opt(initial_state, initial_x, transition_model, time_horizon, number_of_actions, rollout_policy,
                     feature_function, mc_rollouts=100, number_of_dp_iterations=0,
                     reference_distribution_for_truncation=None):
  """
  Solve for optimal policy using dynamic programming.

  :param initial_x: initial features
  :param transition_model:
  :param time_horizon:
  :param rollout_policy: policy for generating data only
  :param number_of_dp_iterations:
  :param reference_distribution_for_truncation: if array of feature vectors provided, then reject samples that
  are outside bounds of those vectors.
  :return:
  """
  # Generate data for fqi
  X, S, R = simulate_from_transition_model(initial_state, initial_x, transition_model, time_horizon, number_of_actions,
                                           rollout_policy, feature_function, mc_rollouts=mc_rollouts,
                                           reference_distribution_for_truncation=reference_distribution_for_truncation)
  # Do FQI
  X = np.vstack([X_[:-1, :] for X_ in X])
  S = np.vstack([S_[1:, :] for S_ in S])
  R = np.hstack([R_[1:] for R_ in R])
  reg = RandomForestRegressor()
  reg.fit(X, R)
  q_ = lambda x_: reg.predict(x_.reshape(1, -1))
  for _ in range(number_of_dp_iterations):
    Q_ = R + np.array([
      np.max([q_(feature_function(s, a, x)) for a in range(number_of_actions)]) for s, x in zip(S, X)])
    reg.fit(X, Q_)
    q_ = lambda x_: reg.predict(x_.reshape(1, -1))

  def pi_opt(s_, x_):
    return np.argmax([q_(feature_function(s_, a_, x_)) for a_ in range(number_of_actions)])

  return pi_opt