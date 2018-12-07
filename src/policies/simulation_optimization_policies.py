import numpy as np
import copy
import logging
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt
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


def compare_glucose_policies(fixed_covariates, coordinate_to_vary_1, coordinate_to_vary_2, policy_1, policy_2):
  """
  Visualize glucose policies on a grid of two covariates (keeping the others in fixed_covariates fixed.)

  :param fixed_covariates:
  :param coordinate_to_vary_1: index of first covariate to vary
  :param coordinate_to_vary_2: index of second covariate to vary
  :param policy_1:
  :param policy_2:
  :return:
  """
  NUM_GRIDPOINTS = 50

  x_ = np.array([1.0, 80.0, 0, 2, 85.0, 1.5, -1, 0, 1])  # Need x from prveious time step for glucose policies

  # Get grid of values at which to evaluate policy
  if coordinate_to_vary_1 == 0:  # Glucose
    grid_1 = np.linspace(50, 200, NUM_GRIDPOINTS)
  else:  # Food or ex
    grid_1 = np.linspace(-3, 3, NUM_GRIDPOINTS)
  if coordinate_to_vary_2 == 0:  # Glucose
    grid_2 = np.linspace(50, 200, NUM_GRIDPOINTS)
  else:  # Food or ex
    grid_2 = np.linspace(-3, 3, NUM_GRIDPOINTS)

  # Evaluate each policy on grid
  policy_1_on_grid = np.zeros((NUM_GRIDPOINTS, NUM_GRIDPOINTS))
  policy_2_on_grid = np.zeros((NUM_GRIDPOINTS, NUM_GRIDPOINTS))
  for i, s_i in enumerate(grid_1):
    for j, s_j in enumerate(grid_2):
      s_ij = copy.copy(fixed_covariates)
      s_ij[coordinate_to_vary_1] = s_i
      s_ij[coordinate_to_vary_2] = s_j
      policy_1_ij = policy_1(s_ij, x_)
      policy_2_ij = policy_2(s_ij, x_)
      policy_1_on_grid[i, j] = policy_1_ij
      policy_2_on_grid[i, j] = policy_2_ij

  # Visualize policies
  f, axarr = plt.subplot(2)
  axarr[0].imshow(policy_1_on_grid)
  axarr[1].imshow(policy_2_on_grid)
  plt.show()

  return



