import numpy as np
import copy
import logging
from matplotlib import colors
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


def simulate_from_transition_model(X_obs, transition_model, time_horizon, number_of_actions,
                                   rollout_policy, feature_function, mc_rollouts=100,
                                   reference_distribution_for_truncation=None):
  """
  ToDo: Simulate from each x in observed X instead of rolling out

  :param reference_distribution_for_truncation: if array of feature vectors provided, then reject samples that
  are outside bounds of those vectors.
  """
  NUMBER_OF_REPS_PER_X = 100

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
  S = []
  X = []
  R = []

  for x_obs in X_obs:
    for rep in range(NUMBER_OF_REPS_PER_X):
      s, r = sample_from_transition_model(x_obs)
      S.append(s)
      R.append(r)
      X.append(x_obs)
  return X, S, R


def solve_for_pi_opt(S_obs, X_obs, transition_model, time_horizon, number_of_actions, rollout_policy,
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
  X, S, R = simulate_from_transition_model(X_obs, S_obs, transition_model, time_horizon, number_of_actions,
                                           rollout_policy, feature_function, mc_rollouts=mc_rollouts,
                                           reference_distribution_for_truncation=reference_distribution_for_truncation)
  # Do FQI
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
  cmap = colors.ListedColormap(['white', 'red'])
  bounds = [0, 0.5, 1]
  norm = colors.BoundaryNorm(bounds, cmap.N)
  f, axarr = plt.subplots(2)
  img = axarr[0].imshow(policy_1_on_grid, cmap=cmap, norm=norm,
                        extent=[np.min(grid_1), np.max(grid_1), np.max(grid_2), np.min(grid_2)])
  axarr[1].imshow(policy_2_on_grid, cmap=cmap, norm=norm,
                  extent=[np.min(grid_1), np.max(grid_1), np.max(grid_2), np.min(grid_2)])
  plt.colorbar(img, cmap=cmap, norm=norm, boundaries=bounds)
  plt.show()

  return



