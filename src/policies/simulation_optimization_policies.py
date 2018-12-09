import numpy as np
import copy
import logging
from sklearn.ensemble import RandomForestRegressor
try:
  from matplotlib import colors
  import matplotlib.pyplot as plt
except:
  pass
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


def simulate_from_transition_model(X_obs, transition_model, test=False, reference_distribution_for_truncation=None):
  """
  ToDo: Simulate from each x in observed X instead of rolling out

  :param reference_distribution_for_truncation: if array of feature vectors provided, then reject samples that
  are outside bounds of those vectors.
  """
  if test:
    NUMBER_OF_REPS_PER_X=1
  else:
    NUMBER_OF_REPS_PER_X = 20

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
  S = np.zeros((0, 3)) 
  X = np.zeros((0, X_obs.shape[1]))
  R = np.zeros(0)

  print('number of obss: {}'.format(X_obs.shape[0]))
  for x_obs in X_obs:
    for rep in range(NUMBER_OF_REPS_PER_X):
      print(rep)
      s, r = sample_from_transition_model(x_obs)
      S = np.vstack((S, s))
      R = np.append(R, r)
      X = np.vstack((X, x_obs))
  return X, S, R


def solve_for_pi_opt(X_obs, transition_model, time_horizon, number_of_actions, rollout_policy,
                     feature_function, mc_rollouts=100, number_of_dp_iterations=0,
                     reference_distribution_for_truncation=None, test=False):
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
  X, S, R = simulate_from_transition_model(X_obs, transition_model, test=test,
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


def compare_glucose_policies(policy_list, label):
  """
  Visualize glucose policies on a grid of two covariates (keeping the others in fixed_covariates fixed.)

  :param fixed_covariates:
  :param coordinate_to_vary_1: index of first covariate to vary
  :param coordinate_to_vary_2: index of second covariate to vary
  :param policy_list: list of tuples (policy name (str), policy (function))
  :return:
  """
  NUM_GRIDPOINTS = 50
  number_of_policies = len(policy_list)

  x_ = np.array([1.0, 80.0, 0, 2, 85.0, 1.5, -1, 0, 1])  # Need x from prveious time step for glucose policies
  s_ = np.array([80.0, 0.0, 0.0])

  glucose_grid = np.linspace(50, 200, NUM_GRIDPOINTS)
  food_grid = exercise_grid = np.linspace(-25, 25, NUM_GRIDPOINTS)

  # Initialize policies, varying 2 covariates and keeping one fixed
  food_and_glucose_policies = [np.zeros((NUM_GRIDPOINTS, NUM_GRIDPOINTS)) for _ in range(number_of_policies)]
  exercise_and_glucose_policies = [np.zeros((NUM_GRIDPOINTS, NUM_GRIDPOINTS)) for _ in range(number_of_policies)]
  food_and_exercise_policies_normal = [np.zeros((NUM_GRIDPOINTS, NUM_GRIDPOINTS)) for _ in range(number_of_policies)]
  food_and_exercise_policies_hypo = [np.zeros((NUM_GRIDPOINTS, NUM_GRIDPOINTS)) for _ in range(number_of_policies)]
  food_and_exercise_policies_hyper = [np.zeros((NUM_GRIDPOINTS, NUM_GRIDPOINTS)) for _ in range(number_of_policies)]

  # Evaluate each policy on grid
  for i, f in enumerate(food_grid):
    for j, e in enumerate(exercise_grid):
      s_ij_normal = [85, f, e]
      s_ij_hyper = [150, f, e]
      s_ij_hypo = [55, f, e]

      for policy_ix, policy_tuple in enumerate(policy_list):
        food_and_exercise_policies_hyper[policy_ix][i, j] = policy_tuple[1](s_ij_hyper, x_)
        food_and_exercise_policies_normal[policy_ix][i, j] = policy_tuple[1](s_ij_normal, x_)
        food_and_exercise_policies_hypo[policy_ix][i, j] = policy_tuple[1](s_ij_hypo, x_)

  for i, g in enumerate(glucose_grid):
    for j, f in enumerate(food_grid):
      s_ij = [g, f, 0]
      for policy_ix, policy_tuple in enumerate(policy_list):
        food_and_glucose_policies[policy_ix][i, j] = policy_tuple[1](s_ij, x_)
    for j, e in enumerate(exercise_grid):
      s_ij = [g, 0, e]
      for policy_ix, policy_tuple in enumerate(policy_list):
        exercise_and_glucose_policies[policy_ix][i, j] = policy_tuple[1](s_ij, x_)

  # Visualize policies
  cmap = colors.ListedColormap(['white', 'red'])
  bounds = [0, 0.5, 1]
  norm = colors.BoundaryNorm(bounds, cmap.N)
  fig, axarr = plt.subplots(nrows=number_of_policies, ncols=5)
  for ix in range(number_of_policies):
    axarr[ix, 0].imshow(food_and_glucose_policies[ix], cmap=cmap, norm=norm,
                        extent=[np.min(glucose_grid), np.max(glucose_grid), np.max(food_grid), np.min(food_grid)])
    axarr[ix, 1].imshow(exercise_and_glucose_policies[ix], cmap=cmap, norm=norm,
                        extent=[np.min(glucose_grid), np.max(glucose_grid), np.max(exercise_grid),
                                np.min(exercise_grid)])
    axarr[ix, 2].imshow(food_and_exercise_policies_normal[ix], cmap=cmap, norm=norm,
                        extent=[np.min(food_grid), np.max(food_grid), np.max(exercise_grid), np.min(exercise_grid)])
    axarr[ix, 3].imshow(food_and_exercise_policies_hypo[ix], cmap=cmap, norm=norm,
                        extent=[np.min(food_grid), np.max(food_grid), np.max(exercise_grid), np.min(exercise_grid)])
    axarr[ix, 4].imshow(food_and_exercise_policies_hyper[ix], cmap=cmap, norm=norm,
                        extent=[np.min(food_grid), np.max(food_grid), np.max(exercise_grid), np.min(exercise_grid)])

    # Set row and column titles
    axarr[ix, 0].set_ylabel(policy_list[ix][0], rotation=0, size='large')
    if ix == 0:
      axarr[ix, 0].set_title('Food x glucose\n(Exercise=0)')
      axarr[ix, 1].set_title('Exercise x glucose\n(Food=0)')
      axarr[ix, 2].set_title('Food x exercise\n(Glucose=85)')
      axarr[ix, 3].set_title('Food x exercise\n(Glucose=55)')
      axarr[ix, 4].set_title('Food x exercise\n(Glucose=150)')
  fig.tight_layout()
  plt_name = 'policy-comparison-{}.png'.format(label)
  plt.savefig(plt_name)
  # plt.show()

  return



