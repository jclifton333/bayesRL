"""
Functions for assessing model-based glucose policies.
"""


import sys
import pdb
import numpy as np
import os
this_dir = os.path.dirname(os.path.abspath(__file__))
project_dir = os.path.join(this_dir, '..', '..')
sys.path.append(project_dir)

from src.environments.Glucose import Glucose
from src.estimation.TransitionModel import GlucoseTransitionModel, glucose_reward_function
import src.policies.simulation_optimization_policies as opt
from sklearn.ensemble import RandomForestRegressor
try:
  import matplotlib.pyplot as plt
except:
  pass
import multiprocessing as mp
from functools import partial
import datetime
import yaml


def evaluate_policy(time_horizon, policy, initial_state_and_x=None):
  """
  :param initial_state:
  :param initial_x: initial features
  :param time_horizon:
  :param policy:
  :param feature_function:
  :return:
  """
  MC_REPLICATES = 100
  env_ = Glucose(nPatients=100)
  env_.reset()
  for t in range(time_horizon):
    actions = []
    for patient in range(MC_REPLICATES):
      s = env_.S[patient][-1, :]
      x = env_.X[patient][-1, :]
      a = policy(s, x)
      actions.append(a)
    env_.step(actions)
  return np.mean(env_.R)


def rollout_and_fit_unconditional_density(test=False):
  # Roll out to get data
  n_patients = 20
  T = 20
  env = Glucose(n_patients)
  env.reset()
  env.step(np.random.binomial(1, 0.3, n_patients))

  for t in range(T):
    env.step(np.random.binomial(1, 0.3, n_patients))

  # Fit model on data
  estimator = GlucoseTransitionModel(test=test)
  X, _ = env.get_state_transitions_as_x_y_pair()
  estimator.fit_unconditional_densities(X)
  return estimator


def rollout_and_fit_density(method='np', alpha_mean=0.0, test=False):
  # Roll out to get data
  n_patients = 20
  T = 20
  env = Glucose(n_patients)
  env.reset()
  env.step(np.random.binomial(1, 0.3, n_patients))

  for t in range(T):
    env.step(np.random.binomial(1, 0.3, n_patients))

  # Fit model on data
  estimator = GlucoseTransitionModel(method=method, alpha_mean=alpha_mean, test=test)
  X, Sp1 = env.get_state_transitions_as_x_y_pair()
  y = Sp1[:, 0]
  # estimator.fit(X, y)
  estimator.fit_p_conditional_density(X, y)
  return estimator


def plot_conditional_density_estimates(alpha_mean=0.0, test=False):
  estimator = rollout_and_fit_density(method='np', alpha_mean=alpha_mean, test=test)

  # Conditional density plots
  estimator.plot_regression_line()
  estimator.plot_density_estimates()
  return


def trajectory_ppc(replicate):
  """
  Generate data, fit np model, and compare observed trajectory to posterior predictive trajectories.

  :return:
  """
  np.random.seed(replicate)

  # Roll out to get data
  n_patients = 20
  T = 20
  env = Glucose(n_patients)
  env.reset()
  env.step(np.random.binomial(1, 0.3, n_patients))

  for t in range(T):
    env.step(np.random.binomial(1, 0.3, n_patients))

  # Fit model on data
  estimator = GlucoseTransitionModel(method='np')
  X, Sp1 = env.get_state_transitions_as_x_y_pair()
  y = Sp1[:, 0]
  estimator.fit(X, y)

  # Simulate data starting at initial state of first patient
  initial_state = env.S[0][0, :]
  initial_x = env.X[0][0, :]
  _, S_sim, R = opt.simulate_from_transition_model(initial_state, initial_x, estimator.draw_from_ppd, T, 2,
                                                   lambda s_, x_: np.random.binomial(1, 0.3),
                                                   opt.glucose_feature_function)

  # PPC for different time points
  times = [4, 9, 14, 19]
  f, axarr = plt.subplots(len(times))
  S_sim = np.array(S_sim)
  for i, t in enumerate(times):
    y, x, _ = axarr[i].hist(S_sim[:, t, 0])
    axarr[i].vlines(env.S[0][t, 0], ymin=0, ymax=y.max())
  plt.show()

  return


def fit_and_compare_mb_and_mf_policies(test=False):
  np.random.seed(0)

  # Roll out to get data
  n_patients = 20
  T = 20
  env = Glucose(n_patients)
  env.reset()
  env.step(np.random.binomial(1, 0.3, n_patients))

  for t in range(T):
    env.step(np.random.binomial(1, 0.3, n_patients))
  # ToDo: implement function to encapsulate fitting mb policy
  X, Sp1 = env.get_state_transitions_as_x_y_pair()
  y = Sp1[:, 0]

  # Get policy under true model
  env = Glucose()
  feature_function = opt.glucose_feature_function

  def rollout_policy(s_, x_):
    return np.random.binomial(1, 0.3)

  def true_transition_model(x):
    f, e = env.generate_food_and_activity()
    g = np.dot(x, env.COEF) + np.random.normal(0, env.SIGMA_GLUCOSE)
    return np.array([g, f, e]), glucose_reward_function(g)

  policy_true = opt.solve_for_pi_opt(X, true_transition_model, T, 2, rollout_policy, feature_function,
                                     number_of_dp_iterations=1)
  # Fit mf policy
  reg = RandomForestRegressor()
  R = np.array([glucose_reward_function(g) for g in y])

  # Step one
  reg.fit(X, R)
  q_ = lambda x_: reg.predict(x_.reshape(1, -1))
  feature_function = opt.glucose_feature_function

  # Step two
  Q_ = R[:-1] + np.array([np.max([q_(feature_function(s, a, x)) for a in range(2)])
                          for s, x in zip(Sp1[:-1], X[1:])])
  reg.fit(X[:-1], Q_)

  def policy_mf(s_, x_):
    return np.argmax([q_(feature_function(s_, a_, x_)) for a_ in range(2)])

  # Get optimal policy under model
  estimator = GlucoseTransitionModel(test=test)
  estimator.fit(X, y)

  transition_model = estimator.draw_from_ppd

  policy_mb = opt.solve_for_pi_opt(X, transition_model, T, 2, rollout_policy, feature_function,
                                   number_of_dp_iterations=1)
  policy_list = [('policy_true', policy_true), ('policy_mb', policy_mb), ('policy_mf', policy_mf)]

  # Compare
  opt.compare_glucose_policies(np.array([80, 0, 0]), 0, 1, policy_list)

  return


def evaluate_glucose_mb_policy(replicate, method, test=False, truncate=False, alpha_mean=0.0):
  """

  :param replicate:
  :param method:
  :param truncate:
  :param alpha_mean: Prior hyperparameter, for mb policies.
  :return:
  """

  np.random.seed(replicate)

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
    estimator = GlucoseTransitionModel(method=method, alpha_mean=alpha_mean, test=test)
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
    if truncate:
      reference_distribution_for_truncation = Sp1
    else:
      reference_distribution_for_truncation = None
    pi = opt.solve_for_pi_opt(initial_state, initial_x, transition_model, T, 2, rollout_policy, feature_function,
                              number_of_dp_iterations=1,
                              reference_distribution_for_truncation=reference_distribution_for_truncation)

  elif method == 'random':
    def pi(s_, x_):
      return np.random.binomial(1, 0.3)

  elif method == 'one_step':  # One step FQI
    reg = RandomForestRegressor()
    X, Sp1 = env.get_state_transitions_as_x_y_pair()
    y = Sp1[:, 0]
    R = np.array([glucose_reward_function(g) for g in y])
    reg.fit(X, R)
    q_ = lambda x_: reg.predict(x_.reshape(1, -1))
    feature_function = opt.glucose_feature_function

    def pi(s_, x_):
      return np.argmax([q_(feature_function(s_, a_, x_)) for a_ in range(2)])

  elif method == 'two_step':
    reg = RandomForestRegressor()
    X, Sp1 = env.get_state_transitions_as_x_y_pair()
    y = Sp1[:, 0]
    R = np.array([glucose_reward_function(g) for g in y])

    # Step one
    reg.fit(X, R)
    q_ = lambda x_: reg.predict(x_.reshape(1, -1))
    feature_function = opt.glucose_feature_function

    # Step two
    Q_ = R[:-1] + np.array([np.max([q_(feature_function(s, a, x)) for a in range(2)])
                            for s, x in zip(Sp1[:-1], X[1:])])
    reg.fit(X[:-1], Q_)

    def pi(s_, x_):
      return np.argmax([q_(feature_function(s_, a_, x_)) for a_ in range(2)])

  elif method == 'true_model':
    # Get policy under true model
    feature_function = opt.glucose_feature_function
    X, Sp1 = env.get_state_transitions_as_x_y_pair()

    def rollout_policy(s_, x_):
      return np.random.binomial(1, 0.3)

    def true_transition_model(x):
      f, e = env.generate_food_and_activity()
      g = np.dot(x, env.COEF) + np.random.normal(0, env.SIGMA_NOISE)
      return np.array([g, f, e]), glucose_reward_function(g)

    pi = opt.solve_for_pi_opt(X, true_transition_model, T, 2, rollout_policy, feature_function,
                              number_of_dp_iterations=1)
  # v_mb_, v_mf_ = estimator.one_step_value_function_ppc(X, S, R)
  # Evaluate policy
  # v = None
  v = evaluate_policy(T, pi)

  return v


def run():
  N_REPLICATES_PER_METHOD = 20
  N_PROCESSES = 20

  # methods = ['np', 'p', 'averaged']
  # alphas = [-1.0, 0.0, 0.5, 1.0, 5.0]
  # methods = [('np', alpha) for alpha in alphas] + [('p', 0), ('averaged', 0)]
  methods = [('np', 0.0), ('averaged', 0)]
  results_dict = {}
  base_name = 'glucose-mb'
  prefix = os.path.join(project_dir, 'src', 'run', 'results', base_name)
  suffix = datetime.datetime.now().strftime("%y%m%d_%H%M%S")
  fname = '{}_{}.yml'.format(prefix, suffix)

  for method, alpha_mean in methods:
    evaluate_partial = partial(evaluate_glucose_mb_policy, method=method, alpha_mean=alpha_mean)
    results = []
    pool = mp.Pool(N_PROCESSES)
    for rep in range(int(N_REPLICATES_PER_METHOD / N_PROCESSES)):
      res = pool.map(evaluate_partial, range(20+rep*N_REPLICATES_PER_METHOD + 20+rep*N_REPLICATES_PER_METHOD + N_PROCESSES))
      results += res
    method_name = '{}-alpha={}'.format(method, alpha_mean)
    results_dict[method_name] = {'mean': float(np.mean(results)), 'se': float(np.std(results))}
    with open(fname, 'w') as outfile:
      yaml.dump(results_dict, outfile)
  print(results_dict)


if __name__ == "__main__":
  rollout_and_fit_unconditional_density()
  # fit_and_compare_mb_and_mf_policies(test=False)
  # evaluate_glucose_mb_policy(0, method='true_model')
