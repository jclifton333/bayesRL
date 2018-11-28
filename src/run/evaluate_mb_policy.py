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
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt
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


def evaluate_glucose_mb_policy(replicate, method):
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
    estimator = GlucoseTransitionModel(method=method)
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

  elif method == 'one_step':  # One step FQI
    reg = RandomForestRegressor()
    X, Sp1 = env.get_state_transitions_as_x_y_pair()
    y = Sp1[:, 0]
    reg.fit(X, y)
    q_ = lambda x_: reg.predict(x_.reshape(1, -1))
    feature_function = opt.glucose_feature_function

    def pi(s_, x_):
      return np.argmax([q_(feature_function(s_, a_, x_)) for a_ in range(2)])

  elif method == 'two_step':
    reg = RandomForestRegressor()
    X, Sp1 = env.get_state_transitions_as_x_y_pair()
    y = Sp1[:, 0]

    # Step one
    reg.fit(X, y)
    q_ = lambda x_: reg.predict(x_.reshape(1, -1))
    feature_function = opt.glucose_feature_function

    # Step two
    Q_ = y[:-1] + np.array([np.max([q_(feature_function(s, a, x)) for a in range(2)])
                            for s, x in zip(Sp1[:-1], X[1:])])
    reg.fit(X[:-1], Q_)

    def pi(s_, x_):
      return np.argmax([q_(feature_function(s_, a_, x_)) for a_ in range(2)])

  # Evaluate policy
  # v = None
  v = evaluate_policy(T, pi)

  return v


def run():
  N_REPLICATES_PER_METHOD = 10
  N_PROCESSES = 2

  methods = ['np', 'p', 'averaged']
  results_dict = {}
  base_name = 'glucose-mb'
  prefix = os.path.join(project_dir, 'src', 'run', 'results', base_name)
  suffix = datetime.datetime.now().strftime("%y%m%d_%H%M%S")
  fname = '{}_{}.yml'.format(prefix, suffix)

  for method in methods:
    evaluate_partial = partial(evaluate_glucose_mb_policy, method=method)
    results = []
    pool = mp.Pool(N_PROCESSES)
    for rep in range(int(N_REPLICATES_PER_METHOD / N_PROCESSES)):
      res = pool.map(evaluate_partial, [2*rep, 2*rep + 1])
      results += res
    results_dict[method] = {'mean': float(np.mean(results)), 'se': float(np.std(results))}
    with open(fname, 'w') as outfile:
      yaml.dump(results_dict, outfile)
  print(results_dict)


if __name__ == "__main__":
  run()
