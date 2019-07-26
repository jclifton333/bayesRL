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


def episode(label, policy_name, T, save=False, monte_carlo_reps=10):
  decay_function = lambda t: 1 / t

  # if policy_name in ['np', 'p', 'averaged']:
  if policy_name in ['kde', 'ar2', 'ar1']:
    tune = True
    fixed_eps = None
    eps = None
    if policy_name == 'kde':
      estimator = transition.KdeGlucoseModel()
    else:
      estimator = transition.LinearGlucoseModel(ar1=(policy_name=='ar1'))
  else:
    tune = False
    if policy_name == 'fixed_eps':
      eps = 0.05

  np.random.seed(label)
  n_patients = 15
  previous_q = None  # Starts off as None, updated in policy function

  tuning_function = policies.expit_epsilon_decay
  policy = policies.glucose_fitted_q
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

  for t in range(T):
    print(t)
    # Get estimated dyna proportion
    if tune:
      X, Sp1 = env.get_state_transitions_as_x_y_pair()
      y = Sp1[:, 0]
      estimator.fit(X, y)
      kwargs = {'n_rep': monte_carlo_reps, 'estimator': estimator, 'decay_function': decay_function}

      tuning_function_parameter = opt.bayesopt(rollout.glucose_npb_rollout, policy, tuning_function,
                                               tuning_function_parameter, T, env, None, kwargs, bounds, explore_)

    eps = decay_function(t)
    action, previous_q = policy(env, estimator, tuning_function, tuning_function_parameter, T, t, previous_q,
                                epsilon=eps)
    _, r = env.step(action)
    cumulative_reward += r

    # Save results
    # if save:
    #   base_name = 'glucose-tuned={}-policy={}-{}'.format(tune, policy_name, label)
    #   prefix = os.path.join(project_dir, 'src', 'run', 'results', base_name)
    #   suffix = datetime.datetime.now().strftime("%y%m%d_%H%M%S")
    #   filename = '{}_{}.yml'.format(prefix, suffix)
    #   results = {'t': float(t), 'regret': float(cumulative_reward)}
    #   with open(filename, 'w') as outfile:
    #     yaml.dump(results, outfile)

  return {'cumulative_reward': float(cumulative_reward), 'epsilon_list': epsilon_list}


def run(policy_name, T, decay_function=None):
  replicates = 24
  num_cpus = replicates
  pool = mp.Pool(processes=num_cpus)

  episode_partial = partial(episode, policy_name=policy_name, T=T, decay_function=decay_function)
  results = pool.map(episode_partial, range(replicates))

  base_name = 'glucose-{}'.format(policy_name)
  prefix = os.path.join(project_dir, 'src', 'run', base_name)
  suffix = datetime.datetime.now().strftime("%y%m%d_%H%M%S")
  filename = '{}_{}.yml'.format(prefix, suffix)
  rewards = [d['cumulative_reward'] for d in results]
  epsilons = [d['epsilon_list'] for d in results]
  results_to_save = {'mean': float(np.mean(rewards)),
                     'se': float(np.std(rewards) / np.sqrt(len(rewards))), 'epsilon_list': epsilons}
  with open(filename, 'w') as outfile:
    yaml.dump(results_to_save, outfile)

  return


if __name__ == '__main__':
  episode(0, 'ar2', 10)


