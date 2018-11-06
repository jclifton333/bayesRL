import sys
import pdb
import os
this_dir = os.path.dirname(os.path.abspath(__file__))
project_dir = os.path.join(this_dir, '..', '..')
sys.path.append(project_dir)

import datetime
import numpy as np
import src.policies.rollout as rollout
import src.estimation.dependent_density as dd
from src.environments.Glucose import Glucose
import src.policies.tuned_bandit_policies as policies
from theano import shared, tensor as tt


def episode(policy_name, label, save=False, monte_carlo_reps=100):
  if save:
    base_name = 'glucose-{}-{}'.format(label, policy_name)
    prefix = os.path.join(project_dir, 'src', 'run', 'results', base_name)
    suffix = datetime.datetime.now().strftime("%y%m%d_%H%M%S")
    filename = '{}_{}.yml'.format(prefix, suffix)

  np.random.seed(label)
  n_patients = 20
  T = 10

  tuning_function = policies.expit_epsilon_decay
  policy = policies.glucose_one_step_policy
  # ToDo: check these
  explore_ = {'zeta0': [1.0, 0.05, 1.0, 0.1], 'zeta1': [30.0, 0.0, 1.0, 0.0], 'zeta2': [0.1, 1.0, 0.01, 1.0]}
  bounds = {'zeta0': (0.025, 2.0), 'zeta1': (0.0, 30.0), 'zeta2': (0.01, 2)}
  env = Glucose(nPatients=n_patients)
  cumulative_regret = 0.0
  env.reset()

  for t in range(T):
    # Get posterior
    X, Sp1 = env.get_state_transitions_as_x_y_pair()
    X = shared(X)
    y = Sp1[:, 0]
    model_, trace_ = dd.dependent_density_regression(X, y)
    kwargs = {'n_reps': monte_carlo_reps, 'x_shared': X, 'model': model_, 'trace': trace_}

    tuning_function_parameter = opt.bayesopt(rollout.glucose_npb_rollout, policy, tuning_function,
                                             tuning_function_parameter, T, env, None, {}, bounds, explore_)

    # print('time {} epsilon {}'.format(t, tuning_function(T,t,tuning_function_parameter)))
    for j in range(nPatients):
      x = copy.copy(env.curr_context)

      beta_hat = env.beta_hat_list
      action = policy(beta_hat, env.sampling_cov_list, x, tuning_function, tuning_function_parameter, T, t, env)
      env.step(action)

      # Compute regret
      expected_rewards = [env.expected_reward(a, env.curr_context) for a in range(env.number_of_actions)]
      expected_reward_at_action = expected_rewards[action]
      optimal_expected_reward = np.max(expected_rewards)
      regret = optimal_expected_reward - expected_reward_at_action
      cumulative_regret += regret

    # Save results
    if save:
      results = {'t': float(t), 'regret': float(cumulative_regret)}
      with open(filename, 'w') as outfile:
        yaml.dump(results, outfile)

  return cumulative_regret