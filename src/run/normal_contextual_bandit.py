import sys
import pdb
import os
this_dir = os.path.dirname(os.path.abspath(__file__))
project_dir = os.path.join(this_dir, '..', '..')
sys.path.append(project_dir)


import matplotlib.pyplot as plt
from src.environments.Bandit import NormalCB
from src.policies import tuned_bandit_policies as tuned_bandit
import copy
import numpy as np
from scipy.linalg import block_diag


def run(policy_name):
  """

  :return:
  """

  # Simulation settings
  replicates = 100
  T = 25

  if policy_name == 'eps':
    tuning_function = lambda a, b, c: 0.05  # Constant epsilon
    policy = tuned_bandit.epsilon_greedy_policy
  elif policy_name == 'eps-decay':
    tuning_function = tuned_bandit.expit_epsilon_decay
    policy = tuned_bandit.epsilon_greedy_policy
  elif policy_name == 'ts':
    tuning_function = lambda a, b, c: 1.0  # No shrinkage
    policy = tuned_bandit.thompson_sampling_policy
  elif policy_name == 'ts-shrink':
    tuning_function = tuned_bandit.expit_truncate
    policy = tuned_bandit.thompson_sampling_policy
  else:
    raise ValueError('Incorrect policy name')

  tuning_function_parameter = np.array([-5, 0.3])  # Initial truncation function parameters
  env = NormalCB(list_of_reward_betas=[np.array([1.0, 1.0]), np.array([1.2, 1.2])])

  rewards = np.zeros((replicates, T))

  # Run sims
  for replicate in range(replicates):
    env.reset()

    # Initial pulls (so we can fit the models)
    for a in range(env.number_of_actions):
      for p in range(env.context_dimension):
        env.step(a)

    # Get initial linear model
    X, A, U = env.X, env.A, env.U
    linear_model_results = {'beta_hat_list': [], 'Xprime_X_inv_list': [], 'X_list': [], 'X_dot_y_list': [],
                            'sample_cov_list': [], 'sigma_hat_list': []}
    for a in range(env.number_of_actions):  # Fit linear model on data from each actions
      # Get observations where action == a
      indices_for_a = np.where(A == a)
      X_a = X[indices_for_a]
      U_a = U[indices_for_a]

      Xprime_X_inv_a = np.linalg.inv(np.dot(X_a.T, X_a))
      X_dot_y_a = np.dot(X_a.T, U_a)
      beta_hat_a = np.dot(Xprime_X_inv_a, X_dot_y_a)
      yhat_a = np.dot(X_a, beta_hat_a)
      sigma_hat_a = np.sum((U_a - yhat_a)**2)
      sample_cov_a = sigma_hat_a * Xprime_X_inv_a
      # Add to linear model results
      linear_model_results['beta_hat_list'].append(beta_hat_a)
      linear_model_results['Xprime_X_inv_list'].append(Xprime_X_inv_a)
      linear_model_results['X_list'].append(X_a)
      linear_model_results['X_dot_y_list'].append(X_dot_y_a)
      linear_model_results['sample_cov_list'].append(sample_cov_a)
      linear_model_results['sigma_hat_list'].append(sigma_hat_a)

    # Estimate context mean and variance
    estimated_context_mean = np.mean(X, axis=0)
    estimated_context_variance = np.cov(X, rowvar=False)

    for t in range(T):
      X = env.X
      tuning_function_parameter = tuned_bandit.bayesopt(tuned_bandit.rollout, policy, tuning_function, tuning_function_parameter,
                                                        linear_model_results, T, t, np.mean(X, axis=0),
                                                        np.cov(X, rowvar=False))
      x = copy.copy(env.curr_context)
      action = policy(beta_hat, linear_model_results['sample_cov_list'], x, tuning_function,
                      tuning_function_parameter, T, t)
      step_results = env.step(action)
      reward = step_results['Utility']

      # Compute regret
      oracle_expected_reward = np.max((np.dot(x, env.list_of_reward_betas[0]), np.dot(x, env.list_of_reward_betas[1])))
      regret = oracle_expected_reward - np.dot(x, env.list_of_reward_betas[a])
      rewards[replicate, t] = regret

      # Update linear model
      linear_model_results = tuned_bandit.update_linear_model_at_action(a, linear_model_results, x, reward)
    print('cumulative rewards: {}'.format(np.sum(rewards[replicate, :])))
  cumulative_rewards = np.sum(rewards, axis=1)
  print('mean cumulative rewards: {} sd cumulative rewards {}'.format(np.mean(cumulative_rewards),
                                                                      np.std(cumulative_rewards)))
  plt.hist(cumulative_rewards)
  plt.show()
  return rewards


if __name__ == '__main__':
  main()


