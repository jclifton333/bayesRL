"""
Policies in which exploration/exploitation tradeoff is parameterized and tuned (TS, UCB, ..?).
"""
import sys
import pdb
import os
this_dir = os.path.dirname(os.path.abspath(__file__))
project_dir = os.path.join(this_dir, '..', '..')
sys.path.append(project_dir)

from scipy.linalg import block_diag
from scipy.special import expit
import src.policies.linear_algebra as la
import pdb
import numpy as np
import copy


def tune_truncated_thompson_sampling(linear_model_results, time_horizon, current_time, estimated_context_mean,
                                     estimated_context_variance, truncation_function, truncation_function_gradient,
                                     initial_zeta):
  """
  :param linear_model_results: dictionary of lists of quantities related to estimated linear models for each action.
  :param time_horizon: 
  :param current_time:
  :param estimated_context_mean:
  :param estimated_context_variance:
  :param truncation_function: Function of (time_horizon-current_time) and parameter zeta (to be optimized), which
                              governs how much to adjust variance of approximate sampling dbn when thompson sampling.
  :param truncation_function_gradient:
  :param initial_zeta:
  :return: 
  """
  MAX_ITER = 100
  TOL = 0.01
  it = 0

  new_zeta = initial_zeta
  number_of_actions = len(estimated_context_mean)
  context_dimension = len(estimated_context_mean)
  zeta_dimension = len(new_zeta)
  policy_gradient = np.zeros(zeta_dimension)
  diff = float('inf')

  while it < MAX_ITER and diff > TOL:
    zeta = new_zeta

    # Sample from distributions that we'll use to determine ''true'' context and reward dbns in rollout
    working_context_mean = np.random.multivariate_normal(estimated_context_mean, estimated_context_variance)
    beta_hat = np.hstack(linear_model_results['beta_hat_list'])
    estimated_beta_hat_variance = block_diag(linear_model_results['sample_cov_list'][0],
                                             linear_model_results['sample_cov_list'][1])
    working_beta = np.random.multivariate_normal(beta_hat, estimated_beta_hat_variance)
    working_beta = working_beta.reshape((number_of_actions, context_dimension))
    # working_beta = beta_hat.reshape((number_of_actions, context_dimension))
    working_sigma_hats = linear_model_results['sigma_hat_list']
    rollout_linear_model_results = copy.copy(linear_model_results)
    for time in range(current_time + 1, time_horizon):
      # Draw beta
      shrinkage = truncation_function(time_horizon, time, zeta)
      beta_hat = np.hstack(rollout_linear_model_results['beta_hat_list'])
      estimated_beta_hat_variance = block_diag(rollout_linear_model_results['sample_cov_list'][0],
                                               rollout_linear_model_results['sample_cov_list'][1])

      beta = np.random.multivariate_normal(beta_hat, shrinkage * estimated_beta_hat_variance)
      beta = beta.reshape((number_of_actions, context_dimension))

      # Draw context
      context = np.random.multivariate_normal(working_context_mean, cov=np.eye(context_dimension))

      # Get action from predicted_rewards and get resulting reward
      predicted_rewards = np.dot(beta, context)
      action = np.argmax(predicted_rewards)
      working_beta_at_action = working_beta[action, :]
      working_sigma_hat_at_action = working_sigma_hats[action]
      reward = np.dot(working_beta_at_action, context) + np.random.normal(scale=np.sqrt(working_sigma_hat_at_action))

      # Update policy gradient
      sample_cov_hat_0 = rollout_linear_model_results['sample_cov_list'][0]
      sample_cov_hat_1 = rollout_linear_model_results['sample_cov_list'][1]
      beta_hat_0 = rollout_linear_model_results['beta_hat_list'][0]
      beta_hat_1 = rollout_linear_model_results['beta_hat_list'][1]

      policy_gradient += normal_ts_policy_gradient(0, context, sample_cov_hat_0, beta_hat_0, sample_cov_hat_1,
                                                   beta_hat_1, working_beta[0, :], working_beta[1, :],
                                                   truncation_function, truncation_function_gradient, time_horizon, time,
                                                   zeta)
      policy_gradient += normal_ts_policy_gradient(1, context, sample_cov_hat_0, beta_hat_0, sample_cov_hat_1,
                                                   beta_hat_1, working_beta[0, :], working_beta[1, :],
                                                   truncation_function, truncation_function_gradient, time_horizon,
                                                   time, zeta)

      # Update linear model
      rollout_linear_model_results = update_linear_model_at_action(action, rollout_linear_model_results, context,
                                                                   reward)
    # Update zeta
    step_size = 1e-3 / (it + 1)
    new_zeta = zeta + step_size * policy_gradient
    diff = np.linalg.norm(new_zeta - zeta) / np.linalg.norm(zeta)
    # print("zeta: {}".format(zeta))
    if not (0.01 < truncation_function(time_horizon, current_time, zeta) < 0.99):  # For stability
      break
    it += 1

  return zeta


def linear_cb_epsilon_greedy_policy(beta_hat, sampling_cov_list, context, tuning_function, tuning_function_parameter,
                                    T, t, env):
  epsilon = tuning_function(T, t, tuning_function_parameter)
  predicted_rewards = np.dot(beta_hat, context)
  greedy_action = np.argmax(predicted_rewards)
  if np.random.random() < epsilon:
    action = np.random.choice(2)
  else:
    action = greedy_action
  return action


def linear_cb_thompson_sampling_policy(beta_hat, sampling_cov_list, context, tuning_function, tuning_function_parameter,
                                       T, t, env):
  shrinkage = tuning_function(T, t, tuning_function_parameter)

  # Sample from estimated sampling dbn
  beta_hat_ = np.array(beta_hat).flatten()
  sampling_cov_ = block_diag(sampling_cov_list[0], sampling_cov_list[1])
  beta_tilde = np.random.multivariate_normal(beta_hat_, shrinkage * sampling_cov_)
  beta_tilde = beta_tilde.reshape(np.array(beta_hat).shape)

  # Estimate rewards and pull arm
  estimated_rewards = np.dot(beta_tilde, context)
  action = np.argmax(estimated_rewards)
  return action


def mab_epsilon_greedy_policy(estimated_means, standard_errors, number_of_pulls, tuning_function,
                              tuning_function_parameter, T, t, env):
  epsilon = tuning_function(T, t, tuning_function_parameter)
  greedy_action = np.argmax(estimated_means)
  if np.random.random() < epsilon:
    action = np.random.choice(2)
  else:
    action = greedy_action
  return action


# Helpers

def expit_truncate(T, t, zeta):
  shrinkage = expit(zeta[0] + zeta[1] * (T - t))
  return shrinkage


def expit_epsilon_decay(T, t, zeta):
  return zeta[0] * expit(zeta[1] + zeta[2]*(T - t))


def stepwise_linear_epsilon(T, t, zeta):
  J = len(zeta)
  interval = int(T/J)
  if t == 0:
    j = 0
  else:
    j = int(np.floor((T-t)/interval))
  epsilon = sum(zeta[:j]) + ((T-t) - j*interval) * zeta[j] / interval
  return epsilon







