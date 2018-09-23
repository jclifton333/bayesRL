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


def update_linear_model(X, y, Xprime_X_inv, x_new, X_dot_y, y_new):
  # Compute new beta hat and associated matrices
  Xprime_X_inv_new = la.sherman_woodbury(Xprime_X_inv, x_new, x_new)
  X_new = np.vstack((X, x_new.reshape(1, -1)))
  X_dot_y_new = X_dot_y + y_new * x_new
  beta_hat_new = la.matrix_dot_vector(Xprime_X_inv_new, X_dot_y_new)

  # Compute new sample covariance
  n, p = X_new.shape
  yhat = la.matrix_dot_vector(X_new, beta_hat_new)
  # sigma_hat = np.sum((yhat - y_new)**2) / (n - p)
  y = np.append(y, y_new)
  sigma_hat = la.sse(yhat, y) / (n - p)
  sample_cov = sigma_hat * Xprime_X_inv_new

  return {'beta_hat': beta_hat_new, 'Xprime_X_inv': Xprime_X_inv_new, 'X': X_new, 'y': y, 'X_dot_y': X_dot_y_new,
          'sample_cov': sample_cov, 'sigma_hat': sigma_hat}


def add_linear_model_results_at_action_to_dictionary(a, linear_model_results, linear_model_results_for_action):
  linear_model_results['beta_hat_list'][a] = linear_model_results_for_action['beta_hat']
  linear_model_results['Xprime_X_inv_list'][a] = linear_model_results_for_action['Xprime_X_inv']
  linear_model_results['X_list'][a] = linear_model_results_for_action['X']
  linear_model_results['X_dot_y_list'][a] = linear_model_results_for_action['X_dot_y']
  linear_model_results['sample_cov_list'][a] = linear_model_results_for_action['sample_cov']
  linear_model_results['sigma_hat_list'][a] = linear_model_results_for_action['sigma_hat']
  linear_model_results['y_list'][a] = linear_model_results_for_action['y']
  return linear_model_results


def update_linear_model_at_action(a, linear_model_results, x_new, y_new):
  """
  Linear model results is a dictionary of lists.
  Each list is a list of corresponding matrices/arrays of observations, indexed by actions a=0,...,nA.
  :param a:
  :param linear_model_results:
  :param x_new
  :param y_new:
  :return:
  """
  X_a = linear_model_results['X_list'][a]
  y_a = linear_model_results['y_list'][a]
  Xprime_X_inv_a = linear_model_results['Xprime_X_inv_list'][a]
  X_dot_y_a = linear_model_results['X_dot_y_list'][a]
  updated_linear_model_results_for_action = update_linear_model(X_a, y_a, Xprime_X_inv_a, x_new, X_dot_y_a, y_new)
  linear_model_results = add_linear_model_results_at_action_to_dictionary(a, linear_model_results,
                                                                          updated_linear_model_results_for_action)
  return linear_model_results


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


def epsilon_greedy_policy(beta_hat, sampling_cov_list, context, tuning_function, tuning_function_parameter,
                          T, t):
  epsilon = tuning_function(T, t, tuning_function_parameter)
  predicted_rewards = np.dot(beta_hat, context)
  greedy_action = np.argmax(predicted_rewards)
  if np.random.random() < epsilon:
    action = np.random.choice(2)
  else:
    action = greedy_action
  return action


def thompson_sampling_policy(beta_hat, sampling_cov_list, context, tuning_function, tuning_function_parameter,
                             T, t):
  shrinkage = tuning_function(T, t, tuning_function_parameter)

  # Sample from estimated sampling dbn
  beta_hat_ = beta_hat.flatten()
  sampling_cov_ = block_diag(sampling_cov_list[0], sampling_cov_list[1])
  beta_tilde = np.random.multivariate_normal(beta_hat_, shrinkage * sampling_cov_)
  beta_tilde = beta_tilde.reshape(beta_hat.shape)

  # Estimate rewards and pull arm
  estimated_rewards = np.dot(beta_tilde, context)
  action = np.argmax(estimated_rewards)
  return action


def oracle_rollout(tuning_function_parameter, policy, linear_model_results, time_horizon, current_time,
                   estimated_context_mean, tuning_function, estimated_context_variance, env, context_sequences):
  # MAX_ITER = 100
  it = 0

  number_of_actions = env.number_of_actions
  context_dimension = env.context_dimension

  score = 0

  # while it < MAX_ITER:
  for context_sequence in context_sequences:  # We use a pre-specified context sequence to reduce variance of comparisons
    # Rollout under drawn working model
    rollout_linear_model_results = copy.deepcopy(linear_model_results)
    episode_score = 0
    for time in range(current_time, time_horizon):
      context = context_sequence[time - current_time]
      beta_hat = np.hstack(rollout_linear_model_results['beta_hat_list']).reshape((number_of_actions,
                                                                                   context_dimension))
      sampling_cov_list = linear_model_results['sample_cov_list']

     # Draw context and take action
      # context = env.draw_context()
      action = policy(beta_hat, sampling_cov_list, context, tuning_function, tuning_function_parameter, time_horizon,
                      time)

      # Get reward from pulled arm
      expected_rewards = [env.expected_reward(a, context) for a in range(env.number_of_actions)]
      expected_reward = expected_rewards[action]
      best_expected_reward = np.max(expected_rewards)
      reward = expected_reward + env.reward_noise(action)

      # Update score (sum of estimated rewards)
      episode_score += (expected_reward - best_expected_reward)

      # Update linear model
      rollout_linear_model_results = update_linear_model_at_action(action, rollout_linear_model_results, context,
                                                                   reward)
    it += 1
    score += (episode_score - score) / it
  return score


def rollout(tuning_function_parameter, policy, linear_model_results, time_horizon, current_time, estimated_context_mean,
            tuning_function, estimated_context_variance, env, context_sequences):

  MAX_ITER = 100
  it = 0

  number_of_actions = len(estimated_context_mean)
  context_dimension = len(estimated_context_mean)

  score = 0

  for context_sequence in context_sequences:
    # Sample from distributions that we'll use to determine ''true'' context and reward dbns in rollout
    # working_context_mean = np.random.multivariate_normal(estimated_context_mean, estimated_context_variance)
    working_context_mean = estimated_context_mean
    beta_hat = np.hstack(linear_model_results['beta_hat_list'])
    estimated_beta_hat_variance = block_diag(linear_model_results['sample_cov_list'][0],
                                             linear_model_results['sample_cov_list'][1])
    # working_beta = np.random.multivariate_normal(beta_hat, estimated_beta_hat_variance) + \
    #   np.random.multivariate_normal(mean=np.zeros(len(beta_hat)), cov=100.0*np.eye(len(beta_hat)))
    # working_beta = beta_hat
    working_beta = np.random.multivariate_normal(beta_hat, estimated_beta_hat_variance)
    working_beta = working_beta.reshape((number_of_actions, context_dimension))
    working_sigma_hats = linear_model_results['sigma_hat_list']

    # Rollout under drawn working model
    rollout_linear_model_results = copy.deepcopy(linear_model_results)
    episode_score = 0
    for time in range(current_time + 1, time_horizon):

      beta_hat = np.hstack(rollout_linear_model_results['beta_hat_list']).reshape((number_of_actions,
                                                                                   context_dimension))
      sampling_cov_list = linear_model_results['sample_cov_list']

     # Draw context and take action
      context = np.random.multivariate_normal(working_context_mean, cov=estimated_context_variance)
      action = policy(beta_hat, sampling_cov_list, context, tuning_function, tuning_function_parameter, time_horizon,
                      time)

      # Get epsilon-greedy action and get resulting reward
      # predicted_rewards = np.dot(beta_hat, context)
      # true_rewards = np.dot(working_beta, context)

      # Get reward from pulled arm
      working_beta_at_action = working_beta[action, :]
      working_sigma_hat_at_action = working_sigma_hats[action]
      expected_reward = np.dot(working_beta_at_action, context)
      reward = expected_reward + np.random.normal(scale=np.sqrt(working_sigma_hat_at_action))

      # Update score (sum of estimated rewards)
      episode_score += expected_reward

      # Update linear model
      rollout_linear_model_results = update_linear_model_at_action(action, rollout_linear_model_results, context,
                                                                   reward)
    it += 1
    score += (episode_score - score) / it
  return score


def mHealth_rollout(tuning_function_parameter, policy, linear_model_results, time_horizon, current_time,
                    estimated_context_mean, tuning_function, estimated_context_variance, env, nPatients,
                    context_sequences):

  MAX_ITER = 100
  it = 0

  number_of_actions = len(estimated_context_mean)
  context_dimension = len(estimated_context_mean)

  score = 0

  for context_sequence in context_sequences:
    # Sample from distributions that we'll use to determine ''true'' context and reward dbns in rollout
    # working_context_mean = np.random.multivariate_normal(estimated_context_mean, estimated_context_variance)
    beta_hat = np.hstack(linear_model_results['beta_hat_list'])
    estimated_beta_hat_variance = block_diag(linear_model_results['sample_cov_list'][0],
                                             linear_model_results['sample_cov_list'][1])
    # working_beta = np.random.multivariate_normal(beta_hat, estimated_beta_hat_variance) + \
    #   np.random.multivariate_normal(mean=np.zeros(len(beta_hat)), cov=100.0*np.eye(len(beta_hat)))
    # working_beta = beta_hat
    working_beta = np.random.multivariate_normal(beta_hat, estimated_beta_hat_variance)
    working_beta = working_beta.reshape((number_of_actions, context_dimension))
    working_sigma_hats = linear_model_results['sigma_hat_list']

    # Rollout under drawn working model
    rollout_linear_model_results = copy.deepcopy(linear_model_results)
    episode_score = 0
    for time in range(current_time, time_horizon):
      beta_hat = np.hstack(rollout_linear_model_results['beta_hat_list']).reshape((number_of_actions,
                                                                                   context_dimension))
      sampling_cov_list = linear_model_results['sample_cov_list']
      for j in range(nPatients):
       # Draw context and take action
        context = context_sequence[time - current_time][j]
        action = policy(beta_hat, sampling_cov_list, context, tuning_function, tuning_function_parameter, time_horizon,
                        time)
  
        # Get epsilon-greedy action and get resulting reward
        # predicted_rewards = np.dot(beta_hat, context)
        # true_rewards = np.dot(working_beta, context)
  
        # Get reward from pulled arm
        working_beta_at_action = working_beta[action, :]
        working_sigma_hat_at_action = working_sigma_hats[action]
        expected_reward = np.dot(working_beta_at_action, context)
        reward = expected_reward + np.random.normal(scale=np.sqrt(working_sigma_hat_at_action))
  
        # Update score (sum of estimated rewards)
        episode_score += expected_reward
  
        # Update linear model
        rollout_linear_model_results = update_linear_model_at_action(action, rollout_linear_model_results, context,
                                                                     reward)
    it += 1
    score += (episode_score - score) / it
  return score


# For truncation functions
def expit_truncate(T, t, zeta):
  shrinkage = expit(zeta[0] + zeta[1] * (T - t))
  return shrinkage


def expit_epsilon_decay(T, t, zeta):
  return zeta[0] * expit(zeta[1] + zeta[2]*(T - t))





