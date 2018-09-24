"""
Functions for simulating rollouts under estimated model for various bandits.
"""
import sys
import pdb
import os
this_dir = os.path.dirname(os.path.abspath(__file__))
project_dir = os.path.join(this_dir, '..', '..')
sys.path.append(project_dir)

import pdb
import numpy as np
import copy
from scipy.linalg import block_diag
from src.environments.Bandit import NormalCB
import src.policies.tuned_bandit_policies as tuned_bandit


def normal_cb_oracle_rollout(tuning_function_parameter, policy, linear_model_results, time_horizon, current_time,
                             estimated_context_mean, tuning_function, estimated_context_variance, env,
                             context_sequences):
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
      rollout_linear_model_results = tuned_bandit.update_linear_model_at_action(action, rollout_linear_model_results, context,
                                                                                reward)
    it += 1
    score += (episode_score - score) / it
  return score


def normal_cb_rollout(tuning_function_parameter, policy, linear_model_results, time_horizon, current_time,
                      estimated_context_mean, tuning_function, estimated_context_variance, env, context_sequences):

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
      rollout_linear_model_results = tuned_bandit.update_linear_model_at_action(action, rollout_linear_model_results, context,
                                                                                reward)
    it += 1
    score += (episode_score - score) / it
  return score


def mHealth_rollout(tuning_function_parameter, policy, time_horizon, current_time, estimated_context_mean,
                    tuning_function, estimated_context_variance, env, nPatients, monte_carlo_reps):

  score = 0
  rollout_env = NormalCB(list_of_reward_betas=env.beta_hat_list, list_of_reward_vars=env.sigma_hat_list,
                         context_mean=estimated_context_mean, context_var=estimated_context_variance)
  for rep in range(monte_carlo_reps):
    rollout_env.reset()
    episode_score = 0
    for time in range(time_horizon):
      beta_hat = rollout_env.beta_hat_list
      sampling_cov_list = rollout_env.sampling_cov_list
      for j in range(nPatients):
        # Draw context and take action
        # context = context_sequence[time - current_time][j]
        action = policy(beta_hat, sampling_cov_list, rollout_env.curr_context, tuning_function,
                        tuning_function_parameter, time_horizon, time)
        expected_reward = rollout_env.expected_reward(action, rollout_env.curr_context)
        optimal_expected_reward = np.max([rollout_env.expected_reward(a, rollout_env.curr_context)
                                          for a in range(rollout_env.number_of_actions)])
        rollout_env.step(action)

        # Update regret
        regret = (expected_reward - optimal_expected_reward)
        episode_score += regret

    print(rep)
    score += (episode_score - score) / (rep + 1)
  return score


def normal_mab_rollout(tuning_function_parameter, policy, time_horizon, current_time, tuning_function, env,
                       nPatients, monte_carlo_reps):
  score = 0

  for it in range(monte_carlo_reps):
    working_means = np.random.normal(loc=env.estimated_means, scale=env.standard_errors)
    working_variances = env.estimated_vars

    # Get initial estimates, which will be updated throughout rollout
    number_of_actions = env.number_of_actions
    number_of_pulls = np.zeros(number_of_actions)
    estimated_means = np.zeros(number_of_actions)
    standard_errors = np.zeros(number_of_actions)
    draws_from_each_arm = [np.array([])]*number_of_actions

    # Rollout under drawn working model
    episode_score = 0
    for time in range(time_horizon):
      for j in range(nPatients):
        action = policy(estimated_means, standard_errors, tuning_function, tuning_function_parameter,
                        time_horizon, time)

        # Get reward from pulled arm
        expected_reward = working_means[action]
        reward = expected_reward + np.random.normal(scale=np.sqrt(working_variances[action]))

        # Update score (sum of estimated rewards)
        episode_score += expected_reward

        # Update model
        number_of_pulls[action] += 1
        estimated_means[action] += (reward - estimated_means[action]) / number_of_pulls[action]
        draws_from_each_arm[action] = np.append(draws_from_each_arm[action], reward)
        n_a = number_of_pulls[action]
        standard_errors[action] = np.sum((draws_from_each_arm[action] - estimated_means[action]) ** 2) / n_a ** 2
    score += (episode_score - score) / (it + 1)
  return score