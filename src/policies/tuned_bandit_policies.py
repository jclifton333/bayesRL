"""
Policies in which exploration/exploitation tradeoff is parameterized and tuned (TS, UCB, ..?).
"""

from scipy.stats import norm
from scipy.linalg import block_diag
from scipy.special import expit
import pdb
import numpy as np
import copy


def normal_ts_policy_probabilities(action, mu_0, sigma_sq_0, mu_1, sigma_sq_1):
  """
  Get probability of actions 0 and 1 at given context under truncated TS policy for 2-armed normal CB.

  :param action: 0 or 1.
  :param mu_0: context . beta_hat_0
  :param sigma_sq_0: context. sample_cov_0 .context * truncation_function
  :param mu_1:
  :param sigma_sq_1:
  :return:
  """
  action_0_prob = 1 - norm.cdf((mu_1 - mu_0) / (sigma_sq_0 + sigma_sq_1))
  return (1 - action)*action_0_prob + action*(1 - action_0_prob)


def normal_ts_policy_gradient(action, context, sample_cov_hat_0, beta_hat_0, sample_cov_hat_1, beta_hat_1,
                              true_beta_0, true_beta_1, truncation_function, truncation_function_derivative, T, t,
                              zeta):
  """

  Policy gradient for a two-armed normal contextual bandit where policy is (soft) truncated TS.
  :return:
  """
  truncation_function_value = truncation_function(T, t, zeta)

  # Policy prob is prob one normal dbn is bigger than another
  mu_0 = np.dot(context, beta_hat_0)
  mu_1 = np.dot(context, beta_hat_1)
  sigma_sq_0 = truncation_function_value * np.dot(context, np.dot(sample_cov_hat_0, context))
  sigma_sq_1 = truncation_function_value * np.dot(context, np.dot(sample_cov_hat_1, context))

  if action:
    diff = (mu_0 - mu_1) / np.max((sigma_sq_0 + sigma_sq_1, 0.01))  # For stability
  else:
    diff = (mu_1 - mu_0) / np.max((sigma_sq_0 + sigma_sq_1, 0.01))

  # Chain rule on 1 - Phi(diff)
  phi_diff = norm.pdf(diff)
  sigma_sq_sum_grad = np.power(sigma_sq_1 + sigma_sq_0, -1) * (sigma_sq_0 + sigma_sq_1) * \
                               truncation_function_derivative(T, t, zeta) / truncation_function_value
  true_expected_reward_0 = np.dot(context, true_beta_0)
  true_expected_reward_1 = np.dot(context, true_beta_1)
  gradient = phi_diff * sigma_sq_sum_grad * (action*true_expected_reward_1 + (1 - action)*true_expected_reward_0)
  return gradient


def sherman_woodbury(A_inv, u, v):
  outer = np.outer(u, v)
  num = np.dot(np.dot(A_inv, outer), A_inv)
  denom = 1.0 + np.dot(np.dot(v, A_inv), u)
  return A_inv - num / denom


def update_linear_model(X, Xprime_X_inv, x_new, X_dot_y, y_new):
  # Compute new beta hat and associated matrices
  Xprime_X_inv_new = sherman_woodbury(Xprime_X_inv, x_new, x_new)
  X_new = np.vstack((X, x_new))
  X_dot_y_new = X_dot_y + y_new * x_new
  beta_hat_new = np.dot(Xprime_X_inv_new, X_dot_y_new)

  # Compute new sample covariance
  n, p = X_new.shape
  yhat = np.dot(X_new, beta_hat_new)
  sigma_hat = np.sum((yhat - y_new)**2) / (n - p)
  sample_cov = sigma_hat * Xprime_X_inv_new

  return {'beta_hat': beta_hat_new, 'Xprime_X_inv': Xprime_X_inv_new, 'X': X_new, 'X_dot_y': X_dot_y_new,
          'sample_cov': sample_cov, 'sigma_hat': sigma_hat}


def add_linear_model_results_at_action_to_dictionary(a, linear_model_results, linear_model_results_for_action):
  linear_model_results['beta_hat_list'][a] = linear_model_results_for_action['beta_hat']
  linear_model_results['Xprime_X_inv_list'][a] = linear_model_results_for_action['Xprime_X_inv']
  linear_model_results['X_list'][a] = linear_model_results_for_action['X']
  linear_model_results['X_dot_y_list'][a] = linear_model_results_for_action['X_dot_y']
  linear_model_results['sample_cov_list'][a] = linear_model_results_for_action['sample_cov']
  linear_model_results['sigma_hat_list'][a] = linear_model_results_for_action['sigma_hat']
  return linear_model_results


def update_linear_model_at_action(a, linear_model_results, x_new, y_new):
  """
  Linear model results is a dictionary of lists.
  Each list is a list of corresponding matrices/arrays of observations, indexed by actions a=0,...,nA.

  :param a:
  :param linear_model_results:
  :param x_new
  :param y_new:
  :returnpdb.set_trace()
  """
  X_a = linear_model_results['X_list'][a]
  Xprime_X_inv_a = linear_model_results['Xprime_X_inv_list'][a]
  X_dot_y_a = linear_model_results['X_dot_y_list'][a]
  updated_linear_model_results_for_action = update_linear_model(X_a, Xprime_X_inv_a, x_new, X_dot_y_a, y_new)
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
    # estimated_beta_hat_variance = block_diag(linear_model_results['sample_cov_list'][0],
    #                                          linear_model_results['sample_cov_list'][1])
    # working_beta = np.random.multivariate_normal(beta_hat, estimated_beta_hat_variance)
    # working_beta = working_beta.reshape((number_of_actions, context_dimension))
    working_beta = beta_hat.reshape((number_of_actions, context_dimension))
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


def tune_epsilon_greedy(linear_model_results, time_horizon, current_time, estimated_context_mean,
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
  policy_gradient = 0

  new_zeta = initial_zeta
  number_of_actions = len(estimated_context_mean)
  context_dimension = len(estimated_context_mean)
#  zeta_dimension = len(new_zeta)
#  policy_gradient = np.zeros(zeta_dimension)
  diff = float('inf')

  while it < MAX_ITER and diff > TOL:
    zeta = new_zeta

    # Sample from distributions that we'll use to determine ''true'' context and reward dbns in rollout
    working_context_mean = np.random.multivariate_normal(estimated_context_mean, estimated_context_variance)
    beta_hat = np.hstack(linear_model_results['beta_hat_list'])
    # estimated_beta_hat_variance = block_diag(linear_model_results['sample_cov_list'][0],
    #                                          linear_model_results['sample_cov_list'][1])
    # working_beta = np.random.multivariate_normal(beta_hat, estimated_beta_hat_variance)
    # working_beta = working_beta.reshape((number_of_actions, context_dimension))
    working_beta = beta_hat.reshape((number_of_actions, context_dimension))
    working_sigma_hats = linear_model_results['sigma_hat_list']
    rollout_linear_model_results = copy.copy(linear_model_results)
    for time in range(current_time + 1, time_horizon):
      # Draw beta
#      shrinkage = truncation_function(time_horizon, time, zeta)
      shrinkage = 1
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
      beta_hat_0 = rollout_linear_model_results['beta_hat_list'][0]
      beta_hat_1 = rollout_linear_model_results['beta_hat_list'][1]
      
      epsilon_function = (1-1/(1+np.exp(-time*zeta)))*2*0.05      
      episilon_gradient = -time*np.exp(-time*zeta)/(1+np.exp(-time*zeta))**2*2*0.05
      action_optimal = np.argmax(np.array([np.dot(beta_hat_0, context), np.dot(beta_hat_1, context)]))
      if action_optimal == 0:
        policy_gradient += np.dot(working_beta[0, :], context)*(-1/2)*episilon_gradient
        policy_gradient += np.dot(working_beta[1, :], context)*(1/2)*episilon_gradient
      else:
        policy_gradient += np.dot(working_beta[0, :], context)*(1/2)*episilon_gradient
        policy_gradient += np.dot(working_beta[1, :], context)*(-1/2)*episilon_gradient

      # Update linear model
      rollout_linear_model_results = update_linear_model_at_action(action, rollout_linear_model_results, context,
                                                                   reward)
    # Update zeta
    step_size = 1e-3 / (it + 1)
    new_zeta = zeta + step_size * policy_gradient
    diff = np.linalg.norm(new_zeta - zeta) / np.linalg.norm(zeta)
    # print("zeta: {}".format(zeta))

    it += 1

  return zeta


# For truncation functions
def expit_truncate(T, t, zeta):
  shrinkage = expit(zeta[0] + zeta[1] * (T - t))
  return shrinkage


def expit_truncate_gradient(T, t, zeta):
  exp_argument = zeta[0] + zeta[1] * (T - t)
  exp_ = np.exp(exp_argument)
  return exp_ / np.power(1 + exp_, 2) * zeta
