from scipy.stats import norm
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
                              truncation_function, truncation_function_derivative, T, t, zeta):
  """

  Policy gradient for a two-armed normal contextual bandit where policy is (soft) truncated TS.
  :return:
  """
  truncation_function_value = truncation_function(T - t, zeta)

  # Policy prob is prob one normal dbn is bigger than another
  mu_0 = np.dot(context, beta_hat_0)
  mu_1 = np.dot(context, beta_hat_1)
  sigma_sq_0 = truncation_function_value * np.dot(context, np.dot(sample_cov_hat_0, context))
  sigma_sq_1 = truncation_function_value * np.dot(context, np.dot(sample_cov_hat_1, context))

  if action:
    diff = (mu_0 - mu_1) / (sigma_sq_0 + sigma_sq_1)
  else:
    diff = (mu_1 - mu_0) / (sigma_sq_0 + sigma_sq_1)

  # Chain rule on 1 - Phi(diff)
  phi_diff = norm.pdf(diff)
  sigma_sq_sum_grad = np.power(sigma_sq_1 + sigma_sq_0, -1) * (sigma_sq_0 + sigma_sq_1) * \
                               truncation_function_derivative(T-t, zeta) / truncation_function_value
  gradient = phi_diff * sigma_sq_sum_grad * (action*mu_1 + (1 - action)*mu_0)

  return gradient


def sherman_woodbury(A_inv, u, v):
  outer = np.outer(u, v)
  num = np.dot(np.dot(A_inv, outer), A_inv)
  denom = 1.0 + np.dot(np.dot(v, A_inv), u)
  return A_inv - num / denom


def update_linear_model(X, Xprime_X_inv, x_new, X_dot_y, y_new):
  # Compute new beta hat and associated matrices
  Xprime_X_inv_new = sherman_woodbury(Xprime_X_inv, x_new, x_new)
  X_new = np.vstack((X, x))
  X_dot_y_new = X_dot_y + y_new * x_new
  beta_hat_new = np.dot(Xprime_X_inv_new, X_dot_y_new)

  # Compute new sample covariance
  n, p = X_new.shape
  yhat = np.dot(X_new, beta_hat_new)
  sigma_hat = np.sum((yhat - y_new)**2) / (n - p)
  sample_cov = sigma_hat * np.dot(X_new, np.dot(Xprime_X_inv_new, X_new.T))

  return {'beta_hat': beta_hat_new, 'Xprime_X_inv': Xprime_X_inv_new, 'X': X_new, 'X_dot_y': X_dot_y_new,
          'sample_cov': sample_cov}


def add_linear_model_results_at_action_to_dictionary(a, linear_model_results, linear_model_results_for_action):
  linear_model_results['beta_hat_list'][a] = linear_model_results_for_action['beta_hat']
  linear_model_results['Xprime_X_inv_list'][a] = linear_model_results_for_action['Xprime_X_inv']
  linear_model_results['X_list'][a] = linear_model_results_for_action['X']
  linear_model_results['X_dot_y_list'][a] = linear_model_results_for_action['X_dot_y']
  linear_model_results['sample_cov_list'][a] = linear_model_results_for_action['sample_cov']
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
  X_a = X_list[a]
  Xprime_X_inv_a = Xprime_X_inv_list[a]
  X_dot_y_a = X_dot_y_list[a]
  updated_linear_model_results_for_action = update_linear_model(X_a, Xprime_X_inv_a, x_new, X_dot_y_a, y_new)
  linear_model_results = add_linear_model_results_at_action_to_dictionary(a, linear_model_results,
                                                                          updated_linear_model_results_for_action)
  return linear_model_results


def truncated_thompson_sampling(env, linear_model_results, time_horizon, current_time, estimated_context_mean,
                                estimated_context_variance, truncation_function, truncation_function_gradient,
                                initial_zeta):
  """

  :param env:
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
  it = 0

  zeta = initial_zeta
  number_of_actions = len(estimated_context_mean)
  context_dimension = len(estimated_context_mean)
  zeta_dimension = len(zeta)
  policy_gradient = np.zeros(zeta_dimension)

  while it < MAX_ITER:
    working_context_mean = np.random.multivariate_normal(estimated_context_mean, estimated_context_variance)
    rollout_linear_model_results = copy.copy(linear_model_results)

    for time in range(current_time + 1, time_horizon):
      # Draw beta
      shrinkage = truncation_function(time_horizon - time, zeta)

      beta_hat = rollout_linear_model_results['beta_hat_list'].flatten()
      estimated_beta_hat_variance = np.vstack(rollout_linear_model_results['sample_cov_list'])

      beta = np.random.multivariate_normal(beta_hat, shrinkage * estimated_beta_hat_variance)
      beta = beta.reshape((number_of_actions, context_dimension))

      # Draw context
      context = np.random.multivariate_normal(working_context_mean, cov=np.eye(context_dimension))

      # Get action from predicted_rewards
      predicted_rewards = np.dot(beta, context)
      action = np.argmax(predicted_rewards)
      reward = env.step(action)

      # Update policy gradient
      sample_cov_hat_0 = rollout_linear_model_results['sample_cov_list'][0]
      sample_cov_hat_1 = rollout_linear_model_results['sample_cov_list'][1]
      beta_hat_0 = rollout_linear_model_results['beta_hat_list'][0]
      beta_hat_1 = rollout_linear_model_results['beta_hat_list'][1]

      policy_gradient += normal_ts_policy_gradient(0, context, sample_cov_hat_0, beta_hat_0, sample_cov_hat_1,
                                                   beta_hat_1, truncation_function, truncation_function_gradient, T,
                                                   t, zeta)
      policy_gradient += normal_ts_policy_gradient(1, context, sample_cov_hat_0, beta_hat_0, sample_cov_hat_1,
                                                   beta_hat_1, truncation_function, truncation_function_gradient, T,
                                                   t, zeta)

      # Update linear model
      rollout_linear_model_results = update_linear_model_at_action(action, rollout_linear_model_results, context,
                                                                   reward)
    # Update zeta
    step_size = 1.0 / (it + 1)
    zeta += step_size * policy_gradient
    it += 1

  return zeta



