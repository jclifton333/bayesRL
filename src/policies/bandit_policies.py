import numpy as np


def truncated_thompson_sampling(env, time_horizon, current_time, estimated_context_mean, estimated_context_variance,
                                estimated_beta_mean, estimated_beta_hat_variance, truncation_function,
                                truncation_function_gradient, initial_eta):
  """

  :param env:
  :param time_horizon: 
  :param current_time:
  :param estimated_context_mean:
  :param estimated_context_variance:
  :param estimated_beta_mean: 
  :param estimated_beta_hat_variance: Estimated sampling covariance matrix of beta_hat.
  :param truncation_function: Function of (time_horizon-current_time) and parameter zeta (to be optimized), which
                              governs how much to adjust variance of approximate sampling dbn when thompson sampling.
  :param truncation_function_gradient:
  :param initial_eta:
  :return: 
  """
  MAX_ITER = 100
  it = 0
  eta = initial_eta
  number_of_actions = len(estimated_context_mean)
  context_dimension = len(estimated_context_mean)

  while it < MAX_ITER:
    working_context_mean = np.random.multivariate_normal(estimated_context_mean, estimated_context_variance)
    for time in range(current_time + 1, time_horizon):
      # Draw beta
      shrinkage = truncation_function(time_horizon - time, initial_eta)
      beta = np.random.multivariate_normal(estimated_beta_mean, shrinkage * estimated_beta_hat_variance)
      beta = beta.reshape((number_of_actions, context_dimension))

      # Draw context
      context = np.random.multivariate_normal(working_context_mean, cov=np.eye(context_dimension))

      # Get action from predicted_rewards
      predicted_rewards = np.dot(beta, working_context_mean)
      action = np.argmax(predicted_rewards)

      reward = env.step(action)






