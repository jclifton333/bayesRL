from scipy.stats import norm
import numpy as np


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


def epsilon_greedy_policy_gradient(linear_model_results, time_horizon, current_time, estimated_context_mean,
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
  TOL = 0.001
  it = 0

  new_zeta = initial_zeta
  number_of_actions = len(estimated_context_mean)
  context_dimension = len(estimated_context_mean)
  zeta_dimension = len(new_zeta)
  diff = float('inf')

  # Sample from distributions that we'll use to determine ''true'' context and reward dbns in rollout
  working_context_mean = np.random.multivariate_normal(estimated_context_mean, estimated_context_variance)
  beta_hat = np.hstack(linear_model_results['beta_hat_list'])
  estimated_beta_hat_variance = block_diag(linear_model_results['sample_cov_list'][0],
                                           linear_model_results['sample_cov_list'][1])
  # working_beta = np.random.multivariate_normal(beta_hat, estimated_beta_hat_variance) + \
  #   np.random.multivariate_normal(mean=np.zeros(len(beta_hat)), cov=100.0*np.eye(len(beta_hat)))
  working_beta = np.random.multivariate_normal(beta_hat, estimated_beta_hat_variance)
  # working_beta = beta_hat
  working_beta = working_beta.reshape((number_of_actions, context_dimension))
  working_sigma_hats = linear_model_results['sigma_hat_list']

  while it < MAX_ITER and diff > TOL:
    zeta = new_zeta
    rollout_linear_model_results = copy.deepcopy(linear_model_results)
    gradients = np.zeros((0, zeta_dimension))
    episode_rewards = []
    episode_policy_gradient = np.zeros(zeta_dimension)
    for time in range(current_time + 1, time_horizon):

      beta_hat = np.hstack(rollout_linear_model_results['beta_hat_list']).reshape((number_of_actions,
                                                                                   context_dimension))

     # Draw context
      context = np.random.multivariate_normal(working_context_mean, cov=estimated_context_variance)

      # Get epsilon-greedy action and get resulting reward
      predicted_rewards = np.dot(beta_hat, context)
      true_rewards = np.dot(working_beta, context)
      # print('correct action {}'.format(np.argmax(predicted_rewards) == np.argmax(true_rewards)))
      # print('pred {} true {}'.format(predicted_rewards, true_rewards))
      # pdb.set_trace()
      greedy_action = np.argmax(predicted_rewards)
      epsilon = expit_epsilon_decay(time_horizon, time, zeta)
      if np.random.random() < epsilon:
        action = np.random.choice(2)
      else:
        action = greedy_action

      working_beta_at_action = working_beta[action, :]
      working_sigma_hat_at_action = working_sigma_hats[action]
      reward = np.dot(working_beta_at_action, context) + np.random.normal(scale=np.sqrt(working_sigma_hat_at_action))

      epsilon_gradient = expit_epsilon_decay_gradient(time_horizon, time, zeta) / 2
      if action == greedy_action:
        epsilon_gradient *= -1
      gradients = np.vstack((gradients, epsilon_gradient))
      episode_rewards.append(reward)

    returns = np.sum(episode_rewards) - np.cumsum(episode_rewards)
    returns_times_gradients = np.multiply(gradients, returns.reshape(-1, 1))
    episode_policy_gradient += np.sum(returns_times_gradients, axis=0)

    # Update zeta
    step_size = 0.001 / (it + 1)
    new_zeta = zeta + step_size * episode_policy_gradient / 10.0
    new_zeta[0] = np.min((1.0, np.max((0.01, new_zeta[0]))))
    diff = np.linalg.norm(new_zeta - zeta) / np.linalg.norm(zeta)
    # print("zeta: {}".format(new_zeta))
    pdb.set_trace()
    it += 1

  return zeta


def expit_epsilon_decay_gradient(T, t, zeta):
  kappa, zeta_0, zeta_1 = zeta

  # Pieces
  exp_ = np.exp(-zeta_0 - zeta_1 * (T - t))
  # one_plus_exp_power = np.power(1 + exp_, -2)

  # # Gradient
  # partial_kappa = 1.0 / (1.0 + exp_)
  # partial_zeta_0 = kappa * one_plus_exp_power * exp_
  # partial_zeta_1 = kappa * one_plus_exp_power * exp_ * (T - t)

  # Gradient of log
  partial_kappa = 1.0 / kappa
  partial_zeta = 1 / (1 + exp_) * exp_ * np.array([1.0, T - t])
  gradient = np.append(partial_kappa, partial_zeta)

  return gradient


def expit_truncate_gradient(T, t, zeta):
  exp_argument = zeta[0] + zeta[1] * (T - t)
  exp_ = np.exp(exp_argument)
  return exp_ / np.power(1 + exp_, 2) * zeta
