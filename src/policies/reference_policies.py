import numpy as np


def linear_cb_best_policy(beta_hat, sampling_cov_list, context, tuning_function, tuning_function_parameter,
                           T, t, env):
  """
  Choose worst arm each time (For reference).

  :param beta_hat:
  :param sampling_cov_list:
  :param context:
  :param tuning_function:
  :param tuning_function_parameter:
  :param T:
  :param t:
  :param env:
  :return:
  """
  expected_rewards = np.array([env.expected_reward(a, context) for a in range(env.number_of_actions)])
  return np.argmax(expected_rewards)


def linear_cb_worst_policy(beta_hat, sampling_cov_list, context, tuning_function, tuning_function_parameter,
                           T, t, env):
  """
  Choose worst arm each time (For reference).

  :param beta_hat:
  :param sampling_cov_list:
  :param context:
  :param tuning_function:
  :param tuning_function_parameter:
  :param T:
  :param t:
  :param env:
  :return:
  """
  expected_rewards = np.array([env.expected_reward(a, context) for a in range(env.number_of_actions)])
  return np.argmin(expected_rewards)