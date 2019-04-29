"""
Try to get globally optimal exploration schedule for bernoulli bandit using dynamic programming.
"""
import numpy as np
import pdb


def initialize_policies_and_value_function(time_horizon):
  # Populate list of policies
  policy = []
  values = []
  for t in range(1, time_horizon + 1):
    policy_at_t = np.zeros((t + 1,  t + 1, t + 1))
    policy.append(policy_at_t)
    value_at_t = np.zeros((t + 1, t + 1, t + 1))
    values.append(value_at_t)
  return policy, values


def backup(prob0, prob1, policy, values, T):
  """
  Perform one dynamic programming backup

  :param prob0: reward prob arm 0
  :param prob1: reward prob arm 1
  :param time_horizon:
  :return:
  """

  # Possible states at time T are given by { (n, s_0, s_1) : n <= T, s_0 <= n, s_1 <= T - n
  # where n = number of pulls of arm0; s_0 = number of successes of arm 0; s_1 = number of successes of arm 1
  for n in range(1, T + 1):  # Starting at 1 since assume each arm has been pulled once before starting
    for s0 in range(1, n + 1):
      for s1 in range(1, T - n + 1):
        value = -float('inf')
        best_eps = None
        for eps in np.linspace(0.0, 1.0, 101):  # Epsilon grid 0.0, 0.01, 0.02, ... 0.99, 1.0
          values_at_tp1 = values[T]
          if (s0 / n) > (s1 / (T - n)):
            q = (1 - eps) * (prob0 * (1 + values_at_tp1[n + 1, s0 + 1, s1]) +
                             (1 - prob0) * (0 + values_at_tp1[n + 1, s0, s1]) ) + \
                eps * (prob1 * (1 + values_at_tp1[n + 1, s0, s1 + 1]) +
                       (1 - prob1) * (0 + values_at_tp1[n + 1, s0, s1]))
          elif (s0 / n) > (s1 / (T - n)):
            q = eps * (prob0 * (1 + values_at_tp1[n + 1, s0 + 1, s1]) +
                       (1 - prob0) * (0 + values_at_tp1[n + 1, s0, s1])) + \
                (1 - eps) * (prob1 * (1 + values_at_tp1[n + 1, s0, s1 + 1]) +
                             (1 - prob1) * (0 + values_at_tp1[n + 1, s0, s1]))
          elif (s0 / n) == (s1 / (T - n)):
            q = 0.5 * (prob0 * (1 + values_at_tp1[n + 1, s0 + 1, s1]) +
                       (1 - prob0) * (0 + values_at_tp1[n + 1, s0, s1])) + \
                0.5 * (prob1 * (1 + values_at_tp1[n + 1, s0, s1 + 1]) +
                       (1 - prob1) * (0 + values_at_tp1[n + 1, s0, s1]))
          if q > value:
            value = q
            best_eps = eps
        policy[T - 1][n-1, s0-1, s1-1] = best_eps
        values[T - 1][n-1, s0-1, s1-1] = value
  return policy, values


def initial_step(prob0, prob1, values, time_horizon):
  """
  Get values at last time step, where optimal policy is epsilon = 0.

  :param prob0:
  :param prob1:
  :param values:
  :param time_horizon:
  :return:
  """
  for n in range(1, time_horizon + 1):
    for s0 in range(1, n + 1):
      for s1 in range(1, time_horizon - n + 1):
        if (s0 / n) > (s1 / (time_horizon - n)):
          values[-1][n-1, s0-1, s1-1] = prob0
        else:
          values[-1][n-1, s0-1, s1-1] = prob1
  return values


def bb_dp(prob0, prob1, time_horizon):
  policy, values = initialize_policies_and_value_function(time_horizon)
  values = initial_step(prob0, prob1, values, time_horizon)
  for t in range(1, time_horizon):
    policy, values = backup(prob0, prob1, policy, values, time_horizon - t)
  return policy, values


if __name__ == "__main__":
  policy, values = bb_dp(0.3, 0.8, 10)
