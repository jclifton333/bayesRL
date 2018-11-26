"""
Try to get globally optimal exploration schedule for bernoulli bandit using dynamic programming.
"""
import numpy as np


def bb_dp(prob0, prob1, time_horizon):
  """

  :param prob0: reward prob arm 0
  :param prob1: reward prob arm 1
  :param time_horizon:
  :return:
  """
  T = time_horizon - 2

  # Populate list of policies
  policy = []
  values = []
  for t in range(1, time_horizon + 1):
    policy_at_t = np.zeros((t + 1,  t + 1, t + 1))
    policy.append(policy_at_t)
    value_at_t = np.zeros((t + 1, t + 1, t + 1))
    values.append(value_at_t)

  # Possible states at time T are given by { (n, s_0, s_1) : n <= T, s_0 <= n, s_1 <= T - n
  # where n = number of pulls of arm0; s_0 = number of successes of arm 0; s_1 = number of successes of arm 1
  for n in range(1, T + 1):
    for s0 in range(1, n + 1):
      for s1 in range(1, T - n + 1):
        for eps in np.linspace(0.0, 1.0, 101):  # Epsilon grid 0.0, 0.01, 0.02, ... 0.99, 1.0
          values_at_tp1 = values[T + 1]
          if (s0 / n) > (s1 / (T - n)):
            q_0 = (1 - eps) * (prob0 * values_at_tp1[n + 1, s0 + 1, s1] + (1 - prob0) * values_at_tp1[n + 1, s0, s1]) + \
              eps * (prob1 * values_at_tp1[n + 1, s0, s1 + 1] + (1 - prob1) * values_at_tp1[n + 1, s0, s1])
          else:
            pass




