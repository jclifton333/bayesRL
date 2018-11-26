"""
Try to get globally optimal exploration schedule for bernoulli bandit using dynamic programming.
"""
import numpy as np


def bb_dp(prob, time_horizon):
  """

  :param prob:
  :param time_horizon:
  :return:
  """
  T = time_horizon - 2

  # Populate list of policies
  policy = []
  for t in range(1, time_horizon + 1):
    policy_at_t = np.zeros((t + 1,  t + 1, t + 1))
    policy.append(policy_at_t)

  # Possible states at time T are given by { (n, s_0, s_1) : n <= T, s_0 <= n, s_1 <= T - n
  # where n = number of pulls of arm0; s_0 = number of successes of arm 0; s_1 = number of successes of arm 1
  for n in range(T):
    for s0 in range(n):
      for s1 in range(T - n):
        for eps in np.linspace(0.0, 1.0, 101):  # Epsilon grid 0.0, 0.01, 0.02, ... 0.99, 1.0
          pass




