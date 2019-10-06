"""
IPW estimators of in-sample plus out-of-sample regret for epsilon-greedy in 2-armed bandit.
"""

import numpy as np
from scipy.stats import norm


def ipw_regret_estimate(in_sample_size, out_of_sample_size, Delta, propensity_estimate):
  """

  :param in_sample_size:
  :param out_of_sample_size:
  :param Delta:
  :param epsilon:
  :param propensity_estimate:
  :return:
  """
  in_sample_regret = Delta * in_sample_size * propensity_estimate
  out_of_sample_denominator = (1/propensity_estimate + 1/(1-propensity_estimate)) / in_sample_size
  out_of_sample_regret = out_of_sample_size * Delta * norm.cdf(-Delta / np.sqrt(out_of_sample_denominator))
  estimated_regret = in_sample_regret + out_of_sample_regret
  return estimated_regret


def max_ipw_regret(in_sample_size, out_of_sample_size, epsilon, min_range, max_range):
  propensity_estimate_ = propensity_estimator(epsilon)
  grid = np.linspace(min_range, max_range, 10)  # ToDo: Compute grid once elsewhere?
  regrets = [ipw_regret_estimate(in_sample_size, out_of_sample_size, Delta_, propensity_estimate_) for Delta_ in grid]
  return np.max(regrets)

