"""
IPW estimators of in-sample plus out-of-sample regret for epsilon-greedy in 2-armed bandit.
"""

import numpy as np
from scipy.stats import norm


def propensity_estimator(epsilon, in_sample_size, Delta, t, propensity_sums):
  """

  :param epsilon:
  :param in_sample_size:
  :param Delta:
  ;param t: current timestep
  :param propensity_sums: tuple (sum 1/pi_tilde, sum 1/(1-pi_tilde)) until t-1
  :return:
  """
  if propensity_sums is None:
    pi_inv_sum, m_pi_inv_sum = 2, 2
  else:
    pi_inv_sum, m_pi_inv_sum = propensity_sums
  propensity_denominator = (pi_inv_sum + m_pi_inv_sum) / (t - 1)**2
  e_ = norm.cdf(-Delta / np.sqrt(propensity_denominator))
  for t in range(t, in_sample_size):
    pi_t = (1 - epsilon) * e_ + epsilon / 2
    pi_inv_sum += 1 / pi_t
    m_pi_inv_sum += 1 / (1 - pi_t)
    propensity_denominator = (pi_inv_sum + m_pi_inv_sum) / t**2
    e_ = norm.cdf(-Delta / np.sqrt(propensity_denominator))
  return pi_t


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


def max_ipw_regret(in_sample_size, out_of_sample_size, epsilon, min_range, max_range, propensity_sums):
  propensity_estimate_ = propensity_estimator(epsilon, in_sample_size, Delta, t, propensity_sums)
  grid = np.linspace(min_range, max_range, 20)  # ToDo: Compute grid once elsewhere?
  regrets = [ipw_regret_estimate(in_sample_size, out_of_sample_size, Delta_, propensity_estimate_) for Delta_ in grid]
  return np.max(regrets)


def minimax_epsilon(in_sample_size, out_of_sample_size, min_range, max_range, propensity_sums):
  epsilon_grid = np.linspace(0, 1.0, 20)
  best_eps = None
  best_regret = float('inf')
  for eps in epsilon_grid:
    eps_regret= max_ipw_regret(in_sample_size, out_of_sample_size, eps, min_range, max_range, propensity_sumsk)
    if eps_regret < best_regret:
      best_eps = eps
      best_regret = eps_regret
  return best_eps







