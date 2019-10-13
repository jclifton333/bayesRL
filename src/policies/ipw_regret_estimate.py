"""
IPW estimators of in-sample plus out-of-sample regret for epsilon-greedy in 2-armed bandit.
"""

import pdb
import numpy as np
from scipy.stats import norm
import copy


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
    pi_sum, pi_inv_sum, m_pi_inv_sum = 0.5, 2, 2
  else:
    pi_sum, pi_inv_sum, m_pi_inv_sum = propensity_sums
  propensity_denominator = (pi_inv_sum + m_pi_inv_sum) / (t + 1)**2
  e_ = norm.cdf(-Delta / np.sqrt(propensity_denominator))
  for t in range(t, in_sample_size):
    pi_t = (1 - epsilon) * e_ + epsilon / 2
    pi_sum += pi_t
    pi_inv_sum += 1 / pi_t
    m_pi_inv_sum += 1 / (1 - pi_t)
    propensity_denominator = (pi_inv_sum + m_pi_inv_sum) / (t+1)**2
    e_ = norm.cdf(-Delta / np.sqrt(propensity_denominator))
  return pi_sum, pi_inv_sum, m_pi_inv_sum


def ipw_regret_estimate(in_sample_size, out_of_sample_size, Delta, propensity_estimate, pi_inv_sum,
                        m_pi_inv_sum):
  """

  :param in_sample_size:
  :param out_of_sample_size:
  :param Delta:
  :param epsilon:
  :param propensity_estimate:
  :return:
  """
  in_sample_regret = Delta * propensity_estimate
  out_of_sample_denominator = (pi_inv_sum + m_pi_inv_sum) / in_sample_size**2
  out_of_sample_regret = out_of_sample_size * Delta * norm.cdf(-Delta / np.sqrt(out_of_sample_denominator))
  estimated_regret = in_sample_regret + out_of_sample_regret
  return estimated_regret


def max_ipw_regret(in_sample_size, out_of_sample_size, epsilon, min_range, max_range, propensity_sums, t):
  grid = np.linspace(min_range, max_range, 20)  # ToDo: Compute grid once elsewhere?
  regrets = []
  for Delta_ in grid:
    propensity_estimate_, pi_inv_sum_, m_pi_inv_sum_ \
      = propensity_estimator(epsilon, in_sample_size, Delta_, t, propensity_sums)
    Delta_regret = ipw_regret_estimate(in_sample_size, out_of_sample_size, Delta_, propensity_estimate_,
                                       pi_inv_sum_, m_pi_inv_sum_)
    regrets.append(Delta_regret)
  return np.max(regrets)


def minimax_epsilon(in_sample_size, out_of_sample_size, min_range, max_range, propensity_sums, t):
  epsilon_grid = np.linspace(0.1, 1.0, 20)
  best_eps = None
  best_regret = float('inf')
  estimated_max_regret_list = []
  for eps in epsilon_grid:
    eps_regret = max_ipw_regret(in_sample_size, out_of_sample_size, eps, min_range, max_range, propensity_sums, t)
    estimated_max_regret_list.append(eps_regret)
    if eps_regret < best_regret:
      best_eps = eps
      best_regret = eps_regret
  return best_eps







