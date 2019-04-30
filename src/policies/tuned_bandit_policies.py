"""
Policies in which exploration/exploitation tradeoff is parameterized and tuned (TS, UCB, ..?).
"""
import sys
import pdb
import os
this_dir = os.path.dirname(os.path.abspath(__file__))
project_dir = os.path.join(this_dir, '..', '..')
sys.path.append(project_dir)

import scipy.integrate as integrate
from scipy.stats import norm
from sklearn.ensemble import RandomForestRegressor
from scipy.linalg import block_diag
from scipy.special import expit
import scipy.stats
import src.policies.linear_algebra as la
import pdb
import numpy as np
import copy

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)


def linear_cb_two_arm_ts_policy(beta_hat, sampling_cov_list, context, tuning_function, tuning_function_parameter,
                                T, t, env):
  truncation = tuning_function(T, t, tuning_function_parameter)


def linear_cb_epsilon_greedy_policy(beta_hat, sampling_cov_list, context, tuning_function, tuning_function_parameter,
                                    T, t, env):
  epsilon = tuning_function(T, t, tuning_function_parameter)
  predicted_rewards = np.dot(beta_hat, context)
  greedy_action = np.argmax(predicted_rewards)
  if np.random.random() < epsilon:
    action = np.random.choice(2)
  else:
    action = greedy_action
  return action


def linear_cb_thompson_sampling_policy(beta_hat, sampling_cov_list, context, tuning_function, tuning_function_parameter,
                                       T, t, env):
  shrinkage = tuning_function(T, t, tuning_function_parameter)
  shrinkage = np.min((shrinkage, 1.0))
  posterior_draw = env.sample_from_posterior(variance_shrinkage=shrinkage)
  mu_hats = [np.dot(context, posterior_draw[a]['beta_draw']) for a in range(env.number_of_actions)]
  return np.argmax(mu_hats)


def linear_cb_ucb_policy(beta_hat, sampling_cov_list, x, tuning_function, 
                         tuning_function_parameter, T, t, env):
  alpha = tuning_function(T, t, tuning_function_parameter)/40 + 0.5
  z = scipy.stats.norm.ppf(alpha)
  estimated_rewards = np.dot(beta_hat, env.curr_context) 
  kesi = []
  for a in range(len(beta_hat)):
    X_a = env.X_list[a]
    n_a = X_a.shape[0]
    residual = env.y_list[a] - np.dot(X_a, beta_hat[a])
    res_multiply_X = np.multiply(X_a, residual.reshape(-1, 1))
    Sigma = np.matmul(res_multiply_X.T, res_multiply_X)
    omega = np.dot(env.curr_context, env.Xprime_X_inv_list[a])
    bound = np.dot(np.dot(omega, Sigma), omega)/np.sqrt(n_a)
    kesi = np.append(kesi, estimated_rewards[a] + z * bound)
  
  action = np.argmax(kesi)
  
  return action


def mab_epsilon_greedy_policy(estimated_means, standard_errors, number_of_pulls, tuning_function,
                              tuning_function_parameter, T, t, env):
  epsilon = tuning_function(T, t, tuning_function_parameter)
  greedy_action = np.argmax(estimated_means)
  if np.random.random() < epsilon:
    action = np.random.choice(2)
  else:
    action = greedy_action
  return action


def normal_mab_ucb_policy(estimated_means, standard_errors, number_of_pulls, tuning_function,
                      tuning_function_parameter, T, t, env):
  ## alpha (percentile): decrease from a little bit smaller than 1 to 1/2 at T
  ## scale and shift
  one_minus_alpha = 0.5*tuning_function(T, t, tuning_function_parameter) + 0.5
  z = scipy.stats.norm.ppf(one_minus_alpha)
  action = np.argmax(estimated_means + z * standard_errors)
  return action
  

def mab_thompson_sampling_policy(estimated_means, standard_errors, number_of_pulls, tuning_function,
                                 tuning_function_parameter, T, t, env):

  shrinkage = tuning_function(T, t, tuning_function_parameter)
  shrinkage = np.min((shrinkage, 1.0))
  posterior_draw = env.sample_from_posterior(variance_shrinkage=shrinkage)
  return np.argmax([posterior_draw[a]['mu_draw'] for a in range(env.number_of_actions)])


def mab_frequentist_ts_policy(estimated_means, standard_errors, number_of_pulls, tuning_function,
                              tuning_function_parameter, T, t, env):
  shrinkage = tuning_function(T, t, tuning_function_parameter)
  shrinkage = np.min((shrinkage, 1.0))
  sampling_dbn_draws = np.array([np.random.normal(mu, shrinkage * se) for mu, se in zip(estimated_means,
                                                                                        standard_errors)])
  return np.argmax(sampling_dbn_draws)


def glucose_one_step_policy(env, tuning_function, tuning_function_parameter, time_horizon, t, fixed_eps=None,
                            X=None, R=None):
  """
  Assuming epsilon-greedy exploration.

  :param env:
  :param tuning_function:
  :param tuning_function_parameter:
  :param time_horizon:
  :param t:
  :return:
  """
  # Get features and response
  if X is None:
    X, R = env.X, env.R
    X = [X_i[2:, :] for X_i in X]
    R = [R_i[:-1] for R_i in env.R]
  X_flat = np.vstack(X)
  R_flat = np.hstack(R)
  # R_flat_check = np.array([env.reward_function(None, x[1:4]) for x in X_flat])
  # if not np.array_equal(R_flat, R_flat_check):
  #   pdb.set_trace()

  # One-step FQI
  m = RandomForestRegressor()
  # Remove infs and nans
  # ToDo: Figure out why there are infs and nans in the first place! 
  not_nan_or_inf_ixs = np.where(np.isfinite(R_flat) == True)
  m.fit(X_flat[not_nan_or_inf_ixs], R_flat[not_nan_or_inf_ixs])

  # Get argmax of fitted function at current state
  if fixed_eps is None:
    epsilon = tuning_function(time_horizon, t, tuning_function_parameter)
  else:
    epsilon = fixed_eps
  action = np.zeros(0)
  # for X_i in X:
  for i in range(env.nPatients):
    # x_i = X_i[-1, :]
    x_i = np.concatenate(([1], env.current_state[i], env.last_state[i], [env.last_action[i]], [0]))
    if np.random.random() < epsilon:
      action_i = int(np.random.choice(2))
    else:
      # number_of_ties = np.sum(m.predict(env.get_state_at_action(0, x_i).reshape(1, -1)) == m.predict(env.get_state_at_action(1, x_i).reshape(1, -1)))
      # print('number of ties: {}'.format(number_of_ties))
      action_i = np.argmax([m.predict(env.get_state_at_action(0, x_i).reshape(1, -1)),
                            m.predict(env.get_state_at_action(1, x_i).reshape(1, -1))])
    action = np.append(action, action_i)

  return action


def probability_truncated_normal_exceedance(l0, u0, l1, u1, mean0, sigma0, mean1, sigma1):
  """
  Probability one truncated normal (between l0 and u0) exceeds another (between l1 and u1).
  For two-armed thompson sampling.
  :param l0:
  :param u0:
  :param l1:
  :param u1:
  :param mean0:
  :param sigma0:
  :param mean1:
  :param sigma1:
  :return:
  """
  def integrand(x0):
    x1_prob = norm.cdf(u1, loc=mean1, scale=sigma1) - norm.cdf(np.max((x0, l1)), loc=mean1, scale=sigma1)
    x0_dens = norm.pdf(x0, loc=mean0, scale=sigma0)
    return x0_dens * x1_prob

  numerator_prob = integrate.quad(integrand, l0, u0)[0]
  denominator_prob = (norm.cdf(u1, loc=mean1, scale=sigma1) - norm.cdf(l1, loc=mean1, scale=sigma1)) * \
    (norm.cdf(u0, loc=mean0, scale=sigma0) - norm.cdf(l0, loc=mean0, scale=sigma0))
  return numerator_prob / denominator_prob


# Helpers

def expit_truncate(T, t, zeta):
  shrinkage = expit(zeta[0] + zeta[1] * (T - t))
  return shrinkage


def expit_epsilon_decay(T, t, zeta):
  return zeta[0] * expit(zeta[2]*(T - t - zeta[1]))


def expit_epsilon_decay_info_state(T, t, zeta, **kwargs):
  """
  Use info state, for two-armed multiarmed bandit only.
  :param T: 
  :param t: 
  :return: 
  """
  xbar_diff, sigma_ratio = kwargs['xbar_diff'], kwargs['sigma_ratio']
  return zeta[0] * expit(zeta[1]*(T-t) + zeta[2]*xbar_diff + zeta[3]*sigma_ratio + zeta[4]*(T-t)*xbar_diff
                         + zeta[5]*(T-t)*sigma_ratio - zeta[6])


def step_function(T, t, zeta):
  J = len(zeta)
  interval = int(T/float(J))
  if t == 0:
    j = J - 1
  else:
    j = int(np.floor((T - t) / interval))
  return zeta[j]


def stepwise_linear_epsilon(T, t, zeta):
  J = len(zeta)
  interval = int(T/float(J))
  if t == 0:
    j = J - 1
  else:
    j = int(np.floor((T-t)/interval))
  epsilon = sum(zeta[:j]) + ((T-t) - j*interval) * zeta[j] / interval
  return epsilon






