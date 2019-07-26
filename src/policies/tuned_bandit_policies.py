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
from scipy.stats import beta
from scipy.linalg import block_diag
from scipy.special import expit
import scipy.stats
import src.policies.linear_algebra as la
import pdb
import numpy as np
from numba import jit
import copy


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
    action = np.random.choice(len(estimated_means))
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

#def bernoulli_mab_ucb_policy(estimated_means, standard_errors, number_of_pulls, 
#                             tuning_function, T, t, tuning_function_parameter, env):
#  # For UCB: get CI from sampling distribution of frequentist estimated p, p_hat
#  # Need to check this part, here env is the real environments, be careful
#  sampling_estimated_means = np.array([sum(env.draws_from_each_arm[a]) for a in range(env.number_of_actions)]).astype(float)/env.number_of_pulls
#  standard_errors = np.sqrt(sampling_estimated_means*(1-sampling_estimated_means)/env.number_of_pulls)
#  one_minus_alpha = 0.5*tuning_function(T, t, tuning_function_parameter) + 0.5
#  z = scipy.stats.norm.ppf(one_minus_alpha)
#  action = np.argmax(sampling_estimated_means + z * standard_errors)
#  return action


def bernoulli_mab_ucb_posterior_policy(estimated_means, standard_errors, number_of_pulls, 
                             tuning_function, tuning_function_parameter, T, t, env):
  # For UCB: get CI from posterior distribution (Beta dbn) 
  # one-sided interval, when tuning_value=0, upper-bound is actually estimated median instaead of estimated mean
  tuning_value = tuning_function(T, t, tuning_function_parameter)
  if tuning_value > 0.9: # such that it falls between [0,1]
    tuning_value = 0.9
  one_minus_alpha = 0.5*tuning_value + 0.5
  upper_bound = []
  for a in range(env.number_of_actions):
    alpha_p = env.posterior_params_dict[a]['alpha_post']
    beta_p = env.posterior_params_dict[a]['beta_post']
    upper_bound = np.append(upper_bound, beta.ppf(one_minus_alpha, alpha_p, beta_p))
  action = np.random.choice(np.flatnonzero(upper_bound == np.max(upper_bound)))
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


def glucose_fitted_q(env, sim_env, tuning_function, tuning_function_parameter, T, t, previous_q, epsilon=0.05,
                     gamma=0.9):
  # Generate fake data if % fake data > 0
  percent_fake = tuning_function(t, T, tuning_function_parameter)
  n_fake = int(np.floor(len(sim_env.X) * percent_fake))
  if n_fake > 0:
    # ToDo: generate fake data
    pass

  # Get features and response
  if X is None:
    X, R = env.X, env.R
    X = [X_i[2:, :] for X_i in X]
    R = [R_i[:-1] for R_i in env.R]
  X_flat = np.vstack(X)
  R_flat = np.hstack(R)
  Xnext_flat = None

  # Get fitted Q target
  Qmax = np.array([np.max([previous_q(x_next, a) for a in range(env.number_of_actions)]) for x_next in Xnext_flat])

  # FQI
  m = RandomForestRegressor()
  m.fit(X_flat, R_flat + gamma * Qmax)

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


def information_expit_epsilon_decay(T, t, zeta, R, delta):
  # 1, delta, R, R*delta, T-t, (T-t)*delta, (T-t)*R, (T-t)*R*delta
  covari = np.kron([1, T-t], np.kron([1, R],[1,delta]))
  return zeta[0] * expit(np.dot(zeta[1:9],covari))


def information_expit_epsilon_decay2(T, t, zeta, delta):
  # 1, delta, T-t, (T-t)*delta
  covari = np.kron([1, T-t], [1,delta])
  return zeta[0] * expit(np.dot(zeta[1:],covari))


def bern_expit_epsilon_decay(T, s1, s2, n1, n2, zeta):
  return zeta[0] * expit(np.dot([1, T-n1-n2, s1, s2, n1, n2], zeta[1:]))



