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
from scipy.linalg import block_diag
from scipy.special import expit
import scipy.stats
import src.policies.linear_algebra as la
import pdb
import numpy as np
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
  X = np.delete(env.X, -1, 0)
  # Get reward
  x = copy.copy(env.curr_context)
  estimated_rewards = np.dot(beta_hat, env.curr_context) 
  
  # Get UCB
  q_hat_0 = estimated_rewards[0]
  q_hat_1 = estimated_rewards[1]
  b = env.X[-1,]
  
  B_0 = X[np.where(env.A==0),][0]
  U_0 = env.U[np.where(env.A==0),][0]
  omega_hat_0 = np.zeros([env.context_dimension, env.context_dimension])
  Sigma_hat_0 = np.zeros([env.context_dimension, env.context_dimension])
  for i in range( int(len(env.A)-sum(env.A)) ):
    omega_hat_0 += np.outer(B_0[i],B_0[i].T) /(len(env.A)-sum(env.A))
    Sigma_hat_0 += np.outer(B_0[i],B_0[i].T)*(U_0[i]-np.dot(B_0[i],beta_hat[0]))**2/(len(env.A)-sum(env.A))
  omega_hat_inv0 = np.linalg.inv(omega_hat_0)
  sigma_hat_0 = np.dot(np.dot(b, np.matmul(np.matmul(omega_hat_inv0, Sigma_hat_0),
                        omega_hat_inv0)), b)
  
  B_1 = X[np.where(env.A==1),][0]
  U_1 = env.U[np.where(env.A==1),][0]
  omega_hat_1 = np.zeros([env.context_dimension, env.context_dimension])
  Sigma_hat_1 = np.zeros([env.context_dimension, env.context_dimension])
  for i in range( int(sum(env.A)) ):
    omega_hat_1 += np.outer(B_1[i],B_1[i].T) /sum(env.A)
    Sigma_hat_1 += np.outer(B_1[i],B_1[i].T)*(U_1[i]-np.dot(B_1[i],beta_hat[1]))**2/sum(env.A)
  omega_hat_inv1 = np.linalg.inv(omega_hat_1)
  sigma_hat_1 = np.dot(np.dot(b, np.matmul(np.matmul(omega_hat_inv1, Sigma_hat_1),
                        omega_hat_inv1)), b)
  
  kexi_0 = q_hat_0 + scipy.stats.norm.ppf(alpha)*sigma_hat_0/np.sqrt(len(env.A))
  kexi_1 = q_hat_1 + scipy.stats.norm.ppf(alpha)*sigma_hat_1/np.sqrt(len(env.A))        
  
  a = np.argmax([kexi_0, kexi_1])
  
  return a

  

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
  alpha = tuning_function(T, t, tuning_function_parameter)/40 + 0.5
  z = scipy.stats.norm.ppf(alpha)
  action = np.argmax(estimated_means + z * standard_errors)
  return action
  

def mab_thompson_sampling_policy(estimated_means, standard_errors, number_of_pulls, tuning_function,
                                 tuning_function_parameter, T, t, env):

  shrinkage = tuning_function(T, t, tuning_function_parameter)
  shrinkage = np.min((shrinkage, 1.0))
  posterior_draw = env.sample_from_posterior(variance_shrinkage=shrinkage)
  return np.argmax([posterior_draw[a]['mu_draw'] for a in range(env.number_of_actions)])


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
  return zeta[0] * expit(zeta[1] + zeta[2]*(T - t))


def stepwise_linear_epsilon(T, t, zeta):
  J = len(zeta)
  interval = int(T/float(J))
  if t == 0:
    j = 0
  else:
    j = int(np.floor((T-t)/interval))
  epsilon = sum(zeta[:j]) + ((T-t) - j*interval) * zeta[j] / interval
  return epsilon







