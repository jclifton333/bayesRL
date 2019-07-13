"""
Models of the form
[[beta_i, sigma_sq_i]_i=1^k], where for each arm i and context x, R_i | x ~ N( phi(x).beta_i, phi(x).theta_i ),
where phi is a feature function.
"""
import sys
import pdb
import os
this_dir = os.path.dirname(os.path.abspath(__file__))
project_dir = os.path.join(this_dir, '..', '..')
sys.path.append(project_dir)

from src.policies import tuned_bandit_policies as tuned_bandit
from functools import partial
import numpy as np
import copy
from numba import njit, jit


def true_cb_regret(policy, true_model, estimated_model, feature_function, num_pulls, t, T, pre_generated_data):
  """

  :param policy:
  :param true_model:
  :param estimated_model:
  :param num_pulls:
  :param t:
  :param T:
  :param pre_generated_data: list whose elements are (T-t) x num_mc_reps arrays of draws for each timepoint,
  corresponding to each arm.
  :return:
  """
  regrets = []
  mc_reps = pre_generated_data[0].shape[1]
  sum_squared_diffs = [params[1] * np.max((pulls - 1, 1))
                       for params, pulls in zip(estimated_model, num_pulls)]  # Need these
  # for stable online update of variance estimate

  for rollout in range(mc_reps):
    regret = 0.0
    num_pulls_rep = copy.copy(num_pulls)
    estimated_model_rollout = copy.deepcopy(estimated_model)
    sum_squared_diffs_rollout = copy.deepcopy(sum_squared_diffs)

    for tprime in range(t, T):
      # Take action
      # ToDo: change pre_generated_data to include context
      context = None
      context_features = feature_function(context)
      a = policy([param[0] for param in estimated_model_rollout], None, num_pulls_rep, tprime)
      reward = pre_generated_data[a][tprime - t, rollout]

      # Update model estimate
      n = num_pulls_rep[a] + 1

      # Incremental update to mean
      previous_beta = estimated_model_rollout[a][0]
      error = reward - np.dot(previous_beta, context_features)
      new_beta = incremental_least_squares(XprimeX_inv, context_features, reward)
      mean_at_new_beta = np.dot(new_beta, context_features)

      # ToDo: this covariance estimator makes no sense
      # Incremental update to ssd, var
      previous_sum_squared_diffs = sum_squared_diffs_rollout[a]
      new_sum_squared_diffs = previous_sum_squared_diffs + error * (reward - mean_at_new_beta)
      new_var = new_sum_squared_diffs / np.max((1.0, n - 1))
      new_theta = incremental_least_squares(XprimeX_inv, context_features, new_var)

      # Update parameter lists
      estimated_model_rollout[a][0] = new_beta
      estimated_model_rollout[a][1] = new_var
      sum_squared_diffs_rollout[a] = new_sum_squared_diffs
      num_pulls_rep[a] = n

      # Get optimal arm
      means_at_each_arm = [np.dot(context_features, true_model[a_][0]) for a_ in range(len(true_model))]
      regret += (np.max(means_at_each_arm) - np.dot(context_features, true_model[a][0]))

    regrets.append(regret)
  return np.mean(regrets)


def cb_sampling_dbn(true_model_params, context_dbn_sampler, feature_function, num_pulls):
  """

  :param true_model_params:
  :param context_dbn_sampler: returns samples from context distribution
  :param num_pulls:
  :return:
  """
  sampled_model = []
  p = len(true_model_params[0][0])
  for arm_params, arm_pulls in zip(true_model_params, num_pulls):
    beta, theta = true_model_params
    contexts = context_dbn_sampler(arm_pulls)  # Draw contexts
    means = np.dot(contexts, beta)  # Get true means and variances at contexts
    sigma_sqs = np.dot(contexts, theta)

    # Sample beta_hat and theta_hat from sampling_dbn
    df = np.max((1.0, arm_pulls - p))
    cov = np.dot(contexts.prime, np.dot(np.diag(sigma_sqs), contexts))  # ToDo: Check this formula
    beta_hat = np.random.multinomial(beta, cov / df)
    theta_hat = None # ToDo: figure out covariance estimator

    sampled_model.append([beta_hat, theta_hat])
  return sampled_model


def mab_regret_sampling_dbn(baseline_policy, proposed_policy, true_model, estimated_model, num_pulls,
                            t, T, sampling_dbn, true_mab_regret, pre_generate_mab_data, sampling_dbn_draws=50,
                            reps_to_compute_regret=100):
  """

  :param baseline_policy:
  :param proposed_policy:
  :param mu:
  :param xbar:
  :param true_model:
  :param estimated_model:
  :param t:
  :param T:
  :return:
  """
  baseline_policy_regrets = []
  proposed_policy_regrets = []
  for rep in range(sampling_dbn_draws):
     # Sample model and pre-generate data
    sampled_model = sampling_dbn(true_model, context_dbn_sampler, feature_function, num_pulls)
    draws_from_sampled_model = pre_generate_mab_data(sampled_model, T-t, reps_to_compute_regret)

    baseline_regret = true_mab_regret(baseline_policy, sampled_model, estimated_model, num_pulls, t, T,
                                      draws_from_sampled_model)
    proposed_regret = true_mab_regret(proposed_policy, sampled_model, estimated_model, num_pulls, t, T,
                                      draws_from_sampled_model)
    baseline_policy_regrets.append(baseline_regret)
    proposed_policy_regrets.append(proposed_regret)
  diffs = np.array(baseline_policy_regrets) - np.array(proposed_policy_regrets)
  return diffs


def cutoff_for_ht(alpha, sampling_dbns):
  cutoffs = [np.percentile(sampling_dbn, (1 - alpha)*100) for sampling_dbn in sampling_dbns]
  return np.max(cutoffs)


def pre_generate_cb_data(true_model, context_dbn_sampler, feature_function, T, mc_reps):
  """
  Pre-generate draws from normal mab.

  :param true_model:
  :param T:
  :return:
  """
  draws_for_each_arm = []
  for arm_params in true_model:
    beta, theta = arm_params[0], arm_params[1]  # Should be sigma, not sigma_sq
    contexts = context_dbn_sampler(T*mc_reps)
    means, sigma_sqs = np.dot(contexts, beta).reshape((T, mc_reps)), np.dot(contexts, theta).reshape((T, mc_reps))
    draws = np.random.normal(loc=means, scale=np.sqrt(sigma_sqs))
    draws_for_each_arm.append(draws, contexts)
  return draws_for_each_arm


def mab_ht_operating_characteristics(baseline_policy, proposed_policy, true_model_list, estimated_model, num_pulls, t, T,
                                     sampling_dbn_sampler, alpha, true_mab_regret, pre_generate_mab_data,
                                     true_model_params, outer_loop_mc_reps=100, inner_loop_mc_reps=100):
  # Get true regrets to see if H0 is true
  draws_from_estimated_model = pre_generate_mab_data(true_model_params, T-t, inner_loop_mc_reps)
  baseline_regret_at_truth = true_mab_regret(baseline_policy, true_model_params, estimated_model, num_pulls, t, T,
                                             draws_from_estimated_model)
  proposed_regret_at_truth = true_mab_regret(proposed_policy, true_model_params, estimated_model, num_pulls, t, T,
                                             draws_from_estimated_model)
  true_diff = baseline_regret_at_truth - proposed_regret_at_truth

  # Rejection rate
  rejections = []
  for sample in range(outer_loop_mc_reps):
    estimated_model = sampling_dbn_sampler(true_model_params, num_pulls)
    reject = conduct_mab_ht(baseline_policy, proposed_policy, true_model_list, estimated_model, num_pulls,
                            t, T, sampling_dbn_sampler, alpha, true_mab_regret, pre_generate_mab_data,
                            mc_reps=inner_loop_mc_reps)
    rejections.append(reject)

  if true_diff > 0:  # H0 false
    type1 = None
    type2 = float(np.mean(rejections))
  else: # H0 true
    type1 = float(np.mean(rejections))
    type2 = None

  return {'type1': type1, 'type2': type2}


def conduct_mab_ht(baseline_policy, proposed_policy, true_model_list, estimated_model, num_pulls,
                   t, T, sampling_dbn_sampler, alpha, true_mab_regret, pre_generate_mab_data, mc_reps=1000):
  """

  :param baseline_policy:
  :param proposed_policy:
  :param true_model_list:
  :param estimated_model:
  :param num_pulls:
  :param t:
  :param T:
  :param sampling_dbn:
  :param mc_reps:
  :return:
  """
  # Pre-generate data from estimated model
  draws_from_estimated_model = pre_generate_mab_data(estimated_model, T-t, mc_reps)

  # Check that estimated proposed regret is smaller than baseline; if not, do not reject
  estimated_baseline_regret = true_mab_regret(baseline_policy, estimated_model, estimated_model, num_pulls, t, T,
                                              draws_from_estimated_model)
  estimated_proposed_regret = true_mab_regret(proposed_policy, estimated_model, estimated_model, num_pulls, t, T,
                                              draws_from_estimated_model)

  test_statistic = estimated_baseline_regret - estimated_proposed_regret

  if test_statistic < 0:
    return False
  else:
    # Get cutoff by searching over possible models
    sampling_dbns = []
    for true_model in true_model_list:
      # Pre-generate data from true_model
      draws_from_true_model = pre_generate_mab_data(estimated_model, T-t, mc_reps)

      # Check if true_model is in H0
      true_baseline_regret = true_mab_regret(baseline_policy, true_model, estimated_model, num_pulls, t, T,
                                             draws_from_true_model)
      true_proposed_regret = true_mab_regret(proposed_policy, true_model, estimated_model, num_pulls, t, T,
                                             draws_from_true_model)
      # If in H0, get sampling dbn
      if true_baseline_regret < true_proposed_regret:
        sampling_dbn = mab_regret_sampling_dbn(baseline_policy, proposed_policy, true_model, estimated_model, num_pulls,
                                               t, T, sampling_dbn_sampler, true_mab_regret, pre_generate_mab_data,
                                               reps_to_compute_regret=mc_reps)
        sampling_dbns.append(sampling_dbn)

    if sampling_dbns:  # If sampling dbns non-empty, compute cutoff
      cutoff = cutoff_for_ht(alpha, sampling_dbns)
      if test_statistic > cutoff:
        return True
      else:
        return False
    else:  # If sampling_dbns empty, use cutoff=0
      return True