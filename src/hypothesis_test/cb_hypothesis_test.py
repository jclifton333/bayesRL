"""
Models of the form
[[beta_i, variance_params_i]_i=1^k], where for each arm i and context x, R_i | x ~ N( phi(x).beta_i, phi(x).theta_i ),
where phi is a feature function.
"""
import sys
import pdb
import os
this_dir = os.path.dirname(os.path.abspath(__file__))
project_dir = os.path.join(this_dir, '..', '..')
sys.path.append(project_dir)

from src.policies import tuned_bandit_policies as tuned_bandit
from src.policies import linear_algebra as la
from functools import partial
from sklearn.metrics.pairwise import pairwise_kernels
import numpy as np
import copy
from numba import njit, jit


def true_cb_regret(policy, true_model, estimated_model, context_features, num_pulls, t, T, pre_generated_data):
  """

  :param policy:
  :param true_model:
  :param estimated_model:
  :param num_pulls:
  :param t:
  :param T:
  :param pre_generated_data: list of length (num actions) whose elements are tuples
    ( reward draws, context draws ), each of size (T-t) x num_mc_reps
  :return:
  """
  regrets = []
  mc_reps = pre_generated_data[0].shape[1]
  context_features_dim = pre_generated_data[1][0].shape[0]
  for rollout in range(mc_reps):
    regret = 0.0
    num_pulls_rep = copy.copy(num_pulls)
    estimated_model_rollout = copy.deepcopy(estimated_model) # ToDo: estimated model should include XprimeX_inv
    XprimeX_invs, Xs, ys, beta_hats = estimated_model_rollout

    for tprime in range(t, T):
      # Take action
      # ToDo: change pre_generated_data to include context
      context_features_tprime = context_features[tprime, rollout]
      means = [np.dot(beta_hat, context_features_tprime) for beta_hat in estimated_model_rollout[3]]
      a = policy(means, None, num_pulls_rep, tprime)
      reward = pre_generated_data[a][tprime - t, rollout]

      # Update model estimate
      n = num_pulls_rep[a] + 1

      # Incremental update to model estimate
      estimated_model_rollout[1][a] = np.vstack((estimated_model_rollout[1][a], context_features_tprime))
      estimated_model_rollout[2][a] = np.hstack((estimated_model_rollout[2][a],reward))
      estimated_model_rollout[0][a] = la.sherman_woodbury(estimated_model_rollout[0][a], context_features_tprime,
                                                          context_features_tprime)
      new_beta = np.dot(estimated_model_rollout[0][a], np.dot(estimated_model_rollout[1][a].T,
                                                              estimated_model_rollout[2][a]))

      # Update parameter lists
      # Using eps-greedy, so don't need to update cov estimate
      estimated_model_rollout[a][3] = new_beta

      # Compute regret
      means_at_each_arm = [np.dot(context_features_tprime, true_model[a_][0]) for a_ in range(len(true_model))]
      regret += (np.max(means_at_each_arm) - np.dot(context_features_tprime, true_model[a][0]))

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
    beta, var_params, _ = true_model_params
    contexts = context_dbn_sampler(arm_pulls)  # Draw contexts
    context_features = [feature_function(c) for c in contexts]
    means = np.dot(context_features, beta)  # Get true means and variances at contexts
    variances = np.dot(context_features, var_params)
    covariance_matrix = np.diag(variances)

    # Sample beta_hat and theta_hat from sampling_dbn
    df = np.max((1.0, arm_pulls - p))
    cov = np.dot(contexts.prime, np.dot(covariance_matrix, contexts))  # ToDo: Check this formula
    beta_hat = np.random.multivariate_normal(beta, cov / df)
    var_params_hat = None # ToDo: figure out covariance estimator

    sampled_model.append([beta_hat, var_params_hat])
  return sampled_model


def cb_regret_sampling_dbn(baseline_policy, proposed_policy, true_model, estimated_model, num_pulls,
                            t, T, sampling_dbn, true_mab_regret, pre_generate_mab_data, context_dbn_sampler,
                           feature_function, sampling_dbn_draws=50, reps_to_compute_regret=100):
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
  # Draw contexts
  contexts = context_dbn_sampler(T*mc_reps)
  context_features = [feature_function(c) for c in contexts]
  context_dim = len(context_features[0])

  # Draw rewards at each arm
  for arm_params in true_model:
    beta, variance_params = arm_params[0], arm_params[1]  # Should be sigma, not sigma_sq
    means, sigma_sqs = np.dot(context_features, beta).reshape((T, mc_reps)), \
                       np.dot(context_features, variance_params).reshape((T, mc_reps))
    draws = np.random.normal(loc=means, scale=np.sqrt(sigma_sqs))
    draws_for_each_arm.append((draws, context_features))

  return draws_for_each_arm, context_features


def cb_ht_operating_characteristics(baseline_policy, proposed_policy, true_model_list, estimated_model, num_pulls, t, T,
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


def conduct_cb_ht(baseline_policy, proposed_policy, true_model_list, estimated_model, num_pulls,
                  t, T, sampling_dbn_sampler, alpha, true_mab_regret, pre_generate_mab_data, context_dbn_sampler,
                  feature_function, mc_reps=1000):
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
  draws_from_estimated_model, context_features = pre_generate_mab_data(estimated_model, context_dbn_sampler,
                                                                       feature_function, T-t, mc_reps)

  # Check that estimated proposed regret is smaller than baseline; if not, do not reject
  estimated_baseline_regret = true_mab_regret(baseline_policy, estimated_model, estimated_model, context_features,
                                              num_pulls, t, T, draws_from_estimated_model)
  estimated_proposed_regret = true_mab_regret(proposed_policy, estimated_model, estimated_model, context_features,
                                              num_pulls, t, T, draws_from_estimated_model)

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
      true_baseline_regret = true_mab_regret(baseline_policy, true_model, estimated_model, context_features, num_pulls,
                                             t, T, draws_from_true_model)
      true_proposed_regret = true_mab_regret(proposed_policy, true_model, estimated_model, context_features, num_pulls,
                                             t, T, draws_from_true_model)
      # If in H0, get sampling dbn
      if true_baseline_regret < true_proposed_regret:
        sampling_dbn = cb_regret_sampling_dbn(baseline_policy, proposed_policy, true_model, estimated_model, num_pulls,
                                               t, T, sampling_dbn_sampler, true_mab_regret, pre_generate_mab_data,
                                               context_dbn_sampler, feature_function, reps_to_compute_regret=mc_reps)
        sampling_dbns.append(sampling_dbn)

    if sampling_dbns:  # If sampling dbns non-empty, compute cutoff
      cutoff = cutoff_for_ht(alpha, sampling_dbns)
      if test_statistic > cutoff:
        return True
      else:
        return False
    else:  # If sampling_dbns empty, use cutoff=0
      return True


if __name__ == "__main__":
  # Settings
  T = 20
  t = 1
  num_candidate_models = 10
  baseline_schedule = [0.05 for _ in range(T)]
  tuning_schedule = [0.2 for _ in range(T)]
  alpha_schedule = [0.025 for _ in range(T)]
  baseline_tuning_function = lambda T, t, zeta: baseline_schedule[t]
  tuning_function = lambda T, t, zeta: tuning_schedule[t]
  # tuning_function = tuned_bandit.expit_epsilon_decay
  policy = tuned_bandit.mab_epsilon_greedy_policy
  tuning_function_parameter = ([0.05, 45, 2.5])

  def feature_function(x):
    return x

  def context_dbn_sampler(n):
    return np.random.multivariate_normal(mean=[0.0, 0.0], size=n)

  # Do hypothesis test
  true_model_params = [(np.random.normal(size=2), np.random.normal(size=2)),
                       (np.random.normal(size=2), np.random.normal(size=2))]
  Xs = [feature_function(context_dbn_sampler(1)), feature_function(context_dbn_sampler(1))]
  XprimeX_invs = [la.sherman_woodbury(np.eye(2), x, x) for x in Xs]
  ys = [np.random.normal(loc=np.dot(p[0], x), scale=np.dot(p[1], x)) for x, b, t_ in zip(Xs, true_model_params)]
  estimated_model = [XprimeX_invs, Xs, ys]
  number_of_pulls = [1, 1]

  def baseline_policy(estimated_means, standard_errors, number_of_pulls_, t):
    return policy(estimated_means, None, number_of_pulls_, tuning_function=baseline_tuning_function,
                  tuning_function_parameter=None, T=T, t=t, env=None)


  def proposed_policy(estimated_means, standard_errors, number_of_pulls_, t):
    return policy(estimated_means, None, number_of_pulls_, tuning_function=tuning_function,
                  tuning_function_parameter=tuning_function_parameter, T=T, t=t, env=None)


  # true_model_list = [[(np.random.normal(p[0], p[1]/np.sqrt(2)), np.random.gamma(1, 2)) for p in estimated_model]]
  # true_model_list = [[(np.random.normal(p[0], p[1]/np.sqrt(2)), np.random.gamma(1)) for p in estimated_model]]
  true_model_list = [[(np.random.normal(0.0, 1.0), np.random.gamma(1)) for p in estimated_model]]
  ans = conduct_cb_ht(baseline_policy, proposed_policy, true_model_list, estimated_model, number_of_pulls, t, T,
                       cb_sampling_dbn, alpha_schedule[t], true_cb_regret,
                       pre_generate_cb_data, context_dbn_sampler, feature_function, mc_reps=100)
