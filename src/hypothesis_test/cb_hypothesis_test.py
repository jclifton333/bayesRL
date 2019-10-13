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
from src.hypothesis_test.mab_hypothesis_test import approximate_posterior_h0_prob
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics.pairwise import pairwise_kernels
import numpy as np
from sklearn.linear_model import Ridge
import copy
from numba import njit, jit


def cb_ipw(env_, action_probs_list_):
  """
  Get ipw-weighted least squares estimate of CB conditional means.

  :param env_:
  :param action_probs_list_:
  :return:
  """
  beta_hats_ = []
  covs_ = []
  for aprobs, X_a, y_a in zip(action_probs_list_, env_.X_list, env_.y_list):
    # Get ipw estimate of beta
    lm = LinearRegression()
    inv_probs = 1 / np.array(aprobs)
    lm.fit(X_a, y_a, sample_weight=inv_probs)
    beta_hats_.append(lm.coef_)

    # Get approximate sampling variance of ipw estimator
    n, p = X_a.shape
    mse_ = np.mean((lm.predict(X_a) - y_a)**2) / np.max((1.0, n - p))
    cov_a = mse_ * np.linalg.inv(np.dot(X_a.T, np.dot(np.diag(inv_probs), X_a)) + 0.01*np.eye(p))
    covs_.append(cov_a)
  return beta_hats_, covs_


def true_cb_regret(policy, true_model, estimated_model, num_pulls, t, T, pre_generated_data):
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
  reward_draws, context_features = pre_generated_data
  mc_reps = reward_draws[0].shape[1]
  context_features_dim = len(context_features[0])
  for rollout in range(mc_reps):
    regret = 0.0
    num_pulls_rep = copy.copy(num_pulls)
    estimated_model_rollout = copy.deepcopy(estimated_model)  # ToDo: estimated model should include XprimeX_inv

    for tprime in range(t, T):
      # Take action
      context_features_tprime = context_features[tprime - t, rollout]
      # ToDo: can probably be optimized
      means = [np.dot(estimated_model_rollout[a_][0], context_features_tprime) for a_ in
               range(len(estimated_model_rollout))]
      a, _ = policy(means, None, num_pulls_rep, tprime)
      reward = reward_draws[a][tprime - t, rollout]

      # Incremental update to model estimate
      estimated_model_rollout[a][3] = np.vstack((estimated_model_rollout[a][3], context_features_tprime))
      estimated_model_rollout[a][4] = np.hstack((estimated_model_rollout[a][4], reward))
      estimated_model_rollout[a][2] = la.sherman_woodbury(estimated_model_rollout[a][2], context_features_tprime,
                                                          context_features_tprime)
      new_beta = np.dot(estimated_model_rollout[a][2], np.dot(estimated_model_rollout[a][3].T,
                                                              estimated_model_rollout[a][4]))

      # Update parameter lists
      # Using eps-greedy, so don't need to update cov estimate
      estimated_model_rollout[a][0] = new_beta

      # Compute regret
      means_at_each_arm = [np.dot(context_features_tprime, true_model[a_][0]) for a_ in range(len(true_model))]
      regret += (np.max(means_at_each_arm) - np.dot(context_features_tprime, true_model[a][0]))

    regrets.append(regret)
  return np.mean(regrets)


def cb_sampling_dbn(true_model_params_, context_dbn_sampler, feature_function, num_pulls):
  """

  :param true_model_params:
  :param context_dbn_sampler: returns samples from context distribution
  :param num_pulls:
  :return:
  """
  sampled_model = []
  p = len(true_model_params_[0][0])
  beta_hats = []
  XprimeX_invs = []
  scales = []
  Xs= []
  ys = []
  num_actions = len(num_pulls)
  for arm in range(num_actions):
    beta_for_arm = true_model_params_[0][arm]
    scale_for_arm = true_model_params_[1][arm]
    arm_pulls = num_pulls[arm]

    contexts = context_dbn_sampler(arm_pulls)  # Draw contexts
    context_features = np.array([feature_function(c) for c in contexts])
    means = np.dot(context_features, beta_for_arm)  # Get true means and variances at contexts

    # Sample beta_hat from sampling_dbn
    y = np.random.normal(loc=means, scale=np.max((0.001, scale_for_arm)))  # In case estimated scale=0
    if context_features.shape[0] == 1:
      XprimeX_inv = la.sherman_woodbury(np.eye(p), context_features[0], context_features[0])
    else:
      XprimeX = np.dot(context_features.T, context_features)
      XprimeX_inv = np.linalg.inv(np.eye(p) + XprimeX)
    beta_hat = np.dot(XprimeX_inv, np.dot(context_features.T, y))

    # Add parameters for this arm
    beta_hats.append(beta_hat)
    scales.append(scale_for_arm)
    XprimeX_invs.append(XprimeX_inv)
    Xs.append(context_features)
    ys.append(y)

  return [beta_hats, scales, XprimeX_invs, Xs, ys]


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
    draws_from_sampled_model = pre_generate_mab_data(sampled_model, context_dbn_sampler, feature_function,
                                                     T-t, reps_to_compute_regret)

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
  for arm in range(len(true_model)):
    beta, scale = true_model[arm][0], true_model[arm][1]  # Should be sigma, not sigma_sq
    means = np.dot(context_features, beta).reshape((T, mc_reps))
    draws = np.random.normal(loc=means, scale=scale)
    draws_for_each_arm.append(draws)

  return draws_for_each_arm, np.array(context_features).reshape((T, mc_reps, context_dim))


def cb_ht_operating_characteristics(baseline_policy, proposed_policy, true_model_list, estimated_model, num_pulls, t, T,
                                    sampling_dbn_sampler, alpha, true_cb_regret, pre_generate_cb_data,
                                    true_model_params, context_dbn_sampler, feature_function,
                                     outer_loop_mc_reps=100, inner_loop_mc_reps=100):
  # Get true regrets to see if H0 is true
  draws_from_estimated_model = pre_generate_cb_data(true_model_params, context_dbn_sampler, feature_function, T-t,
                                                     inner_loop_mc_reps)
  baseline_regret_at_truth = true_cb_regret(baseline_policy, true_model_params, estimated_model, num_pulls, t, T,
                                             draws_from_estimated_model)
  proposed_regret_at_truth = true_cb_regret(proposed_policy, true_model_params, estimated_model, num_pulls, t, T,
                                             draws_from_estimated_model)
  true_diff = baseline_regret_at_truth - proposed_regret_at_truth

  # Rejection rate
  rejections = []
  for sample in range(outer_loop_mc_reps):
    # ToDo: add estimated_model[2:] to estimated_model_ instead?
    estimated_model_ = sampling_dbn_sampler(true_model_params, context_dbn_sampler, feature_function, num_pulls)
    # reject = conduct_cb_ht(baseline_policy, proposed_policy, true_model_list, estimated_model_, num_pulls,
    #                        t, T, sampling_dbn_sampler, alpha, true_cb_regret, pre_generate_cb_data,
    #                        context_dbn_sampler, feature_function, mc_reps=inner_loop_mc_reps)
    reject = conduct_approximate_cb_ht(baseline_policy, proposed_policy, true_model_list, estimated_model_, num_pulls,
                                       t, T, sampling_dbn_sampler, alpha, true_cb_regret, pre_generate_cb_data,
                                       context_dbn_sampler, feature_function, mc_reps=inner_loop_mc_reps)
    rejections.append(reject)

  if true_diff > 0:  # H0 false
    type1 = None
    type2 = float(np.mean(rejections))
  else: # H0 true
    type1 = float(np.mean(rejections))
    type2 = None

  return {'type1': type1, 'type2': type2}


def conduct_approximate_cb_ht(baseline_policy, proposed_policy, true_model_list, estimated_model, num_pulls,
                              t, T, sampling_dbn_sampler, alpha, true_cb_regret, pre_generate_cb_data, context_dbn_sampler,
                              feature_function, contamination=0.99, mc_reps=1000):
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
  pre_generated_data = pre_generate_cb_data(estimated_model, context_dbn_sampler, feature_function, T-t, mc_reps)

  # Check that estimated proposed regret is smaller than baseline; if not, do not reject
  estimated_baseline_regret = true_cb_regret(baseline_policy, estimated_model, estimated_model,
                                             num_pulls, t, T, pre_generated_data)
  estimated_proposed_regret = true_cb_regret(proposed_policy, estimated_model, estimated_model,
                                              num_pulls, t, T, pre_generated_data)

  test_statistic = estimated_baseline_regret - estimated_proposed_regret
  print('test_statistic: {}'.format(test_statistic))

  if test_statistic < 0:
    return False
  else:
    diff_sampling_dbn = []
    for true_model in true_model_list:  # Assuming true_model_list are draws from approximate sampling dbn
      # Pre-generate data from sampled model
      pre_generated_data = pre_generate_cb_data(estimated_model, context_dbn_sampler, feature_function, T-t, mc_reps)

      # Compute regret at sampled model
      true_baseline_regret = true_cb_regret(baseline_policy, true_model, estimated_model, num_pulls,
                                            t, T, pre_generated_data)
      true_proposed_regret = true_cb_regret(proposed_policy, true_model, estimated_model, num_pulls,
                                             t, T, pre_generated_data)
      diff_sampling_dbn.append(true_baseline_regret - true_proposed_regret)

    # Reject is posterior probability of null is small
    posterior_h0_prob, _ = approximate_posterior_h0_prob(diff_sampling_dbn, epsilon=contamination)
    return posterior_h0_prob < alpha


def is_cb_h0_true(baseline_policy, proposed_policy, estimated_model, number_of_pulls, t, T, true_cb_regret_,
                  pre_generate_cb_data_, true_model_params_, true_model_context_sampler_, mc_reps, feature_function):
  """
  Determine whether h0 is true starting at time t, using Monte Carlo estimates of regrets under true model.

  :param baseline_policy:
  :param proposed_policy:
  :param estimated_model:
  :param number_of_pulls:
  :param t:
  :param T:
  :param true_cb_regret_:
  :param pre_generate_cb_data_:
  :param true_model_params_:
  :return:
  """
  draws_from_estimated_model = pre_generate_cb_data_(true_model_params_, true_model_context_sampler_, feature_function,
                                                     T-t, mc_reps)
  baseline_regret_at_truth = true_cb_regret_(baseline_policy, true_model_params_, estimated_model, number_of_pulls,
                                             t, T, draws_from_estimated_model)
  proposed_regret_at_truth = true_cb_regret_(proposed_policy, true_model_params_, estimated_model, number_of_pulls,
                                             t, T, draws_from_estimated_model)
  true_diff = baseline_regret_at_truth - proposed_regret_at_truth
  h0_true = true_diff < 0
  return h0_true, true_diff


def conduct_cb_ht(baseline_policy, proposed_policy, true_model_list, estimated_model, num_pulls,
                  t, T, sampling_dbn_sampler, alpha, true_cb_regret, pre_generate_cb_data, context_dbn_sampler,
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
  pre_generated_data = pre_generate_cb_data(estimated_model, context_dbn_sampler, feature_function, T-t, mc_reps)

  # Check that estimated proposed regret is smaller than baseline; if not, do not reject
  estimated_baseline_regret = true_cb_regret(baseline_policy, estimated_model, estimated_model,
                                             num_pulls, t, T, pre_generated_data)
  estimated_proposed_regret = true_cb_regret(proposed_policy, estimated_model, estimated_model,
                                              num_pulls, t, T, pre_generated_data)

  test_statistic = estimated_baseline_regret - estimated_proposed_regret

  if test_statistic < 0:
    return False
  else:
    # get cutoff by searching over possible models
    sampling_dbns = []
    for true_model in true_model_list:
      # pre-generate data from true_model
      pre_generated_data = pre_generate_cb_data(estimated_model, context_dbn_sampler, feature_function, T-t, mc_reps)

      # check if true_model is in h0
      true_baseline_regret = true_cb_regret(baseline_policy, true_model, estimated_model, num_pulls,
                                             t, T, pre_generated_data)
      true_proposed_regret = true_cb_regret(proposed_policy, true_model, estimated_model, num_pulls,
                                             t, T, pre_generated_data)
      # if in h0, get sampling dbn
      if true_baseline_regret < true_proposed_regret:
        sampling_dbn = cb_regret_sampling_dbn(baseline_policy, proposed_policy, true_model, estimated_model, num_pulls,
                                               t, T, sampling_dbn_sampler, true_cb_regret, pre_generate_cb_data,
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
  tuning_schedule = [0.00 for _ in range(T)]
  alpha_schedule = [0.025 for _ in range(T)]
  baseline_tuning_function = lambda T, t, zeta: baseline_schedule[t]
  tuning_function = lambda T, t, zeta: tuning_schedule[t]
  # tuning_function = tuned_bandit.expit_epsilon_decay
  policy = tuned_bandit.mab_epsilon_greedy_policy
  tuning_function_parameter = ([0.05, 45, 2.5])

  def feature_function(x):
    return x

  def context_dbn_sampler(n):
    X = np.random.multivariate_normal(mean=[0.0, 0.0], cov=np.eye(2), size=n)
    return X
    # if n == 1:
    #   return X[0]
    # else:
    #   return X

  # Generate true model and contexts
  num_sampling_dbn_draws = 100
  num_draws_per_arm = 10
  true_model_params = [[np.random.normal(size=2), np.random.normal(size=2)],
                       [np.random.gamma(1), np.random.gamma(1)]]
  Xs = [feature_function(context_dbn_sampler(num_draws_per_arm)),
        feature_function(context_dbn_sampler(num_draws_per_arm))]
  XprimeX_invs = [np.linalg.inv(np.dot(x.T, x)) for x in Xs]
  y_obs = [np.random.normal(loc=np.dot(x, b), scale=v) for b, v, x in zip(true_model_params[0], true_model_params[1],
                                                                          Xs)]

  beta_hats = []
  scale_hats = []
  ys = []
  for xprimex_inv, x, b, v in zip(XprimeX_invs, Xs, true_model_params[0], true_model_params[1]):
    beta_hats_at_arm = []
    scale_hats_at_arm = []
    ys_at_arm = []
    for draw in range(num_sampling_dbn_draws):
      # Draw ys at features x
      y_draw = np.random.normal(loc=np.dot(x, b), scale=v)

      # Fit model given drawn ys
      beta_hat_at_draw = np.dot(xprimex_inv, np.dot(x.T, y_draw))
      beta_hats_at_arm.append(beta_hat_at_draw)
      scale_hats_at_arm.append(v)  # Assuming scale is known
      ys_at_arm.append(y_draw)
    beta_hats.append(beta_hats_at_arm)
    scale_hats.append(scale_hats_at_arm)
    ys.append(ys_at_arm)

  estimated_model = [true_model_params[0], true_model_params[1], XprimeX_invs, Xs, y_obs]
  number_of_pulls = [num_draws_per_arm, num_draws_per_arm]

  def baseline_policy(estimated_means, standard_errors, number_of_pulls_, t):
    return policy(estimated_means, None, number_of_pulls_, tuning_function=baseline_tuning_function,
                  tuning_function_parameter=None, T=T, t=t, env=None)

  def proposed_policy(estimated_means, standard_errors, number_of_pulls_, t):
    return policy(estimated_means, None, number_of_pulls_, tuning_function=tuning_function,
                  tuning_function_parameter=tuning_function_parameter, T=T, t=t, env=None)

  sampled_model_list = []
  for model in range(num_sampling_dbn_draws):
    sampled_model_list.append([[beta_hats[0][model], beta_hats[1][model]],
                               [scale_hats[0][model], scale_hats[1][model]],
                               [XprimeX_invs[0], XprimeX_invs[1]],
                               [Xs[0], Xs[1]],
                               [ys[0], ys[1]]])

  # Hypothesis test
  operating_char_dict = cb_ht_operating_characteristics(baseline_policy, proposed_policy, sampled_model_list,
                                                        estimated_model, number_of_pulls,
                                                        t, T, cb_sampling_dbn, alpha_schedule[t],
                                                        true_cb_regret,
                                                        pre_generate_cb_data, true_model_params, context_dbn_sampler,
                                                        feature_function,
                                                        inner_loop_mc_reps=100, outer_loop_mc_reps=100)
  print(operating_char_dict)
