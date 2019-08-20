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
from sklearn.neighbors import KernelDensity
from scipy.stats import t, norm
from scipy.integrate import quad
from numba import njit, jit


def approximate_posterior_h0_prob(empirical_dbn, df=3):
  """
  Compute the approximate posterior probability that X < 0, treating empirical_dbn as likelihood and using
  heavy-tailed.

  :param empirical_dbn:
  :return:
  """
  # Generate data from prior
  prior_draws = np.random.normal(size=100)

  # Get histograms and combine
  empirical_histogram = np.histogram(empirical_dbn, density=True)
  bins_ = empirical_histogram[1]
  prior_histogram = np.histogram(prior_draws, bins=bins_, density=True)

  # Get probability less than 0
  posterior_density = empirical_histogram[0] * prior_histogram[0]
  total_mass = np.sum(posterior_density)
  mass_less_than_0 = np.sum(posterior_density[np.where(bins_ <= 0)])

  return mass_less_than_0 / total_mass, [float(d) for d in posterior_density]


def stratified_bootstrap_indices(num_actions, actions):
  """
  Sample split within each arm, bootstrap oversample to original sample size. Return corresponding indices

  :param num_actions:
  :param actions:
  :return:
  """
  bootstrap_ixs_1 = []
  bootstrap_ixs_2 = []
  for a in range(num_actions):
    action_ixs = np.where(actions == a)[0]
    num_action_a = len(action_ixs)
    if num_action_a > 1:
      num_ixs_1 = int(num_action_a / 2)
      # Sample split
      ixs_1 = np.random.choice(action_ixs, num_ixs_1)
      ixs_2 = [ix for ix in action_ixs if ix not in ixs_1]

      # Oversample
      ixs_1 = np.random.choice(ixs_1, num_action_a, replace=True).astype(int).tolist()
      ixs_2 = np.random.choice(ixs_2, num_action_a, replace=True).astype(int).tolist()

      bootstrap_ixs_1 += ixs_1
      bootstrap_ixs_2 += ixs_2
    else:
      bootstrap_ixs_1 += action_ixs.tolist()
      bootstrap_ixs_2 += action_ixs.tolist()
  return bootstrap_ixs_1, bootstrap_ixs_2


def ipw(num_actions, actions, action_probs, rewards):
  """
  Helper function to compute IPW estimates of the MAB.

  :param actions:
  :param action_probs:
  :param rewards:
  :return:
  """
  mean_estimates = []
  pooled_sse = 0.0  # Assuming all arms have same variance
  for a in range(num_actions):
    a_ixs = np.where(actions == a)
    inverse_probs_for_a = 1 / action_probs[a_ixs]
    rewards_for_a = rewards[a_ixs]
    mean_estimate = np.dot(rewards_for_a, inverse_probs_for_a) / np.sum(inverse_probs_for_a)
    mean_estimates.append(mean_estimate)
    if len(rewards_for_a) > 0:
      pooled_sse += np.dot((rewards_for_a - mean_estimate)**2, inverse_probs_for_a) / np.sum(inverse_probs_for_a)
  pooled_std = np.sqrt(pooled_sse / (len(rewards) - 1 ))
  std_estimates = [pooled_std]*num_actions
  return mean_estimates, std_estimates


def estimated_normal_mab_regret(policy, t, T, pre_generated_data, mu_opts, true_means, estimated_means, actions,
                                action_probs, reward_history):
  """
  Estimate normal_mab_regret with sample-split IPW estimates.

  :param policy:
  :param num_pulls:
  :param t:
  :param T:
  :param pre_generated_data:
  :return:
  """
  num_actions = len(np.unique(actions))  # Assuming all actions have been observed!

  regrets = []
  mc_reps = pre_generated_data[0].shape[1]

  for rollout in range(mc_reps):
    estimated_mean_rollout = estimated_means[rollout]

    # Initialize data for rollout
    mu_opt = mu_opts[rollout]
    regret = 0.0
    num_pulls_rep = [np.sum(actions == a) for a in range(num_actions)]
    # For stable online estimate of variance

    for tprime in range(t, T):
      # Take action
      a, _ = policy(estimated_mean_rollout, None, num_pulls_rep, tprime)
      reward = pre_generated_data[a][tprime - t, rollout]

      # Update model estimate
      n = num_pulls_rep[a] + 1

      # Incremental update to mean
      previous_mean = estimated_mean_rollout[a]
      error = reward - previous_mean
      new_mean = previous_mean + (error / n)

      # Update parameter lists
      estimated_mean_rollout[a] = new_mean
      num_pulls_rep[a] = n

      regret += (mu_opt - true_means[rollout, a])
    regrets.append(regret)
  return np.mean(regrets)


def true_normal_mab_regret(policy, true_model, estimated_model, num_pulls, t, T, pre_generated_data):
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
  mu_opt = np.max([params[0] for params in true_model])
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
      a, _ = policy([param[0] for param in estimated_model_rollout], None, num_pulls_rep, tprime)
      reward = pre_generated_data[a][tprime - t, rollout]

      # Update model estimate
      n = num_pulls_rep[a] + 1

      # Incremental update to mean
      previous_mean = estimated_model_rollout[a][0]
      error = reward - previous_mean
      new_mean = previous_mean + (error / n)

      # Incremental update to ssd, var
      previous_sum_squared_diffs = sum_squared_diffs_rollout[a]
      new_sum_squared_diffs = previous_sum_squared_diffs + error * (reward - new_mean)
      new_var = new_sum_squared_diffs / np.max((1.0, n - 1))

      # Update parameter lists
      estimated_model_rollout[a][0] = new_mean
      estimated_model_rollout[a][1] = new_var
      sum_squared_diffs_rollout[a] = new_sum_squared_diffs
      num_pulls_rep[a] = n

      regret += (mu_opt - true_model[a][0])
    regrets.append(regret)
  return np.mean(regrets)


def normal_mab_sampling_dbn(true_model_params, num_pulls):
  """

  :param true_model_params:
  :param num_pulls:
  :return:
  """
  sampled_model = []
  for arm_params, arm_pulls in zip(true_model_params, num_pulls):
    mean, sigma_sq = arm_params # Shouldsdf be sigma, not sigma_sq
    sampled_mean = np.random.normal(loc=mean, scale=sigma_sq / np.sqrt(arm_pulls))
    pulls_m1 = np.max((1.0, arm_pulls - 1))
    sampled_variance = (sigma_sq / pulls_m1) * np.random.gamma(pulls_m1/2, 2)
    sampled_model.append([sampled_mean, sampled_variance])
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
    sampled_model = sampling_dbn(true_model, num_pulls)
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
  cutoffs = [np.percentile(sampling_dbn, (1-alpha)*100) for sampling_dbn in sampling_dbns]
  return np.max(cutoffs)


def pre_generate_normal_mab_data(true_model, T, mc_reps):
  """
  Pre-generate draws from normal mab.

  :param true_model:
  :param T:
  :return:
  """
  draws_for_each_arm = []
  for arm_params in true_model:
    mu, sigma_sq = arm_params[0], arm_params[1]  # Should be sigma, not sigma_sq
    draws = np.random.normal(loc=mu, scale=sigma_sq, size=(T, mc_reps))
    draws_for_each_arm.append(draws)
  return draws_for_each_arm


def pre_generate_normal_mab_data_from_ipw(T, mc_reps, t, num_actions, actions, action_probs, reward_history):
  # Collect means and variances
  true_locs = np.zeros((0, num_actions))
  true_scales = np.zeros((0, num_actions))
  estimated_locs = np.zeros((0, num_actions))
  for mc_rep in range(mc_reps):
    estimated_ixs, true_ixs = stratified_bootstrap_indices(num_actions, actions)
    # Get mean that will be used for model
    true_mean, true_std = ipw(num_actions, actions[true_ixs], action_probs[true_ixs],
                              reward_history[true_ixs])
    true_locs = np.vstack((true_locs, true_mean))
    true_scales = np.vstack((true_scales, true_std))

    # Get mean that will be used for policy estimate
    estimated_mean, _ = ipw(num_actions, actions[estimated_ixs], action_probs[estimated_ixs],
                            reward_history[estimated_ixs])
    estimated_locs = np.vstack((estimated_locs, estimated_mean))

  # Draw from corresponding distributions
  draws_for_each_arm = []
  for arm in range(num_actions):
    draws = np.random.normal(loc=true_locs[:, arm], scale=true_scales[:, arm], size=(T, mc_reps))
    draws_for_each_arm.append(draws)

  mu_opts = true_locs.max(axis=1)  # For computing regret later
  return draws_for_each_arm, mu_opts, true_locs, estimated_locs


def is_h0_true(baseline_policy, proposed_policy, estimated_model, num_pulls, t, T,
               true_mab_regret, pre_generate_mab_data,
               true_model_params, inner_loop_mc_reps=100):

  # Get true regrets to see if H0 is true
  draws_from_estimated_model = pre_generate_mab_data(true_model_params, T-t, inner_loop_mc_reps)
  baseline_regret_at_truth = true_mab_regret(baseline_policy, true_model_params, estimated_model, num_pulls, t, T,
                                             draws_from_estimated_model)
  proposed_regret_at_truth = true_mab_regret(proposed_policy, true_model_params, estimated_model, num_pulls, t, T,
                                             draws_from_estimated_model)
  true_diff = baseline_regret_at_truth - proposed_regret_at_truth
  h0_true = true_diff < 0
  return h0_true, true_diff


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


def conduct_approximate_mab_ht(baseline_policy, proposed_policy, true_model_list, estimated_model, num_pulls,
                               t, T, sampling_dbn_sampler, alpha, true_mab_regret, pre_generate_mab_data,
                               num_actions, actions, action_probs, reward_history, mc_reps=1000):
  """
  Draw from estimated sampling dbn of (Baseline regret - Proposed regret)
  and reject if alpha^th percentile is above 0.
  """
  draws_from_estimated_model, mu_opts, true_means, estimated_means = \
    pre_generate_normal_mab_data_from_ipw(T, mc_reps, t, num_actions, actions, action_probs, reward_history)
  draws_from_estimated_model = pre_generate_mab_data(estimated_model, T-t, mc_reps)
  # estimated_means = [p[0] for p in estimated_model]

  print(true_means.mean(axis=0))
  # Check that estimated proposed regret is smaller than baseline; if not, do not reject
  estimated_baseline_regret = estimated_normal_mab_regret(baseline_policy, t, T, draws_from_estimated_model, mu_opts,
                                                          true_means, estimated_means, actions, action_probs,
                                                          reward_history)
  estimated_proposed_regret = estimated_normal_mab_regret(proposed_policy, t, T, draws_from_estimated_model, mu_opts,
                                                          true_means, estimated_means, actions, action_probs,
                                                          reward_history)
  test_statistic = estimated_baseline_regret - estimated_proposed_regret

  if test_statistic < 0:
    return False, test_statistic, None
  else:
    diff_sampling_dbn = []
    for true_model in true_model_list:  # ToDo: Assuming true_model_list are draws from approx sampling dbn!
      # Pre-generate data from true_model
      draws_from_true_model = pre_generate_mab_data(estimated_model, T-t, mc_reps)

      # Check if true_model is in H0
      true_baseline_regret = true_mab_regret(baseline_policy, true_model, estimated_model, num_pulls, t, T,
                                             draws_from_true_model)
      true_proposed_regret = true_mab_regret(proposed_policy, true_model, estimated_model, num_pulls, t, T,
                                             draws_from_true_model)
      diff_sampling_dbn.append(true_baseline_regret - true_proposed_regret)
    # Reject if alpha^th percentile < 0
    alpha_th_percentile = np.percentile(diff_sampling_dbn, 100*alpha)
    # posterior_h0_prob, posterior_density = approximate_posterior_h0_prob(diff_sampling_dbn)
    return (alpha_th_percentile < 0), test_statistic, None


def conduct_mab_ht(baseline_policy, proposed_policy, true_model_list, estimated_model, num_pulls,
                   t, T, sampling_dbn_sampler, alpha, true_mab_regret, pre_generate_mab_data,
                   num_actions, actions, action_probs, reward_history, mc_reps=1000):
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
  # ToDo: Make sure hyp. test direction is correct!
  draws_from_estimated_model, mu_opts, true_means, estimated_means = \
    pre_generate_normal_mab_data_from_ipw(T, mc_reps, t, num_actions, actions, action_probs, reward_history)
  draws_from_estimated_model = pre_generate_mab_data(estimated_model, T-t, mc_reps)
  # estimated_means = [p[0] for p in estimated_model]

  # Check that estimated proposed regret is smaller than baseline; if not, do not reject
  estimated_baseline_regret = estimated_normal_mab_regret(baseline_policy, t, T, draws_from_estimated_model, mu_opts,
                                                          true_means, estimated_means, actions, action_probs,
                                                          reward_history)
  estimated_proposed_regret = estimated_normal_mab_regret(proposed_policy, t, T, draws_from_estimated_model, mu_opts,
                                                          true_means, estimated_means, actions, action_probs,
                                                          reward_history)
  test_statistic = estimated_baseline_regret - estimated_proposed_regret

  if test_statistic < 0:
    return False, test_statistic
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
        return True, test_statistic
      else:
        return False, test_statistic
    else:  # If sampling_dbns empty, use cutoff=0
      return True, test_statistic


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

  # Do hypothesis test
  true_model_params = [(0.0, 1.0), (1.0, 1.0)]
  pulls_from_each_arm = [np.random.normal(loc=p[0], scale=p[1], size=2) for p in true_model_params]
  estimated_model = [[np.mean(pulls), np.std(pulls)] for pulls in pulls_from_each_arm]
  number_of_pulls = [2, 2]

  def baseline_policy(estimated_means, standard_errors, number_of_pulls_, t):
    return policy(estimated_means, None, number_of_pulls_, tuning_function=baseline_tuning_function,
                  tuning_function_parameter=None, T=T, t=t, env=None)

  def proposed_policy(estimated_means, standard_errors, number_of_pulls_, t):
    return policy(estimated_means, None, number_of_pulls_, tuning_function=tuning_function,
                  tuning_function_parameter=tuning_function_parameter, T=T, t=t, env=None)

  # true_model_list = [[(np.random.normal(p[0], p[1]/np.sqrt(2)), np.random.gamma(1, 2)) for p in estimated_model]]
  # true_model_list = [[(np.random.normal(p[0], p[1]/np.sqrt(2)), np.random.gamma(1)) for p in estimated_model]]
  true_model_list = [[(np.random.normal(0.0, 1.0), np.random.gamma(1)) for p in estimated_model]]
  for i in range(10):
    ans = conduct_mab_ht(baseline_policy, proposed_policy, true_model_list, estimated_model, number_of_pulls, t, T,
                         normal_mab_sampling_dbn, alpha_schedule[t], true_normal_mab_regret,
                         pre_generate_normal_mab_data, mc_reps=100)
    # Get operating characteristics
    operating_char_dict = mab_ht_operating_characteristics(baseline_policy, proposed_policy, true_model_list,
                                                           estimated_model, number_of_pulls,
                                                           t, T, normal_mab_sampling_dbn, alpha_schedule[t],
                                                           true_normal_mab_regret,
                                                           pre_generate_normal_mab_data, true_model_params, 
                                                           inner_loop_mc_reps=100, outer_loop_mc_reps=200)
    print(operating_char_dict)



