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
      a = policy([param[0] for param in estimated_model_rollout], None, num_pulls_rep, tprime)
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
  cutoffs = [np.percentile(sampling_dbn, (1 - alpha)*100) for sampling_dbn in sampling_dbns]
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
                                                           inner_loop_mc_reps=500, outer_loop_mc_reps=500)
    print(operating_char_dict)



