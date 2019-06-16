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


def normal_sampling_dbn(model_params, num_pulls):
  """

  :param model_params: List of tuples [(mu1, sigma1), (mu2, sigma2), ...]
  :param num_pulls: List of [num pulls arm 1, num pulls arms 2, ...]
  :return:
  """
  standard_deviations = [param[1] / num_pulls_ for param, num_pulls_, in zip(model_params, num_pulls)]
  means = [param[0] for param in model_params]
  return np.random.normal(loc=means, scale=standard_deviations)


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
  for rollout in range(mc_reps):
    regret = 0.0
    num_pulls_rep = copy.copy(num_pulls)
    estimated_model_rollout = copy.deepcopy(estimated_model)
    for tprime in range(t, T):
      # Take action
      a = policy(estimated_model_rollout, num_pulls_rep, t)
      reward = pre_generated_data[a][t, rollout]

      # Update model estimate
      n = num_pulls_rep[a] + 1
      estimated_model_rollout[a][-1] = np.append(estimated_model_rollout[a][-1], reward)

      # Incremental updates to mean, var
      # ToDo: watch out for instability in the variance update
      previous_mean = estimated_model_rollout[a][0]
      error = reward - previous_mean
      new_mean = previous_mean + (error / n)
      previous_var = estimated_model_rollout[a][1]
      new_var = (n-2)/(n-1)*previous_var + error**2 / n
      estimated_model_rollout[a][0] = new_mean
      estimated_model_rollout[a][1] = new_var
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
    mean, sigma = arm_params
    sampled_mean = np.random.normal(loc=mean, scale=sigma/np.sqrt(arm_pulls))
    sampled_variance = (sigma**2 / (arm_pulls - 1)) * np.random.gamma((arm_pulls - 1)/2, 2)
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
    mu, sigma = arm_params[0], arm_params[1]
    draws = np.random.normal(loc=mu, scale=sigma, size=(T, mc_reps))
    draws_for_each_arm.append(draws)
  return draws_for_each_arm


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
  print('test statistic: {}'.format(test_statistic))
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
      print('cutoff: {}'.format(cutoff))
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
  alpha_schedule = [0.05 for _ in range(T)]
  baseline_tuning_function = lambda T, t, zeta: baseline_schedule[t]
  tuning_function = tuned_bandit.expit_epsilon_decay
  policy = tuned_bandit.mab_epsilon_greedy_policy
  tuning_function_parameter = ([0.05, 45, 2.5])

  # Do hypothesis test
  pulls_from_each_arm = [np.random.normal(loc=0.0, size=2), np.random.normal(loc=1.0, size=2)]
  estimated_model = [[np.mean(pulls), np.std(pulls), pulls] for pulls in pulls_from_each_arm]
  number_of_pulls = [2, 2]

  def baseline_policy(estimated_model_params, number_of_pulls_, t):
    estimated_means = [param[0] for param in estimated_model_params]
    return policy(estimated_means, None, number_of_pulls_, tuning_function=baseline_tuning_function,
                  tuning_function_parameter=None, T=T, t=t, env=None)

  def proposed_policy(estimated_model_params, number_of_pulls_, t):
    estimated_means = [param[0] for param in estimated_model_params]
    return policy(estimated_means, None, number_of_pulls_, tuning_function=tuning_function,
                  tuning_function_parameter=tuning_function_parameter, T=T, t=t, env=None)

  true_model_list = [[(np.random.normal(0.0), np.random.gamma(1.0)) for i in range(2)]
                     for j in range(num_candidate_models)]
  for i in range(1):
    ans = conduct_mab_ht(baseline_policy, proposed_policy, true_model_list, estimated_model, number_of_pulls, t, T,
                         normal_mab_sampling_dbn, alpha_schedule[t], true_normal_mab_regret,
                         pre_generate_normal_mab_data, mc_reps=100)
    print(ans)



