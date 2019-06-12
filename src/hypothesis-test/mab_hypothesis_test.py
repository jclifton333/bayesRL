import numpy as np
import copy


def true_normal_mab_regret(policy, true_model, estimated_model, num_pulls, t, T, mc_reps=5000):
  """

  :param policy:
  :param true_model:
  :param estimated_model:
  :param num_pulls:
  :param t:
  :param T:
  :param mc_reps:
  :return:
  """
  mu_opt = np.max([params[0] for params in true_model])
  regrets = []
  for rollout in range(mc_reps):
    regret = 0.0
    num_pulls_rep = copy.copy(num_pulls)
    estimated_model_rollout = copy.copy(estimated_model)
    for tprime in range(t, T):
      # Take action
      action = policy(estimated_model_rollout, true_model, t, T)
      reward = np.random.normal(true_model[a][0], true_model[a][1])

      # Update model estimate
      estimated_model[a][-1].append(reward)
      estimated_model[a][0] = np.mean(estimated_model[a][-1])
      estimated_model[a][1] = np.std(estimated_model[a][-1])

      regret += (mu_opt - true_model[a][0])
    regrets.append(regret)
  return np.mean(regrets)


def normal_mab_sampling_dbn(true_model_params, num_pulls):
  """

  :param true_model_params:
  :param num_pulls:
  :return:
  """
  xbar_draws = []
  for arm_params, arm_pulls in zip(true_model_params, num_pulls):
    mean, sigma = arm_params
    xbar_draw = np.random.normal(loc=mean, scale=sigma / np.sqrt(num_pulls))
    xbar_draws.append(xbar_draw)
  return xbar_draws


def mab_regret_sampling_dbn(baseline_policy, proposed_policy, true_model, estimated_model, num_pulls,
                            t, T, sampling_dbn, mc_reps=1000):
  """

  :param baseline_policy:
  :param proposed_policy:
  :param mu:
  :param xbar:
  :param true_model:
  :param estimated_model:
  :param t:
  :param T:
  :param mc_reps:
  :return:
  """
  baseline_policy_regrets = []
  proposed_policy_regrets = []
  for rep in range(mc_reps):
    sampled_model = sampling_dbn(true_model, num_pulls)
    baseline_regret = true_mab_regret(baseline_policy, true_model, sampled_model, estimated_model, num_pulls, t, T)
    proposed_regret = true_mab_regret(proposed_policy, true_model, sampled_model, estimated_model, num_pulls, t, T)
    baseline_policy_regrets.append(baseline_regret)
    proposed_policy_regrets.append(proposed_regret)
  diffs = np.array(baseline_policy_regrets) - np.array(proposed_policy_regrets)
  return diffs


def cutoff_for_ht(alpha, sampling_dbns):
  cutoffs = [np.percentile(sampling_dbn, (1 - alpha)*100) for sampling_dbn in sampling_dbns]
  return np.max(cutoffs)


def conduct_mab_ht(baseline_policy, proposed_policy, true_model_list, estimated_model, num_pulls,
                   t, T, sampling_dbn, alpha, mc_reps=1000):
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
  # Get cutoff by searching over possible models
  sampling_dbns = []
  for true_model in true_model_list:
    # ToDo: check if in H0
    sampling_dbn = mab_regret_sampling_dbn(baseline_policy, proposed_policy, true_model, estimated_model, num_pulls,
                                           t, T, sampling_dbn, mc_reps=1000)
    sampling_dbns.append(sampling_dbn)
  cutoff = cutoff_for_ht(alpha, sampling_dbns)

  # Compare test statistic to cutoff
  estimated_baseline_regret = true_normal_mab_regret(baseline_policy, estimated_model, estimated_model, num_pulls, t, T,
                                                     mc_reps=5000)
  estimated_proposed_regret = true_normal_mab_regret(proposed_policy, estimated_model, estimated_model, num_pulls, t, T,
                                                     mc_reps=5000)
  test_statistic = estimated_baseline_regret - estimated_proposed_regret
  if test_statistic > cutoff:
    return True
  else:
    return False





