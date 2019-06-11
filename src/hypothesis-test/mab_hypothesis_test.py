import numpy as np


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


def mab_hypothesis_test(baseline_policy, proposed_policy, xbar, true_model, estimated_model, num_pulls,
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
    baseline_regret = true_mab_regret(baseline_policy, true_model, sampled_model, num_pulls, t, T)
    proposed_regret = true_mab_regret(proposed_policy, true_model, sampled_model, num_pulls, t, T)
    baseline_policy_regrets.append(baseline_regret)
    proposed_policy_regrets.append(proposed_regret)
  diffs = np.array(baseline_policy_regrets) - np.array(proposed_policy_regrets)
  return diffs
