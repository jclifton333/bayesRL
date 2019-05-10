"""
Hypothesis testing tuned regret against baseline for one-armed Gaussian bandit, i.e.
  \mu_0 known
  X_i \sim N(\mu_1, 1), \mu_1 unknown.

Actions code as
  0: known arm
  1: unknown arm
"""
import numpy as np


def true_regret(eta, policy, mu_0, mu_1, xbar, num_pulls, t, T):
  """
  Get the true regret (under mu_0 and mu_1) of given policy and a given initial xbar.

  :param eta:
  :param policy:
  :param mu_0:
  :param mu_1:
  :param xbar:
  :param num_pulls:
  :param t:
  :param T:
  :return:
  """
  regret = 0.0
  num_pulls_rep = num_pulls
  xbar_rollout = xbar
  for tprime in range(t, T):
    eta_tprime = eta(xbar_rollout, t, T)
    action = policy(xbar_rollout, eta_tprime)

    if action:
      x_tprime = np.random.normal(loc=mu_1)
      num_pulls_rep += 1
      xbar_rollout += (x_tprime - xbar_rollout) / num_pulls_rep

    regret += (mu_0 < mu_1)*(mu_1 - mu_0)*(1 - action) + (mu_0 > mu_1)*(mu_0 - mu_1)*action
  return regret


def regret_diff_sampling_dbn(eta_baseline, eta_hat, policy, mu_0, mu_1, num_pulls, t, T, mc_reps=1000):
  """
  Get sampling distribution of test statistic
    diff = R_t:T(eta_baseline, xbar) - R_t:T(eta_hat, xbar).

  :param eta: Tuning schedule
  :param policy: Exploration policy which takes xbar, eta as argument and returns action
  :param mu_0: Known arm mean
  :param mu_1: Unknown arm mean
  :param num_pulls: Number of pulls of the unknown arm
  :param t: Current time
  :param T: Time horizon
  :return:
  """
  diffs = []

  for rep in range(mc_reps):
    xbar = np.random.normal(loc=mu_1, scale=np.sqrt(1 / num_pulls))
    regret_eta_baseline = true_regret(eta_baseline, policy, mu_0, mu_1, xbar, num_pulls, t, T)
    regret_eta_hat = true_regret(eta_hat, policy, mu_0, mu_1, xbar, num_pulls, t, T)
    diffs.append(regret_eta_baseline - regret_eta_hat)

  return diffs


def rejection_rate(cutoff, eta_baseline, eta_hat, policy, mu_0, mu_1, num_pulls, t, T, mc_reps=1000):
  """
  Get rejection rate of the test that rejects when
    diff = R_t:T(eta_baseline, xbar) - R_t:T(eta_hat, xbar) > cutoff.

  :param eta_baseline:
  :param eta_hat:
  :param policy:
  :param mu_0:
  :param mu_1:
  :param num_pulls:
  :param t:
  :param T:
  :param mc_reps:
  :return:
  """
  sampling_dbn_of_diff = regret_diff_sampling_dbn(eta_baseline, eta_hat, policy, mu_0, mu_1, num_pulls, t, T,
                                                  mc_reps=1000)
  rejection_rate_ = np.mean(sampling_dbn_of_diff > cutoff)
  return rejection_rate_


def operating_characteristics(cutoff, eta_baseline, eta_hat, policy, mu_0, xbar, num_pulls, t, T, mc_reps=1000):
  """
  Compute power and type 1 error at a given cutoff.

  :param cutoff:
  :param eta_baseline:
  :param eta_hat:
  :param policy:
  :param mu_0:
  :param xbar:
  :param num_pulls:
  :param t:
  :param T:
  :param mc_reps:
  :return:
  """
  CANDIDATE_MU1 = np.linspace(mu_0 - 5, mu_0 + 5, 10)  # mu_1 vals to search over when computing operating chars.

  type_1_error = 0.0
  power = 1.0

  for mu_1 in CANDIDATE_MU1:
    # Decide if H0 or H1 is true
    regret_eta_baseline = true_regret(eta_baseline, policy, mu_0, mu_1, xbar, num_pulls, t, T)
    regret_eta_hat = true_regret(eta_hat, policy, mu_0, mu_1, xbar, num_pulls, t, T)

    if regret_eta_baseline <= regret_eta_hat:  # H0
      type_1_error_at_mu1 = rejection_rate(cutoff, eta_baseline, eta_hat, policy, mu_0, mu_1, num_pulls, t, T,
                                           mc_reps=mc_reps)
      type_1_error = np.max((type_1_error, type_1_error_at_mu1))
    else:  # H1
      power_at_mu1 = rejection_rate(cutoff, eta_baseline, eta_hat, policy, mu_0, mu_1, num_pulls, t, T,
                                    mc_reps=mc_reps)
      power = np.min((power, power_at_mu1))
  return {'type_1_error': type_1_error, 'power': power}


def operating_characteristics_curves(eta_baseline, eta_hat, policy, mu_0, xbar, num_pulls, t, T, mc_reps=1000):
  """
  Compute type 1 error and power as cutoff varies.

  :param eta_baseline:
  :param eta_hat:
  :param policy:
  :param mu_0:
  :param xbar:
  :param num_pulls:
  :param t:
  :param T:
  :param mc_reps:
  :return:
  """
  CUTOFFS = np.linspace(0.1, T-t, 20)

  for cutoff in CUTOFFS:
   operating_characteristics_ = \
     operating_characteristics(cutoff, eta_baseline, eta_hat, policy, mu_0, xbar, num_pulls, t, T, mc_reps=1000)


