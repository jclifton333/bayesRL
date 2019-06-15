"""
Hypothesis testing tuned regret against baseline for one-armed Gaussian bandit, i.e.
  \mu_0 known
  X_i \sim N(\mu_1, 1), \mu_1 unknown.

Actions code as
  0: known arm
  1: unknown arm
"""
import numpy as np
import matplotlib.pyplot as plt
import pdb
import copy
from scipy.stats import norm


def true_regret(eta, policy, mu_0, mu_1, xbar, num_pulls, t, T, mc_reps=5000):
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
  regrets = []
  for rollout in range(mc_reps):
    regret = 0.0
    num_pulls_rep = copy.copy(num_pulls)
    xbar_rollout = copy.copy(xbar)
    for tprime in range(t, T):
      eta_tprime = eta(xbar_rollout, t, tprime, T)
      action = policy(xbar_rollout, mu_0, eta_tprime)

      if action:
        x_tprime = np.random.normal(loc=mu_1)
        num_pulls_rep += 1
        xbar_rollout += (x_tprime - xbar_rollout) / num_pulls_rep

      regret += (mu_0 < mu_1)*(mu_1 - mu_0)*(1 - action) + (mu_0 > mu_1)*(mu_0 - mu_1)*action
    regrets.append(regret)
  return np.mean(regrets)


def regret_diff_sampling_dbn(eta_baseline, eta_hat, policy, mu_0, mu_1, x_bar, num_pulls, t, T, mc_reps=100):
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
  regret_eta_baselines = []
  regret_eta_hats = []
  for rep in range(mc_reps):
    # Mimic sampling splitting
    xbar_model = np.random.normal(loc=mu_1, scale=np.sqrt(1 / num_pulls))
    regret_eta_baseline = true_regret(eta_baseline, policy, mu_0, xbar_model, x_bar, num_pulls, t, T)
    regret_eta_hat = true_regret(eta_hat, policy, mu_0, xbar_model, x_bar, num_pulls, t, T)
    regret_eta_baselines.append(regret_eta_baseline)
    regret_eta_hats.append(regret_eta_hat)
  diffs = np.array(regret_eta_baselines) - np.array(regret_eta_hats)
  return diffs


def empirical_cutoff(alpha, sampling_dbn):
  """
  Get cutoff corresponding to exceedence probability of alpha wrt sampling_dbn.

  :param alpha:
  :param sampling_dbn:
  :return:
  """
  empirical_cutoff_ = np.percentile(sampling_dbn, (1-alpha)*100)
  return empirical_cutoff_


def uniform_empirical_cutoff(alpha, sampling_dbns):
  """
  Get cutoff needed to control type1 error across mus corresponding to elements of sampling_dbns.
  :param alpha:
  :param sampling_dbns:
  :return:
  """
  cutoffs = [empirical_cutoff(alpha, sampling_dbn_) for sampling_dbn_ in sampling_dbns]
  return np.max(cutoffs)


def rejection_rate(cutoff, sampling_dbn_of_diff, eta_baseline, eta_hat, policy, mu_0, mu_1, num_pulls, t, T,
                   mc_reps=1000):
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
  rejection_rate_ = np.mean(sampling_dbn_of_diff > cutoff)
  return rejection_rate_


def operating_characteristics(cutoff, sampling_dbns, eta_baseline, eta_hat, policy, mu_0, xbar, num_pulls, t, T,
                              candidate_mu1s, mc_reps=1000):
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
  type_1_error = 0.0
  power = 0.0

  for sampling_dbn_, mu_1 in zip(sampling_dbns, candidate_mu1s):
    # Decide if H0 or H1 is true
    # ToDo: May not want to hold xbar fixed? (because mu_1 is varying!)
    regret_eta_baseline = true_regret(eta_baseline, policy, mu_0, mu_1, xbar, num_pulls, t, T)
    regret_eta_hat = true_regret(eta_hat, policy, mu_0, mu_1, xbar, num_pulls, t, T)

    if regret_eta_baseline <= regret_eta_hat:  # H0
      type_1_error_at_mu1 = rejection_rate(cutoff, sampling_dbn_, eta_baseline, eta_hat, policy, mu_0, mu_1, num_pulls,
                                           t, T, mc_reps=mc_reps)
      type_1_error = np.max((type_1_error, type_1_error_at_mu1))
    else:  # H1
      power_at_mu1 = rejection_rate(cutoff, sampling_dbn_, eta_baseline, eta_hat, policy, mu_0, mu_1, num_pulls, t, T,
                                    mc_reps=mc_reps)
      pdb.set_trace()
      power = np.max((power, power_at_mu1))
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
  CUTOFFS = np.linspace(0, 3, 30)
  CANDIDATE_MU1 = np.linspace(mu_0 - 5, mu_0 + 5, 10)  # mu_1 vals to search over when computing operating chars.
  powers = []
  type_1_errors = []
  sampling_dbns = []

  for mu_1 in CANDIDATE_MU1:
    sampling_dbn__mu_1 = \
      regret_diff_sampling_dbn(eta_baseline, eta_hat, policy, mu_0, mu_1, xbar, num_pulls, t, T, mc_reps=1000)
    sampling_dbns.append(sampling_dbn__mu_1)

  for cutoff in CUTOFFS:
    operating_characteristics_ = \
     operating_characteristics(cutoff, sampling_dbns, eta_baseline, eta_hat, policy, mu_0, xbar, num_pulls, t, T,
                               mc_reps=1000)
    powers.append(operating_characteristics_['power'])
    type_1_errors.append(operating_characteristics_['type_1_error'])

  return {'cutoffs': CUTOFFS, 'powers': powers, 'type_1_errors': type_1_errors}




if __name__ == "__main__":
  # Bandit settings
  mu_0 = 0.0
  mu_1 = 1.0
  T = 20
  mc_reps = 100

  # Policy settings
  eta_hat = lambda xbar_, t_start, t_, T_: 0.5 / (t_ - t_start + 1)
  eta_baseline = lambda xbar_, t_start, t_, T_: 1.0 / (t_ - t_start + 1)

  def eps_greedy_policy(xbar_, mu_0, eta_):
    if np.random.random() > eta_:
      return xbar_ > mu_0
    else:
      return np.random.choice(2)

  policy = eps_greedy_policy
  t_list = [10]
  num_pulls_list = [5]
  alphas_list = [0.05]

  powers = []
  type_1_errors = []
  for alpha, t, num_pulls in zip(alphas_list, t_list, num_pulls_list):
    t = int(t)
    xbar = np.random.normal(mu_1, scale=1 / np.sqrt(num_pulls))
    candidate_mu1_lower = xbar - 1.96/np.sqrt(num_pulls)
    candidate_mu1_upper = xbar + 1.96/np.sqrt(num_pulls)
    candidate_mu1s = np.linspace(candidate_mu1_lower, candidate_mu1_upper, 10)

    # Get sampling dbns
    sampling_dbns = []
    sampling_dbns_h0 = []
    sampling_dbns_h1 = []
    for mu_1_hypothesis in candidate_mu1s:
      print('sampling dbn for mu1={}'.format(mu_1_hypothesis))
      sampling_dbn_mu_1 = \
        regret_diff_sampling_dbn(eta_baseline, eta_hat, policy, mu_0, mu_1_hypothesis, xbar, num_pulls, t, T, mc_reps=10)
      normalized_sampling_dbn = sampling_dbn_mu_1
      sampling_dbns.append(normalized_sampling_dbn)

      # Check if H0 obtains for this mu_1; if so, add to list
      regret_eta_baseline = true_regret(eta_baseline, policy, mu_0, mu_1_hypothesis, xbar, num_pulls, t, T)
      regret_eta_hat = true_regret(eta_hat, policy, mu_0, mu_1_hypothesis, xbar, num_pulls, t, T)
      if regret_eta_hat >= regret_eta_baseline:
        sampling_dbns_h0.append(normalized_sampling_dbn)
      else:
        sampling_dbns_h1.append(normalized_sampling_dbn)

    if len(sampling_dbns_h0) > 0:
      cutoff = uniform_empirical_cutoff(alpha, sampling_dbns_h0)
    else:
      cutoff = 0.0

    # Operating characteristics
    mean_power = np.mean([np.mean(np.array(sd1) > cutoff) for sd1 in sampling_dbns_h1])
    max_t1error = np.max([np.mean(np.array(sd0) > cutoff) for sd0 in sampling_dbns_h0])

    powers.append(mean_power)
    type_1_errors.append(max_t1error)
    print('power: {}\nalpha: {}'.format(powers, type_1_errors))

  plt.scatter(t_list, powers, label='power')
  plt.scatter(t_list, type_1_errors, label='alpha')
  plt.scatter()

  # cutoffs = operating_characteristics_curves_['cutoffs']
  # powers = operating_characteristics_curves_['powers']
  # type_1_errors = operating_characteristics_curves_['type_1_errors']

  # # plt.scatter(type_1_errors, powers)
  # # plt.xlabel('alphas')
  # # plt.ylabel('powers')
  # plt.plot(cutoffs, powers, label='powers')
  # plt.plot(cutoffs, type_1_errors, label='alphas')
  # plt.show()





