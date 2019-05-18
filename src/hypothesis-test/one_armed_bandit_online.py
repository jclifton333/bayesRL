"""
Evaluate operating characteristics of hypothesis test at each step of one-armed bandit.
"""
import numpy as np
import one_armed_bandit as oab


def optimal_simple_eps_fixed_policy(policy, mu_0, mu_1, T):
  best_regret = float('inf')
  best_theta = 0.5
  for theta in np.linspace(0.01, 0.2, 10):
    eta_hat_theta = lambda xbar_, t_start, t_, T_: theta
    regret_theta = oab.true_regret(eta_hat_theta, policy, mu_0, mu_1, 0.0, 0, 0, T)
    if regret_theta < best_regret:
      best_regret = regret_theta
      best_theta = theta
  return best_theta


def optimal_simple_eps_decay_policy(policy, mu_0, mu_1, T):
  """
  Optimize theta for epsilon schedules of the form theta / t using grid search.

  :return:
  """
  best_regret = float('inf')
  best_theta = 0.5
  for theta in np.linspace(0.5, 5, 10):
    eta_hat_theta = lambda xbar_, t_start, t_, T_: theta / (t_ + 1)
    regret_theta = oab.true_regret(eta_hat_theta, policy, mu_0, mu_1, 0.0, 0, 0, T)
    if regret_theta < best_regret:
      best_regret = regret_theta
      best_theta = theta
  return best_theta


def online_oab_with_hypothesis_test(policy, baseline_exploration_schedule, alpha_schedule, mu_0=0.0, mu_1=1.0, T=50,
                                    sampling_dbn_draws=100):

  # Initial pull from unknown arm
  xbar = np.random.normal(loc=mu_1)
  num_pulls = 1

  for t in range(T):
    # Get action
    exploration_parameter = baseline_exploration_schedule(xbar, t, None, T)
    a = policy(xbar, mu_0, exploration_parameter)

    # Observe reward
    if a == 1:
      num_pulls += 1
      x_t = np.random.normal(loc=mu_1)
      xbar += (x_t - xbar) / num_pulls

    # Estimate optimal tuning schedule
    best_exploration_parameter = optimal_simple_eps_fixed_policy(policy, mu_0, xbar, T)
    estimated_exploration_schedule = lambda xbar_, t_, tprime, T_: best_exploration_parameter
    print(t)

    # Hypothesis test operating characteristics
    alpha_t = alpha_schedule(t)

    # Range of mu_1's to consider
    candidate_mu1_lower = xbar - 1.96/np.sqrt(num_pulls)
    candidate_mu1_upper = xbar + 1.96/np.sqrt(num_pulls)
    candidate_mu1s = np.linspace(candidate_mu1_lower, candidate_mu1_upper, 10)

    # Sampling distributions of regret under each mu_1 and corresponding true regrets
    sampling_dbns_h0 = []
    sampling_dbns_h1 = []
    cutoff = 0.0
    for mu_1_hypothesis in candidate_mu1s:
      sampling_dbn_mu_1 = \
        oab.regret_diff_sampling_dbn(baseline_exploration_schedule, estimated_exploration_schedule, policy, mu_0,
                                     mu_1_hypothesis, xbar, num_pulls, t, T,
                                     mc_reps=10)

      # Check if H0 obtains for this mu_1; if so, add to list
      regret_eta_baseline = oab.true_regret(baseline_exploration_schedule, policy, mu_0, mu_1_hypothesis, xbar,
                                            num_pulls, t, T)
      regret_eta_hat = oab.true_regret(estimated_exploration_schedule, policy, mu_0, mu_1_hypothesis, xbar, num_pulls,
                                       t, T)

      print('mu1 hyp: {} regret eta hat: {} regret eta baseline: {}'.format(mu_1_hypothesis, regret_eta_hat, regret_eta_baseline))
      if regret_eta_hat >= regret_eta_baseline:
        sampling_dbns_h0.append(sampling_dbn_mu_1)
        # Update cutoff value to ensure t1error bound of alpha
        cutoff_at_mu1 = oab.uniform_empirical_cutoff(alpha_t, sampling_dbn_mu_1)
        cutoff = np.max((cutoff, cutoff_at_mu1))
      else:
        sampling_dbns_h1.append(sampling_dbn_mu_1)

    # Operating characteristics
    max_t1error = np.max([np.mean(sd0 > cutoff) for sd0 in sampling_dbns_h0])
    mean_power = np.mean([np.mean(sd1 > cutoff) for sd1 in sampling_dbns_h1])

    # P(H0) and P(H1) (conditional on this set of candidate mu1s...)
    prob_h0 = len(sampling_dbns_h0) / len(candidate_mu1s)
    prob_h1 = len(sampling_dbns_h1) / len(candidate_mu1s)

    print('t: {}\nt1error: {} power:{}\nph0: {} ph1: {}\n'.format(t, max_t1error, mean_power, prob_h0, prob_h1))

  return


if __name__ == "__main__":
  # Sim settings
  T = 20
  sampling_dbn_draws = 100

  def policy(xbar_, mu0_, eps_):
    if np.random.random() < eps_:
      return np.random.choice(2)
    else:
      return xbar_ > mu0_

  def baseline_exploration_schedule(xbar_, t_, tprime, T_):
    return 0.05

  def alpha_schedule(t_):
    return 0.05

  # Run
  online_oab_with_hypothesis_test(policy, baseline_exploration_schedule, alpha_schedule, mu_0=0.0, mu_1=1.0, T=T,
                                  sampling_dbn_draws=sampling_dbn_draws)





