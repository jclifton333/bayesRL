"""
Evaluate operating characteristics of hypothesis test at each step of one-armed bandit.
"""
import numpy as np
import one_armed_bandit as oab
import os
import yaml
import datetime


def optimal_simple_eps_fixed_policy(policy, mu_0, mu_1_conf_dbn, T, num_draws=100):
  best_regret = float('inf')
  best_theta = 0.01
  for theta in np.linspace(0.01, 0.2, 10):
    eta_hat_theta = lambda xbar_, t_start, t_, T_: theta
    regret_theta = 0.0
    for draw in range(num_draws):
      mu_1 = mu_1_conf_dbn()
      regret_theta_draw = oab.true_regret(eta_hat_theta, policy, mu_0, mu_1, 0.0, 0.0, 0, T, mc_reps=1)
      regret_theta += (regret_theta_draw - regret_theta) / (draw + 1)
    if regret_theta < best_regret:
      best_regret = regret_theta
      best_theta = theta
  return best_theta


def optimal_simple_eps_decay_policy(policy, mu_0, mu_1_conf_dbn, T, num_draws=100):
  """
  Optimize theta for epsilon schedules of the form theta / t using grid search.

  :return:
  """
  best_regret = float('inf')
  best_theta = 0.5
  for theta in np.linspace(0.5, 5, 10):
    eta_hat_theta = lambda xbar_, t_start, t_, T_: theta / (t_ + 1)
    regret_theta = 0.0
    for draw in range(num_draws):
      mu_1 = mu_1_conf_dbn()
      regret_theta_draw = oab.true_regret(eta_hat_theta, policy, mu_0, mu_1, 0.0, 0, T, mc_reps=1)
      regret_theta += (regret_theta_draw - regret_theta) / (draw + 1)
    if regret_theta < best_regret:
      best_regret = regret_theta
      best_theta = theta
  return best_theta


def online_oab_with_hypothesis_test(policy, baseline_exploration_schedule, alpha_schedule, mu_0=0.0, mu_1=1.0, T=50,
                                    sampling_dbn_draws=100):

  # Initial pull from unknown arm
  xbar = np.random.normal(loc=mu_1)
  num_pulls = 1

  alpha_schedule_lst = [float(alpha_schedule(v)) for v in range(T)]
  baseline_exploration_schedule_lst = [float(baseline_exploration_schedule(mu_1, v, None, T)) for v in range(T)]

  # Prepare to collect and save results
  settings = {'policy': policy.__name__, 'mu': [float(mu_0), float(mu_1)], 'T': T,
              'alpha_schedule': alpha_schedule_lst, 'baseline_exploration_schedule': baseline_exploration_schedule_lst}
  results_at_each_timestep = {'power': [], 'ph1': []}
  if not os.path.exists('oc-results'):
      os.makedirs('oc-results')
  prefix = 'oc-results/{}-T={}'.format(settings['policy'], T)
  suffix = datetime.datetime.now().strftime("%y%m%d_%H%M%S")
  results_fname = '{}_{}.yml'.format(prefix, suffix)

  for t in range(T):
    # Get action
    exploration_parameter = baseline_exploration_schedule(xbar, t, None, T
    a = policy(xbar, mu_0, exploration_parameter)

    # Observe reward
    if a == 1:
      num_pulls += 1
      x_t = np.random.normal(loc=mu_1)
      xbar += (x_t - xbar) / num_pulls

    # Estimate optimal tuning schedule
    def mu_1_conf_dbn():
      return np.random.normal(loc=xbar, scale=1/np.sqrt(num_pulls))

    best_exploration_parameter = optimal_simple_eps_fixed_policy(policy, mu_0, mu_1_conf_dbn, T)
    estimated_exploration_schedule = lambda xbar_, t_, tprime, T_: best_exploration_parameter

    # Hypothesis test operating characteristics
    alpha_t = alpha_schedule(t)

    # Range of mu_1's to consider
    candidate_mu1_lower = xbar - 1.96/np.sqrt(num_pulls)
    candidate_mu1_upper = xbar + 1.96/np.sqrt(num_pulls)
    candidate_mu1s = np.linspace(candidate_mu1_lower, candidate_mu1_upper, 10)

    # Sampling distributions of regret under each mu_1 and corresponding true regrets
    sampling_dbns_h0 = []
    sampling_dbns_h1 = []
    for mu_1_hypothesis in candidate_mu1s:
      sampling_dbn_mu_1 = \
        oab.regret_diff_sampling_dbn(baseline_exploration_schedule, estimated_exploration_schedule, policy, mu_0,
                                     mu_1_hypothesis, xbar, num_pulls, t, T,
                                     mc_reps=100)

      # Check if H0 obtains for this mu_1; if so, add to list
      regret_eta_baseline = oab.true_regret(baseline_exploration_schedule, policy, mu_0, mu_1_hypothesis, xbar,
                                            num_pulls, t, T, mc_reps=100)
      regret_eta_hat = oab.true_regret(estimated_exploration_schedule, policy, mu_0, mu_1_hypothesis, xbar, num_pulls,
                                       t, T, mc_reps=100)
      if regret_eta_hat >= regret_eta_baseline:
        sampling_dbns_h0.append(sampling_dbn_mu_1)
      else:
        sampling_dbns_h1.append(sampling_dbn_mu_1)

    # Operating characteristics
    cutoff = oab.uniform_empirical_cutoff(alpha_t, sampling_dbns_h0)
    max_t1error = np.max([np.mean(sd0 > cutoff) for sd0 in sampling_dbns_h0])
    mean_power = np.mean([np.mean(sd1 > cutoff) for sd1 in sampling_dbns_h1])

    # P(H0) and P(H1) (conditional on this set of candidate mu1s...)
    prob_h0 = len(sampling_dbns_h0) / len(candidate_mu1s)
    prob_h1 = len(sampling_dbns_h1) / len(candidate_mu1s)

    # Update results
    results_at_each_timestep['power'].append(mean_power)
    results_at_each_timestep['ph1'].append(prob_h1)

    print('t: {}\nt1error: {} power:{}\nph0: {} ph1: {}\n'.format(t, max_t1error, mean_power, prob_h0, prob_h1))

    # Save results
    results = {'settings': settings, 'results': results_at_each_timestep}
    with open(results_fname, 'w') as outfile:
      yaml.dump(results, outfile)

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





