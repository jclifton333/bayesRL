import sys
import pdb
import os
this_dir = os.path.dirname(os.path.abspath(__file__))
project_dir = os.path.join(this_dir, '..', '..')
sys.path.append(project_dir)

from src.policies import tuned_bandit_policies as tuned_bandit
from src.policies import gittins_index_policies as gittins
from src.policies import rollout
from src.environments.Bandit import NormalMAB
import src.policies.global_optimization as opt
from functools import partial
import datetime
import yaml
import multiprocessing as mp
import mab_hypothesis_test as ht
import numpy as np


def operating_chars_episode(label, policy_name, baseline_schedule, alpha_schedule, std=0.1,
                            list_of_reward_mus=[0.3,0.6], T=50, monte_carlo_reps=1000, bias_only=False,
                            test=False):
  """
  Run an episode only until H0 is rejected; collect operating chars.

  :param label:
  :param policy_name:
  :param baseline_schedule:
  :param alpha_schedule:
  :param std:
  :param list_of_reward_mus:
  :param T:
  :param monte_carlo_reps:
  :param bias_only: if True, only check bias of regret diff estimate at end of trial.
  :param test:
  :return:
  """
  TUNE_INTERVAL = 10
  np.random.seed(label)

  if test:
    NUM_CANDIDATE_HYPOTHESES = 5
    mc_reps_for_ht = 500
  else:
    NUM_CANDIDATE_HYPOTHESES = 100  # Number of candidate null models to consider when conducting ht
    mc_reps_for_ht = 1000
  # np.random.seed(label)

  # Settings
  positive_zeta = False
  baseline_tuning_function = lambda T, t, zeta: baseline_schedule[t]
  tuning_function = tuned_bandit.expit_epsilon_decay
  policy = tuned_bandit.mab_epsilon_greedy_policy
  if policy_name == 'baseline':
    tune = False
  else:
    tune = True
  ht_rejected = False
  no_rejections_yet = True
  tuning_function_parameter = np.array([0.05, 45, 2.5])
  bounds = {'zeta0': (0.05, 1.0), 'zeta1': (1.0, 49.0), 'zeta2': (0.01, 2.5)}
  explore_ = {'zeta0': [1.0, 0.05, 1.0, 0.1, 0.1, 0.05, 4.43802103],
              'zeta1': [50.0, 49.0, 1.0, 49.0, 1.0, 1.0,  85.04499728],
              'zeta2': [0.1, 2.5, 1.0, 2.5, 2.5, 2.5, 0.09655535]}
  posterior_sample = True
  when_hypothesis_rejected = float('inf')

  # Initialize environment
  env = NormalMAB(list_of_reward_mus=list_of_reward_mus, list_of_reward_vars=[std**2]*len(list_of_reward_mus))
  true_model_params = [[mu, np.sqrt(var)] for mu, var in zip(env.list_of_reward_mus, env.list_of_reward_vars)]

  cumulative_regret = 0.0
  mu_opt = np.max(env.list_of_reward_mus)
  env.reset()
  tuning_parameter_sequence = []

  # For IPW model estimates
  action_probs = np.array([])
  rewards = np.array([])
  actions = np.array([])

  # Initial pulls
  for a in range(env.number_of_actions):
    r = env.step(a)['Utility']
    action_probs = np.append(action_probs, 1)
    rewards = np.append(rewards, r)
    actions = np.append(actions, a)

  estimated_means_list = []
  estimated_vars_list = []
  t1_error = None
  t2_errors = []
  alpha_at_rejection = None
  alphas_at_non_rejections = []
  true_diffs = []
  test_statistics = []

  for t in range(T):
    estimated_means_list, estimated_stds_list = ht.ipw(env.number_of_actions, actions, action_probs, rewards)
    estimated_model = [[mu, s] for mu, s in zip(estimated_means_list, estimated_stds_list)]

    def baseline_policy(means, standard_errors, num_pulls, tprime):
      return policy(means, standard_errors, num_pulls, baseline_tuning_function, None, T, tprime, None)

    def proposed_policy(means, standard_errors, num_pulls, tprime):
      return policy(means, standard_errors, num_pulls, tuning_function, tuning_function_parameter, T, tprime, None)

    true_model_list = []  # Construct list of candidate models by drawing from sampling dbn
    for draw in range(NUM_CANDIDATE_HYPOTHESES):
      sampled_model = env.sample_from_posterior()
      param_list_for_sampled_model = [[sampled_model[a]['mu_draw'], np.sqrt(sampled_model[a]['var_draw'])]
                                      for a in range(env.number_of_actions)]
      true_model_list.append(param_list_for_sampled_model)

    # Get true regret of baseline
    h0_true, true_diff = ht.is_h0_true(baseline_policy, proposed_policy, estimated_model, env.number_of_pulls, t, T,
                                       ht.true_normal_mab_regret, ht.pre_generate_normal_mab_data, true_model_params,
                                       inner_loop_mc_reps=mc_reps_for_ht)

    time_to_tune = (tune and t > 0 and t % TUNE_INTERVAL == 0)
    # Propose a tuned policy if ht has not already been rejected
    if (time_to_tune and not ht_rejected and not bias_only) or bias_only:
      if posterior_sample:
        reward_means = []
        reward_vars = []
        for rep in range(monte_carlo_reps):
          draws = env.sample_from_posterior()
          means_for_each_action = []
          vars_for_each_action = []
          for a in range(env.number_of_actions):
            mean_a = draws[a]['mu_draw']
            var_a = draws[a]['var_draw']
            means_for_each_action.append(mean_a)
            vars_for_each_action.append(var_a)
          reward_means.append(means_for_each_action)
          reward_vars.append(vars_for_each_action)
      else:
        reward_means = None
        reward_vars = None

      sim_env = NormalMAB(list_of_reward_mus=env.estimated_means, list_of_reward_vars=env.estimated_vars)
      pre_simulated_data = sim_env.generate_mc_samples(monte_carlo_reps, T, reward_means=reward_means,
                                                       reward_vars=reward_vars)
      tuning_function_parameter = opt.bayesopt(rollout.mab_rollout_with_fixed_simulations, policy, tuning_function,
                                               tuning_function_parameter, T, env, monte_carlo_reps,
                                               {'pre_simulated_data': pre_simulated_data},
                                               bounds, explore_, positive_zeta=positive_zeta, test=test)
      tuning_parameter_sequence.append([float(z) for z in tuning_function_parameter])

      # Test regret of baseline vs tuned schedule
      ht_rejected, test_statistic = ht.conduct_mab_ht(baseline_policy, proposed_policy, true_model_list,
                                                      estimated_model, env.number_of_pulls, t, T,
                                                      ht.normal_mab_sampling_dbn,
                                                      alpha_schedule[t], ht.true_normal_mab_regret,
                                                      ht.pre_generate_normal_mab_data, env.number_of_actions,
                                                      actions, action_probs, rewards, mc_reps=mc_reps_for_ht)
      print(test_statistic, true_diff, t)
      test_statistics.append(float(test_statistic))
      true_diffs.append(float(true_diff))

      if ht_rejected and no_rejections_yet:
        when_hypothesis_rejected = int(t)
        no_rejections_yet = False

    if ht_rejected and tune:
      action, action_prob = policy(env.estimated_means, env.standard_errors, env.number_of_pulls, tuning_function,
                                    tuning_function_parameter, T, t, env)
    else:
      action, action_prob = policy(env.estimated_means, env.standard_errors, env.number_of_pulls,
                                   baseline_tuning_function, None, T, t, env)

    # Take step and update obs for computing IPW
    r = env.step(action)['Utility']
    action_probs = np.append(action_probs, action_prob)
    rewards = np.append(rewards, r)
    actions = np.append(actions, action)

    if ht_rejected:
      alpha_at_rejection = float(alpha_schedule[t])
      t1_error = int(h0_true)
      if not bias_only:
        break
    else:
      alphas_at_non_rejections.append(float(alpha_schedule[t]))
      t2_errors.append(int(1-h0_true))

  return {'when_hypothesis_rejected': when_hypothesis_rejected,
          'baseline_schedule': baseline_schedule, 'alpha_schedule': alpha_schedule, 'type1': t1_error,
          'type2': t2_errors, 'alpha_at_rejection': alpha_at_rejection,
          'alphas_at_non_rejections': alphas_at_non_rejections, 'true_diffs': true_diffs,
          'test_statistics': test_statistics}


def episode(label, policy_name, baseline_schedule, alpha_schedule, std=0.1, list_of_reward_mus=[0.3,0.6], T=50,
            monte_carlo_reps=1000, test=False):
  """
  Currently assuming eps-greedy.

  :param label:
  :param std:
  :param list_of_reward_mus:
  :param T:
  :param monte_carlo_reps:
  :param posterior_sample:
  :return:
  """
  TUNE_INTERVAL = 10
  np.random.seed(label)

  if test:
    NUM_CANDIDATE_HYPOTHESES = 5
    mc_reps_for_ht = 5
  else:
    NUM_CANDIDATE_HYPOTHESES = 100  # Number of candidate null models to consider when conducting ht
    mc_reps_for_ht = 500
  # np.random.seed(label)

  # Settings
  # posterior_sample = False
  bootstrap_posterior = False
  positive_zeta = False
  baseline_tuning_function = lambda T, t, zeta: baseline_schedule[t]
  tuning_function = tuned_bandit.expit_epsilon_decay
  policy = tuned_bandit.mab_epsilon_greedy_policy
  if policy_name == 'baseline':
    tune = False
  else:
    tune = True
  ht_rejected = False
  no_rejections_yet = True
  tuning_function_parameter = np.array([0.05, 45, 2.5])
  bounds = {'zeta0': (0.05, 1.0), 'zeta1': (1.0, 49.0), 'zeta2': (0.01, 2.5)}
  explore_ = {'zeta0': [1.0, 0.05, 1.0, 0.1, 0.1, 0.05, 4.43802103],
              'zeta1': [50.0, 49.0, 1.0, 49.0, 1.0, 1.0,  85.04499728],
              'zeta2': [0.1, 2.5, 1.0, 2.5, 2.5, 2.5, 0.09655535]}
  posterior_sample = True
  when_hypothesis_rejected = float('inf')

  # Initialize environment
  env = NormalMAB(list_of_reward_mus=list_of_reward_mus, list_of_reward_vars=[std**2]*len(list_of_reward_mus))
  true_model_params = [[mu, np.sqrt(var)] for mu, var in zip(env.list_of_reward_mus, env.list_of_reward_vars)]

  cumulative_regret = 0.0
  mu_opt = np.max(env.list_of_reward_mus)
  env.reset()
  tuning_parameter_sequence = []

  # For IPW model estimates
  action_probs = np.array([])
  rewards = np.array([])
  actions = np.array([])

  # Initial pulls
  for a in range(env.number_of_actions):
    r = env.step(a)['Utility']
    action_probs = np.append(action_probs, 1)
    rewards = np.append(rewards, r)
    actions = np.append(actions, a)

  estimated_means_list = []
  estimated_vars_list = []
  actions_list = []
  rewards_list = []
  t1_errors = []
  powers = []
  for t in range(T):
    estimated_means_list, estimated_stds_list = ht.ipw(env.number_of_actions, actions, action_probs, rewards)
    estimated_model = [[mu, s] for mu, s in zip(estimated_means_list, estimated_stds_list)]

    # Stuff needed for hypothesis test / operating chars
    estimated_model = [[xbar, np.sqrt(sigma_sq_hat)] for xbar, sigma_sq_hat
                       in zip(env.estimated_means, env.estimated_vars)]

    def baseline_policy(means, standard_errors, num_pulls, tprime):
      return policy(means, standard_errors, num_pulls, baseline_tuning_function, None, T, tprime, None)

    def proposed_policy(means, standard_errors, num_pulls, tprime):
      return policy(means, standard_errors, num_pulls, tuning_function, tuning_function_parameter, T, tprime, None)

    true_model_list = []  # Construct list of candidate models by drawing from sampling dbn
    for draw in range(NUM_CANDIDATE_HYPOTHESES):
      sampled_model = env.sample_from_posterior()
      param_list_for_sampled_model = [[sampled_model[a]['mu_draw'], np.sqrt(sampled_model[a]['var_draw'])]
                                      for a in range(env.number_of_actions)]
      true_model_list.append(param_list_for_sampled_model)

    time_to_tune = (tune and t > 0 and t % TUNE_INTERVAL == 0)
    if time_to_tune and not ht_rejected:  # Propose a tuned policy if ht has not already been rejected
      if posterior_sample:
        reward_means = []
        reward_vars = []
        for rep in range(monte_carlo_reps):
          if bootstrap_posterior:
            draws = env.sample_from_bootstrap()
          else:
            draws = env.sample_from_posterior()
          means_for_each_action = []
          vars_for_each_action = []
          for a in range(env.number_of_actions):
            mean_a = draws[a]['mu_draw']
            var_a = draws[a]['var_draw']
            means_for_each_action.append(mean_a)
            vars_for_each_action.append(var_a)
          reward_means.append(means_for_each_action)
          reward_vars.append(vars_for_each_action)
      else:
        reward_means = None
        reward_vars = None

      sim_env = NormalMAB(list_of_reward_mus=env.estimated_means, list_of_reward_vars=env.estimated_vars)
      pre_simulated_data = sim_env.generate_mc_samples(monte_carlo_reps, T, reward_means=reward_means,
                                                       reward_vars=reward_vars)
      tuning_function_parameter = opt.bayesopt(rollout.mab_rollout_with_fixed_simulations, policy, tuning_function,
                                               tuning_function_parameter, T, env, monte_carlo_reps,
                                               {'pre_simulated_data': pre_simulated_data},
                                               bounds, explore_, positive_zeta=positive_zeta, test=test)
      tuning_parameter_sequence.append([float(z) for z in tuning_function_parameter])

      # Conduct hypothesis test
      ht_rejected, test_statistic = ht.conduct_mab_ht(baseline_policy, proposed_policy, true_model_list,
                                                      estimated_model, env.number_of_pulls, t, T,
                                                      ht.normal_mab_sampling_dbn,
                                                      alpha_schedule[t], ht.true_normal_mab_regret,
                                                      ht.pre_generate_normal_mab_data, env.number_of_actions,
                                                      actions, action_probs, rewards, mc_reps=mc_reps_for_ht)
      if ht_rejected and no_rejections_yet:
        when_hypothesis_rejected = int(t)
        no_rejections_yet = False

    if ht_rejected and tune:
      action, action_prob = policy(env.estimated_means, env.standard_errors, env.number_of_pulls, tuning_function,
                                    tuning_function_parameter, T, t, env)
    else:
      action, action_prob = policy(env.estimated_means, env.standard_errors, env.number_of_pulls,
                                   baseline_tuning_function, None, T, t, env)

    # Take step and update obs for computing IPW
    r = env.step(action)['Utility']
    action_probs = np.append(action_probs, action_prob)
    rewards = np.append(rewards, r)
    actions = np.append(actions, action)

    # Compute regret
    regret = mu_opt - env.list_of_reward_mus[action]
    cumulative_regret += regret

  return {'cumulative_regret': cumulative_regret, 'when_hypothesis_rejected': when_hypothesis_rejected,
          'baseline_schedule': baseline_schedule, 'alpha_schedule': alpha_schedule, 'type1': t1_errors,
          'power': powers}


def operating_chars_run(label, policy_name, std=0.1, list_of_reward_mus=[0.3,0.6], save=True, T=10,
                        monte_carlo_reps=100, bias_only=False, test=False):
  BASELINE_SCHEDULE = [0.1 for _ in range(T)]
  ALPHA_SCHEDULE = [float(0.5 / (T - t)) for t in range(T)]

  if test:
    replicates = num_cpus = 1
    T = 40
    monte_carlo_reps = 5
  else:
    replicates = 48
    num_cpus = 48

  pool = mp.Pool(processes=num_cpus)
  episode_partial = partial(operating_chars_episode, policy_name=policy_name, baseline_schedule=BASELINE_SCHEDULE,
                            alpha_schedule=ALPHA_SCHEDULE, std=std, T=T, monte_carlo_reps=monte_carlo_reps,
                            list_of_reward_mus=list_of_reward_mus, bias_only=bias_only, test=test)
  num_batches = int(replicates / num_cpus)

  results = []
  if test:
    results.append(episode_partial(0))
  else:
    for batch in range(label*num_batches, (label+1)*num_batches):
      results_for_batch = pool.map(episode_partial, range(batch*num_cpus, (batch+1)*num_cpus))
      results += results_for_batch

  t1_errors = np.array([d['type1'] for d in results])
  nominal_rejection_alphas = np.array([d['alpha_at_rejection'] for d in results])
  t2_errors = np.hstack([d['type2'] for d in results])
  nominal_accept_alphas = np.hstack([d['alphas_at_non_rejections'] for d in results])
  test_statistics = np.hstack([d['test_statistics'] for d in results])
  true_diffs = np.hstack([d['true_diffs'] for d in results])

  return t1_errors, nominal_rejection_alphas, t2_errors, nominal_accept_alphas, test_statistics, true_diffs


def run(label, policy_name, std=0.1, list_of_reward_mus=[0.3,0.6], save=True, T=10, monte_carlo_reps=100, test=False):
  """

  :return:
  """
  BASELINE_SCHEDULE = [0.1 for _ in range(T)]
  ALPHA_SCHEDULE = [float(0.5 / (T - t)) for t in range(T)]

  if test:
    replicates = num_cpus = 1
    T = 40
    monte_carlo_reps = 5
  else:
    replicates = 48*8
    num_cpus = 48

  episode_partial = partial(episode, policy_name=policy_name, baseline_schedule=BASELINE_SCHEDULE,
                            alpha_schedule=ALPHA_SCHEDULE, std=std, T=T, monte_carlo_reps=monte_carlo_reps,
                            list_of_reward_mus=list_of_reward_mus, test=test)

  if test:
    results = [episode_partial(0)]
  else:
    pool = mp.Pool(processes=num_cpus)
    num_batches = int(replicates / num_cpus)
    results = []
    for batch in range(label*num_batches, (label+1)*num_batches):
      results_for_batch = pool.map(episode_partial, range(batch*num_cpus, (batch+1)*num_cpus))
      results += results_for_batch

  # results = pool.map(episode_partial, range(replicates))
  cumulative_regret = [np.float(d['cumulative_regret']) for d in results]
  when_hypothesis_rejected = [d['when_hypothesis_rejected'] for d in results]
  type1_errors = [d['type1'] for d in results]
  powers = [d['power'] for d in results]
  # Save results
  if save and not test:
    results = {'T': float(T), 'mean_regret': float(np.mean(cumulative_regret)),
               'std_regret': float(np.std(cumulative_regret)),
               'regret list': [float(r) for r in cumulative_regret], 'baseline_schedule': BASELINE_SCHEDULE,
               'alpha_schedule': ALPHA_SCHEDULE, 'when_hypothesis_rejected': when_hypothesis_rejected,
               'type1': type1_errors, 'powers': powers, 'std': std}

    base_name = \
      'normalmab-{}-numAct-{}'.format(policy_name, len(list_of_reward_mus))
    prefix = os.path.join(project_dir, 'src', 'run', base_name)
    suffix = datetime.datetime.now().strftime("%y%m%d_%H%M%S")
    filename = '{}_{}.yml'.format(prefix, suffix)
    with open(filename, 'w') as outfile:
      yaml.dump(results, outfile)

  return


if __name__ == "__main__":
  list_of_reward_mus_10 = [0.74, 0.15, 0.34, 0.48, 0.53, 0.23, 0.47, 0.51, 0.71, 0.42]
  list_of_reward_mus_5 = [0.73, 0.56, 0.33, 0.04, 0.66]
  run(0, 'eps_decay', T=50, list_of_reward_mus=list_of_reward_mus_5, test=False)
  run(0, 'eps_decay', T=50, list_of_reward_mus=list_of_reward_mus_5, std=1.0, test=False)
