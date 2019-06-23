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


def episode(label, baseline_schedule, alpha_schedule, std=0.1, list_of_reward_mus=[0.3,0.6], T=50,
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
  if test:
    NUM_CANDIDATE_HYPOTHESES = 1
    mc_reps_for_ht = 5
  else:
    NUM_CANDIDATE_HYPOTHESES = 20  # Number of candidate null models to consider when conducting ht
    mc_reps_for_ht = 100
  np.random.seed(label)

  # Settings
  # posterior_sample = False
  bootstrap_posterior = False
  positive_zeta = False
  baseline_tuning_function = lambda T, t, zeta: baseline_schedule[t]
  tuning_function = tuned_bandit.expit_epsilon_decay
  policy = tuned_bandit.mab_epsilon_greedy_policy
  ht_rejected = False
  tuning_function_parameter = np.array([0.05, 45, 2.5])
  bounds = {'zeta0': (0.05, 1.0), 'zeta1': (1.0, 49.0), 'zeta2': (0.01, 2.5)}
  explore_ = {'zeta0': [1.0, 0.05, 1.0, 0.1, 0.1, 0.05, 4.43802103],
              'zeta1': [50.0, 49.0, 1.0, 49.0, 1.0, 1.0,  85.04499728],
              'zeta2': [0.1, 2.5, 1.0, 2.5, 2.5, 2.5, 0.09655535]}
  posterior_sample = True
  when_hypothesis_rejected = float('inf')

  # Initialize environment
  env = NormalMAB(list_of_reward_mus=list_of_reward_mus, list_of_reward_vars=[std**2]*len(list_of_reward_mus))

  cumulative_regret = 0.0
  mu_opt = np.max(env.list_of_reward_mus)
  env.reset()
  tuning_parameter_sequence = []
  # Initial pulls
  for a in range(env.number_of_actions):
    env.step(a)

  estimated_means_list = []
  estimated_vars_list = []
  actions_list = []
  rewards_list = []
  for t in range(T):
    estimated_means_list.append([float(xbar) for xbar in env.estimated_means])
    estimated_vars_list.append([float(s) for s in env.estimated_vars])

    if not ht_rejected:  # Propose a tuned policy if ht has not already been rejected
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
      estimated_model = [[xbar, np.sqrt(sigma_sq_hat)] for xbar, sigma_sq_hat
                         in zip(env.estimated_means, env.estimated_vars)]

      def baseline_policy(means, standard_errors, num_pulls, tprime):
        return policy(means, standard_errors, num_pulls, baseline_tuning_function, None, T, tprime, None)

      def proposed_policy(means, standard_errors, num_pulls, tprime):
        return policy(means, standard_errors, num_pulls, tuning_function, tuning_function_parameter, T, tprime, None)

      true_model_list = []  # Construct list of candidate models by drawing from sampling dbn
      for draw in range(NUM_CANDIDATE_HYPOTHESES):
        sampled_model = env.sample_from_bootstrap()
        param_list_for_sampled_model = [[sampled_model[a]['mu_draw'], sampled_model[a]['var_draw']]
                                        for a in range(env.number_of_actions)]
        true_model_list.append(param_list_for_sampled_model)
      ht_rejected = ht.conduct_mab_ht(baseline_policy, proposed_policy, true_model_list, estimated_model,
                                      env.number_of_pulls, t, T, ht.normal_mab_sampling_dbn,
                                      alpha_schedule[t], ht.true_normal_mab_regret, ht.pre_generate_normal_mab_data,
                                      mc_reps=mc_reps_for_ht)
      if ht_rejected:
        when_hypothesis_rejected = int(t)

    if ht_rejected:
      action = policy(env.estimated_means, env.standard_errors, env.number_of_pulls, tuning_function,
                      tuning_function_parameter, T, t, env)
    else:
      action = policy(env.estimated_means, env.standard_errors, env.number_of_pulls, baseline_tuning_function,
                      None, T, t, env)

    res = env.step(action)
    u = res['Utility']
    actions_list.append(int(action))
    rewards_list.append(float(u))

    # Compute regret
    regret = mu_opt - env.list_of_reward_mus[action]
    cumulative_regret += regret

  return {'cumulative_regret': cumulative_regret, 'when_hypothesis_rejected': when_hypothesis_rejected,
          'baseline_schedule': baseline_schedule, 'alpha_schedule': alpha_schedule}


def run(std=0.1, list_of_reward_mus=[0.3,0.6], save=True, T=50, monte_carlo_reps=100, test=False):
  """

  :return:
  """
  POLICY_NAME = 'eps-greedy-ht'
  BASELINE_SCHEDULE = [0.1 for _ in range(T)]
  ALPHA_SCHEDULE = [0.05 for _ in range(T)]

  if test:
    replicates = num_cpus = 1
    T = 5
    monte_carlo_reps = 5
  else:
    replicates = 24
    num_cpus = 24

  pool = mp.Pool(processes=num_cpus)
  episode_partial = partial(episode, baseline_schedule=BASELINE_SCHEDULE, alpha_schedule=ALPHA_SCHEDULE,
                            std=std, T=T, monte_carlo_reps=monte_carlo_reps,
                            list_of_reward_mus=list_of_reward_mus, test=test)
  num_batches = int(replicates / num_cpus)

  results = []
  # for i in range(10):
  #   episode_partial(i)

  for batch in range(num_batches):
    results_for_batch = pool.map(episode_partial, range(batch*num_cpus, (batch+1)*num_cpus))
    results += results_for_batch

  # results = pool.map(episode_partial, range(replicates))
  cumulative_regret = [np.float(d['cumulative_regret']) for d in results]
  when_hypothesis_rejected = [d['when_hypothesis_rejected'] for d in results]
  # Save results
  if save:
    results = {'T': float(T), 'mean_regret': float(np.mean(cumulative_regret)),
               'std_regret': float(np.std(cumulative_regret)),
               'regret list': [float(r) for r in cumulative_regret], 'baseline_schedule': BASELINE_SCHEDULE,
               'alpha_schedule': ALPHA_SCHEDULE, 'when_hypothesis_rejected': when_hypothesis_rejected}

    base_name = \
      'normalmab-{}-numAct-{}'.format(POLICY_NAME, len(list_of_reward_mus))
    prefix = os.path.join(project_dir, 'src', 'run', base_name)
    suffix = datetime.datetime.now().strftime("%y%m%d_%H%M%S")
    filename = '{}_{}.yml'.format(prefix, suffix)
    with open(filename, 'w') as outfile:
      yaml.dump(results, outfile)

  return


if __name__ == "__main__":
  run(test=False)