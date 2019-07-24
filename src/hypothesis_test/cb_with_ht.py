import sys
import pdb
import os
this_dir = os.path.dirname(os.path.abspath(__file__))
project_dir = os.path.join(this_dir, '..', '..')
sys.path.append(project_dir)

from src.policies import tuned_bandit_policies as tuned_bandit
from src.policies import rollout
from src.environments.Bandit import LinearCB, NormalCB
import src.policies.global_optimization as opt
from functools import partial
import datetime
import yaml
import multiprocessing as mp
import cb_hypothesis_test as ht
import numpy as np


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
  if test:
    NUM_CANDIDATE_HYPOTHESES = 5
    mc_reps_for_ht = 5
  else:
    NUM_CANDIDATE_HYPOTHESES = 20  # Number of candidate null models to consider when conducting ht
    mc_reps_for_ht = 500
  np.random.seed(label)

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
  # ToDo: implement heteroskedastic CB; current assuming attribute list_of_reward_thetas
  env = NormalCB(num_initial_pulls=1)
  true_model_params = [[beta, theta] for beta, theta in zip(env.list_of_reward_betas, env.list_of_reward_thetas)]

  cumulative_regret = 0.0
  mu_opt = np.max(env.list_of_reward_mus)
  env.reset()
  tuning_parameter_sequence = []
  # Initial pulls
  for a in range(env.number_of_actions):
    env.step(a)

  t1_errors = []
  powers = []
  for t in range(T):
    # Stuff needed for hypothesis test / operating chars
    # ToDo: assuming env.theta_hat_list
    estimated_model = [[beta_hat, theta_hat, XprimeX_inv, X, y] for beta_hat, theta_hat, XprimeX_inv, X, y
                       in zip(env.beta_hat_list, env.theta_hat_list, env.Xprime_X_inv_list,
                              env.X_list, env.y_list)]

    def baseline_policy(means, standard_errors, num_pulls, tprime):
      return policy(means, standard_errors, num_pulls, baseline_tuning_function, None, T, tprime, None)

    def proposed_policy(means, standard_errors, num_pulls, tprime):
      return policy(means, standard_errors, num_pulls, tuning_function, tuning_function_parameter, T, tprime, None)

    true_model_list = []  # Construct list of candidate models by drawing from sampling dbn
    for draw in range(NUM_CANDIDATE_HYPOTHESES):
      sampled_model = env.sample_from_posterior()
      # ToDo: not sure if this makes sense
      param_list_for_sampled_model = [[sampled_model[a]['beta_draw'], np.sqrt(sampled_model[a]['theta_draw'])]
                                      + estimated_model[a][2:]
                                      for a in range(env.number_of_actions)]
      true_model_list.append(param_list_for_sampled_model)

    # Get operating characteristics
    if not ht_rejected:
      operating_char_dict = ht.cb_ht_operating_characteristics(baseline_policy, proposed_policy, true_model_list,
                                                                estimated_model, env.number_of_pulls,
                                                                t, T, ht.normal_mab_sampling_dbn, alpha_schedule[t],
                                                                ht.true_normal_mab_regret,
                                                                ht.pre_generate_normal_mab_data, true_model_params,
                                                                context_dbn_sampler, feature_function,
                                                                inner_loop_mc_reps=100, outer_loop_mc_reps=100)

    if tune and not ht_rejected:  # Propose a tuned policy if ht has not already been rejected
      gen_model_parameters = []
      reward_means = []
      reward_vars = []
      for rep in range(monte_carlo_reps):
        draws = env.sample_from_posterior()
        betas_for_each_action = []
        thetas_for_each_action = []
        for a in range(env.number_of_actions):
          beta_a = draws[a]['beta_draw']
          theta_a = draws[a]['theta_draw']
          betas_for_each_action.append(beta_a)
          thetas_for_each_action.append(theta_a)
          param_dict = {'reward_betas': betas_for_each_action, 'reward_thetas': thetas_for_each_action,
                        'context_mean': draws['context_mu_draw'], 'context_var': draws['context_var_draw']}
#                          'context_max': draws['context_max']}
          gen_model_parameters.append(param_dict)

      sim_env = NormalCB(list_of_reward_betas=list_of_reward_betas, context_mean=context_mean, context_var=context_var,
                         list_of_reward_vars=list_of_reward_vars)
      pre_simulated_data = sim_env.generate_mc_samples(mc_replicates, T, n_patients=n_patients,
                                                       gen_model_params=gen_model_parameters)
      tuning_function_parameter = opt.bayesopt(rollout.normal_cb_rollout_with_fixed_simulations, policy,
                                               tuning_function, tuning_function_parameter, T,
                                               sim_env, mc_replicates,
                                               {'pre_simulated_data': pre_simulated_data},
                                               bounds, explore_, positive_zeta=positive_zeta)
      # Hypothesis testing
      ht_rejected = ht.conduct_cb_ht(baseline_policy, proposed_policy, true_model_list, estimated_model,
                                      env.number_of_pulls, t, T, ht.cb_sampling_dbn,
                                      alpha_schedule[t], ht.true_cb_regret, ht.pre_generate_cb_data,
                                      context_dbn_sampler, feature_function, mc_reps=mc_reps_for_ht)
      if ht_rejected and no_rejections_yet:
        when_hypothesis_rejected = int(t)
        no_rejections_yet = False

    if ht_rejected and tune:
      action = policy(env.estimated_means, env.standard_errors, env.number_of_pulls, tuning_function,
                      tuning_function_parameter, T, t, env)
    else:
      action = policy(env.estimated_means, env.standard_errors, env.number_of_pulls, baseline_tuning_function,
                      None, T, t, env)

    t1_errors.append(operating_char_dict['type1'])
    powers.append(operating_char_dict['type2'])

    res = env.step(action)
    u = res['Utility']
    actions_list.append(int(action))
    rewards_list.append(float(u))

    # Compute regret
    regret = mu_opt - env.list_of_reward_mus[action]
    cumulative_regret += regret

  return {'cumulative_regret': cumulative_regret, 'when_hypothesis_rejected': when_hypothesis_rejected,
          'baseline_schedule': baseline_schedule, 'alpha_schedule': alpha_schedule, 'type1': t1_errors,
          'power': powers}


def run(policy_name, std=0.1, list_of_reward_mus=[0.3,0.6], save=True, T=10, monte_carlo_reps=100, test=False):
  """

  :return:
  """
  BASELINE_SCHEDULE = [0.1 for _ in range(T)]
  ALPHA_SCHEDULE = [float(0.5 / (T - t)) for t in range(T)]

  if test:
    replicates = num_cpus = 1
    T = 5
    monte_carlo_reps = 5
  else:
    replicates = 48
    num_cpus = 48

  pool = mp.Pool(processes=num_cpus)
  episode_partial = partial(episode, policy_name=policy_name, baseline_schedule=BASELINE_SCHEDULE,
                            alpha_schedule=ALPHA_SCHEDULE, std=std, T=T, monte_carlo_reps=monte_carlo_reps,
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
  type1_errors = [d['type1'] for d in results]
  powers = [d['power'] for d in results]
  # Save results
  if save:
    results = {'T': float(T), 'mean_regret': float(np.mean(cumulative_regret)),
               'std_regret': float(np.std(cumulative_regret)),
               'regret list': [float(r) for r in cumulative_regret], 'baseline_schedule': BASELINE_SCHEDULE,
               'alpha_schedule': ALPHA_SCHEDULE, 'when_hypothesis_rejected': when_hypothesis_rejected,
               'type1': type1_errors, 'powers': powers}

    base_name = \
      'normalmab-{}-numAct-{}'.format(policy_name, len(list_of_reward_mus))
    prefix = os.path.join(project_dir, 'src', 'run', base_name)
    suffix = datetime.datetime.now().strftime("%y%m%d_%H%M%S")
    filename = '{}_{}.yml'.format(prefix, suffix)
    with open(filename, 'w') as outfile:
      yaml.dump(results, outfile)

  return


if __name__ == "__main__":
  # std = 0.1
  # list_of_reward_mus = [0.3, 0.6]
  # save = False
  # T = 50
  # monte_carlo_reps = 100
  # test = True
  # BASELINE_SCHEDULE = [0.1 for _ in range(T)]
  # ALPHA_SCHEDULE = [0.05 for _ in range(T)]
  # policy_name = 'ht'

  # episode_partial = partial(episode, policy_name=policy_name, baseline_schedule=BASELINE_SCHEDULE,
  #   alpha_schedule = ALPHA_SCHEDULE, std = std, T = T, monte_carlo_reps = monte_carlo_reps,
  #   list_of_reward_mus = list_of_reward_mus, test = test)
  # episode_partial(0)

  run('eps-greedy-ht', std=1.0)
  run('eps-greedy-ht')