import sys
import pdb
import os
this_dir = os.path.dirname(os.path.abspath(__file__))
project_dir = os.path.join(this_dir, '..', '..')
sys.path.append(project_dir)

from src.policies import tuned_bandit_policies as tuned_bandit
from src.policies import rollout
from src.environments.Bandit import LinearCB, NormalCB
from sklearn.linear_model import Ridge, LinearRegression
import src.policies.global_optimization as opt
from functools import partial
import datetime
import yaml
import multiprocessing as mp
import src.hypothesis_test.cb_hypothesis_test as ht
import numpy as np


def operating_chars_episode(label, policy_name, alpha_schedule, baseline_schedule, contamination, n_patients=15,
                            list_of_reward_betas=[[-10, 0.4, 0.4, -0.4], [-9.8, 0.6, 0.6, -0.4]],
                            context_mean=np.array([0.0, 0.0, 0.0]),
                            context_var=np.array([[1.0,0,0], [0,1.,0], [0, 0, 1.]]), list_of_reward_vars=[1, 1], T=50,
                            mc_replicates=100, test=False, use_default_tuning_parameter=False):
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
  TUNE_INTERVAL = 5
  DONT_TUNE_UNTIL = 30
  np.random.seed(label)

  if test:
    NUM_CANDIDATE_HYPOTHESES = 5
    mc_reps_for_ht = 5
  else:
    NUM_CANDIDATE_HYPOTHESES = 100  # Number of candidate null models to consider when conducting ht
    mc_reps_for_ht = 500

  # Settings
  feature_function = lambda z: z
  positive_zeta = False
  # baseline_tuning_function = lambda T, t, zeta: baseline_schedule[t]
  # tuning_function = tuned_bandit.expit_epsilon_decay
  tuning_function = lambda T, t, zeta: baseline_schedule[t]
  baseline_tuning_function = lambda T, t, zeta: 0.5
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
  when_hypothesis_rejected = float('inf')

  # Initialize environment
  env = NormalCB(num_initial_pulls=1, list_of_reward_betas=list_of_reward_betas, context_mean=context_mean,
                 context_var=context_var, list_of_reward_vars=list_of_reward_vars)

  def true_context_sampler(n_):
    X_ = np.random.multivariate_normal(mean=env.context_mean, cov=env.context_var, size=n_)
    return np.column_stack((np.ones(n_), X_))

  true_model_params = [[beta, np.sqrt(var)] for beta, var in zip(env.list_of_reward_betas, env.list_of_reward_vars)]

  t1_errors = []
  t2_errors = []
  alpha_at_h0 = []
  action_probs = [[1.0] for a in range(len(list_of_reward_betas))]
  for t in range(T):
    print(t)
    time_to_tune = (tune and t > 0 and t % TUNE_INTERVAL == 0 and t >= DONT_TUNE_UNTIL)

    ## Hypothesis testing setup ##
    estimated_model = [[beta_hat, sigma_hat, XprimeX_inv, X, y] for beta_hat, sigma_hat, XprimeX_inv, X, y
                       in zip(env.beta_hat_list, env.sigma_hat_list, env.Xprime_X_inv_list,
                              env.X_list, env.y_list)]

    def baseline_policy(means, standard_errors, num_pulls, tprime):
      return policy(means, standard_errors, num_pulls, baseline_tuning_function, None, T, tprime, None)

    def proposed_policy(means, standard_errors, num_pulls, tprime):
      return policy(means, standard_errors, num_pulls, tuning_function, tuning_function_parameter, T, tprime, None)

    true_model_list = []  # Construct list of candidate models by drawing from sampling dbn
    print('sampling candidate models')

    if time_to_tune:
      # beta_hats_, beta_covs_ = ht.cb_ipw(env, action_probs)
      for draw in range(NUM_CANDIDATE_HYPOTHESES):
        # sampled_model = env.sample_from_posterior(beta_hats=list_of_reward_betas) # ToDo: using true model for debugging
        sampled_model = env.sample_from_posterior()
        param_list_for_sampled_model = [[sampled_model[a]['beta_draw'], np.sqrt(sampled_model[a]['var_draw'])]
                                        for a in range(env.number_of_actions)]
        true_model_list.append(param_list_for_sampled_model)
      beta0s = np.array([m[0][0] for m in true_model_list])
      beta1s = np.array([m[1][0] for m in true_model_list])
      print('mean beta0: {}'.format(beta0s.mean(axis=0)))
      print('std beta0: {}'.format(beta0s.std(axis=0)))
      print('mean beta1: {}'.format(beta1s.mean(axis=0)))
      print('std beta1: {}'.format(beta1s.std(axis=0)))
      ## End hypothesis testing setup ##

    if time_to_tune:
      gen_model_parameters = []
      for rep in range(mc_replicates):
        draws = env.sample_from_posterior()
        betas_for_each_action = []
        vars_for_each_action = []
        for a in range(env.number_of_actions):
          beta_a = draws[a]['beta_draw']
          var_a = draws[a]['var_draw']
          betas_for_each_action.append(beta_a)
          vars_for_each_action.append(var_a)
          param_dict = {'reward_betas': betas_for_each_action, 'reward_vars': vars_for_each_action,
                        'context_mean': draws['context_mu_draw'], 'context_var': draws['context_var_draw']}
#                          'context_max': draws['context_max']}
          gen_model_parameters.append(param_dict)

      sim_env = NormalCB(num_initial_pulls=1, list_of_reward_betas=betas_for_each_action, context_mean=context_mean,
                         context_var=context_var, list_of_reward_vars=vars_for_each_action)
      print('pre-simulating data')
      pre_simulated_data = sim_env.generate_mc_samples(mc_replicates, T, n_patients=n_patients,
                                                       gen_model_params=gen_model_parameters)

      print('tuning')
      if use_default_tuning_parameter:
        tuning_function_parameter = np.array([0.05, 49., 2.5])
      else:
        tuning_function_parameter = opt.bayesopt(rollout.normal_cb_rollout_with_fixed_simulations, policy,
                                                 tuning_function, tuning_function_parameter, T,
                                                 sim_env, mc_replicates,
                                                 {'pre_simulated_data': pre_simulated_data},
                                                 bounds, explore_, positive_zeta=positive_zeta, test=test)
      ## Hypothesis testing ##

      def context_dbn_sampler(n_):
        X_ = np.random.multivariate_normal(mean=env.estimated_context_mean, cov=env.estimated_context_cov, size=n_)
        return X_

      print('hypothesis testing')
      number_of_pulls = [len(y_list_) for y_list_ in env.y_list]
      # ToDo: using true context dbn sampler for debugging
      ht_rejected = ht.conduct_approximate_cb_ht(baseline_policy, proposed_policy, true_model_list, estimated_model,
                                                 number_of_pulls, t, T, ht.cb_sampling_dbn, alpha_schedule[t],
                                                 ht.true_cb_regret,
                                                 ht.pre_generate_cb_data, true_context_sampler, feature_function,
                                                 mc_reps=mc_reps_for_ht, contamination=contamination)

      ## Get true regret of baseline ##
      h0_true, true_diff_ = ht.is_cb_h0_true(baseline_policy, proposed_policy, estimated_model, number_of_pulls,
                                             t, T, ht.true_cb_regret, ht.pre_generate_cb_data, true_model_params,
                                             true_context_sampler, mc_reps_for_ht, feature_function)
      print('true diff: {}'.format(true_diff_))
      print('beta hat: {}'.format(env.beta_hat_list))

      if ht_rejected and no_rejections_yet:
        when_hypothesis_rejected = int(t)
        no_rejections_yet = False

    print(env.estimated_context_cov)
    estimated_means = [np.dot(env.curr_context, b) for b in env.beta_hat_list]
    action, action_prob = policy(estimated_means, None, None, baseline_tuning_function, None, T, t, env)
    action_probs[action].append(action_prob)
    env.step(action)

    ## Record operating characteristics ##
    if time_to_tune:
      if h0_true:
        t1_errors.append(int(ht_rejected))
        alpha_at_h0.append(float(alpha_schedule[t]))
      else:
        t2_errors.append(int(1-ht_rejected))
      if ht_rejected: # Break as soon as there is a rejection
        break

  return {'when_hypothesis_rejected': when_hypothesis_rejected,
          'baseline_schedule': baseline_schedule, 'alpha_schedule': alpha_schedule, 'type1': t1_errors,
          'type2': t2_errors, 'alpha_at_h0': alpha_at_h0}


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


def operating_chars_run(label, contamination, T=50, replicates=36, test=False,
                        use_default_tuning_parameter=False, save=True):
  BASELINE_SCHEDULE = [np.max((0.01, 0.5 / (t + 1))) for t in range(T)]
  ALPHA_SCHEDULE = [float(1.0 / (T - t)) for t in range(T)]

  if test:
    replicates = num_cpus = 1
    monte_carlo_reps = 5
  else:
    num_cpus = 36
  episode_partial = partial(operating_chars_episode, policy_name='eps-decay', baseline_schedule=BASELINE_SCHEDULE,
                            alpha_schedule=ALPHA_SCHEDULE, contamination=contamination, T=T, test=test,
                            use_default_tuning_parameter=use_default_tuning_parameter)
  num_batches = int(replicates / num_cpus)

  results = []
  if test or replicates == 1:
    results.append(episode_partial(0))
  else:
    pool = mp.Pool(processes=num_cpus)
    for batch in range(label*num_batches, (label+1)*num_batches):
      results_for_batch = pool.map(episode_partial, range(batch*num_cpus, (batch+1)*num_cpus))
      results += results_for_batch

  t1_errors = []
  for d in results:
    t1_errors += d['type1']
  t2_errors = [e for d in results for e in d['type2']]
  alphas_at_h0 = [a for d in results for a in d['alpha_at_h0']]

  if save:
    results = {'t1_errors': t1_errors, 'alphas_at_h0': alphas_at_h0,
               't2_errors': t2_errors}
    base_name = 'eps-cb-contam={}'.format(contamination)
    prefix = os.path.join(project_dir, 'src', 'run', base_name)
    suffix = datetime.datetime.now().strftime("%y%m%d_%H%M%S")
    filename = '{}_{}.yml'.format(prefix, suffix)
    with open(filename, 'w') as outfile:
      yaml.dump(results, outfile)


if __name__ == "__main__":
  T = 50
  test = False
  use_default_tuning_parameter = True
  BASELINE_SCHEDULE = [np.max((0.01, 0.5 / (t + 1))) for t in range(T)]
  ALPHA_SCHEDULE = [float(1.0 / (T - t)) for t in range(T)]
  # for contamination in [0.0, 0.5, 0.99]:
  #   operating_chars_run(1, contamination, T=T, replicates=36*4, test=False,
  #                       use_default_tuning_parameter=use_default_tuning_parameter)
  contamination = 0.9
  episode_partial = partial(operating_chars_episode, policy_name='cb_ht', baseline_schedule=BASELINE_SCHEDULE,
                            alpha_schedule=ALPHA_SCHEDULE, contamination=contamination, T=T, test=test,
                            use_default_tuning_parameter=True)
  episode_partial(1)

