import sys
import pdb
import os
this_dir = os.path.dirname(os.path.abspath(__file__))
project_dir = os.path.join(this_dir, '..', '..')
sys.path.append(project_dir)


from src.policies import tuned_bandit_policies as tuned_bandit
from src.policies import gittins_index_policies as gittins
from src.policies import rollout
from src.environments.Bandit import BernoulliMAB
import src.policies.global_optimization as opt
import numpy as np
from functools import partial
import datetime
import yaml
import multiprocessing as mp


def episode(policy_name, label, list_of_reward_mus=[0.3, 0.6], T=50, monte_carlo_reps=1000, posterior_sample=True):
  np.random.seed(label)

  # ToDo: Create factory function that encapsulates this behavior
  posterior_sample = True
  bootstrap_posterior = False
  sampling_sample = False # whether to sample from sampling distribution
  positive_zeta = False
  if policy_name == 'eps':
    tuning_function = lambda a, b, c: 0.05  # Constant epsilon
    policy = tuned_bandit.mab_epsilon_greedy_policy
    tune = False
    tuning_function_parameter = None
  elif policy_name == 'greedy':
    tuning_function = lambda a, b, c: 0.00  # Constant epsilon
    policy = tuned_bandit.mab_epsilon_greedy_policy
    tune = False
    tuning_function_parameter = None
  elif policy_name == 'eps-decay-fixed':
    tuning_function = tuned_bandit.expit_epsilon_decay
    policy = tuned_bandit.mab_epsilon_greedy_policy
    tune = False
    tuning_function_parameter = np.array([0.050, 45.0000,  2.5000])
  elif policy_name == 'eps-decay':
    tuning_function = tuned_bandit.expit_epsilon_decay
    policy = tuned_bandit.mab_epsilon_greedy_policy
    tune = True
    tuning_function_parameter = np.array([0.05, -45, 2.5])
    bounds = {'zeta0': (0.05, 2.0), 'zeta1': (1.0, 49.0), 'zeta2': (0.01, 2.5)}
    explore_ = {'zeta0': [1.0, 1.0, 1.0, 1.90980867, 5.848, 0.4466, 10.177], 
                'zeta1': [-25.0, -49.0, -1.0, -49.94980088, -88.9, -50, -87.55],
                'zeta2': [0.1, 2.5, 2.0, 1.88292034, 0.08, 0.1037, 0.094]}
    posterior_sample = True
  elif policy_name == 'eps-decay-posterior-sample':
    tuning_function = tuned_bandit.expit_epsilon_decay
    policy = tuned_bandit.mab_epsilon_greedy_policy
    tune = True
    tuning_function_parameter = np.ones(10)*0.05
    posterior_sample = True
  elif policy_name == 'eps-decay-bootstrap-sample':
    tuning_function = tuned_bandit.stepwise_linear_epsilon
    policy = tuned_bandit.mab_epsilon_greedy_policy
    tune = True
    tuning_function_parameter = np.ones(10)*0.05
    posterior_sample = True
    bootstrap_posterior = True
  elif policy_name == 'ts-decay-posterior-sample':
    tuning_function = tuned_bandit.expit_epsilon_decay
    policy = tuned_bandit.mab_thompson_sampling_policy
    tune = True
    tuning_function_parameter = np.ones(10)*0.1
    posterior_sample = True
  elif policy_name == 'ts':
    tuning_function = lambda a, b, c: 1.0
    policy = tuned_bandit.mab_thompson_sampling_policy
    tune = False
    tuning_function_parameter = None
  elif policy_name == 'ts-fixed':
    tuning_function = tuned_bandit.stepwise_linear_epsilon
    policy = tuned_bandit.mab_thompson_sampling_policy
    tune = False
    tuning_function_parameter = np.array([0.8, 0.8, 0.8, 0.8, 0.0, 0.8, 0.0, 0.8, 0.8, 0.8])
  elif policy_name == 'posterior-ts-tuned':
    tuning_function = tuned_bandit.expit_epsilon_decay
    policy = tuned_bandit.mab_thompson_sampling_policy
    tune = True
    tuning_function_parameter = np.array([0.8, 49.0, 2.5])
    posterior_sample = True
#    bounds = {'zeta0': (0.8, 2.0), 'zeta1': (1.0, 49.0), 'zeta2': (0.01, 2.5)}
#    explore_ = {'zeta0': [1.0, 1.0, 1.0], 'zeta1': [25.0, 49.0, 1.0], 'zeta2': [0.1, 2.5, 2.0]}
    ## Add the explore_ points which are close to 1/(t+1), and also enlarge the bounds ##
    bounds = {'zeta0': (0.05, 2.0), 'zeta1': (1.0, 49.0), 'zeta2': (0.01, 2.5)}
    explore_ = {'zeta0': [1.0, 1.0, 1.0, 1.90980867, 5.848, 0.4466, 10.177], 
                'zeta1': [25.0, 49.0, 1.0, 49.94980088, 88.9, 50, 87.55], 
                'zeta2': [0.1, 2.5, 2.0, 1.88292034, 0.08, 0.1037, 0.094]}
  elif policy_name == 'ucb-tune-posterior':
    bounds = {'zeta0': (0.05, 2.0), 'zeta1': (1.0, 49.0), 'zeta2': (0.01, 2.5)}
    explore_ = {'zeta0': [1.0, 1.0, 1.0, 1.90980867, 5.848, 0.4466, 10.177], 
                'zeta1': [25.0, 49.0, 1.0, 49.94980088, 88.9, 50, 87.55], 
                'zeta2': [0.1, 2.5, 2.0, 1.88292034, 0.08, 0.1037, 0.094]}
    tuning_function = tuned_bandit.expit_epsilon_decay
    policy = tuned_bandit.bernoulli_mab_ucb_posterior_policy
    tune = True
    tuning_function_parameter = np.array([1.0, 89.0, 5.0])
  elif policy_name == 'ucb':
#    tuning_function = lambda a, b, c: 0.05
    tuning_function = lambda a, b, c: 0.9
    policy = tuned_bandit.bernoulli_mab_ucb_posterior_policy
    tune = False
    tuning_function_parameter = None
  elif policy_name == 'ucb-fixed-decay':
    tuning_function = lambda a, b, c: 0.9/(b+1.0)
    policy = tuned_bandit.bernoulli_mab_ucb_policy
    tune = False
    tuning_function_parameter = None
  elif policy_name == 'frequentist-ts-fixed':
    tuning_function = tuned_bandit.expit_epsilon_decay
    policy = tuned_bandit.mab_frequentist_ts_policy
    tune = False
    tuning_function_parameter = np.array([0.8, 49.0, 2.5])
    posterior_sample = True
  elif policy_name == 'frequentist-ts':
    tuning_function = lambda a, b, c: 1.0
    policy = tuned_bandit.mab_frequentist_ts_policy
    tune = False
    tuning_function_parameter = None
    posterior_sample = True
  elif policy_name == 'frequentist-ts-fixed-decay':
    tuning_function = lambda a, b, c: 1.0/(b+1.0)
    policy = tuned_bandit.mab_frequentist_ts_policy
    tune = False
    tuning_function_parameter = None
    posterior_sample = True
  elif policy_name == 'frequentist-ts-tuned':
    tuning_function = tuned_bandit.expit_epsilon_decay
    policy = tuned_bandit.mab_frequentist_ts_policy
    tune = True
    tuning_function_parameter = np.array([0.8, 49.0, 2.5])
    posterior_sample = True
    bounds = {'zeta0': (0.05, 2.0), 'zeta1': (1.0, 49.0), 'zeta2': (0.01, 2.5)}
    explore_ = {'zeta0': [1.0, 1.0, 1.0, 1.90980867, 5.848, 0.4466, 10.177], 
                'zeta1': [25.0, 49.0, 1.0, 49.94980088, 88.9, 50, 87.55], 
                'zeta2': [0.1, 2.5, 2.0, 1.88292034, 0.08, 0.1037, 0.094]}
  elif policy_name == 'gittins':
    tuning_function = lambda a, b, c: None
    tuning_function_parameter = None
    tune = False
    policy = gittins.normal_mab_gittins_index_policy
  else:
    raise ValueError('Incorrect policy name')

  env = BernoulliMAB(list_of_reward_mus=list_of_reward_mus)

  cumulative_regret = 0.0
  mu_opt = np.max(env.list_of_reward_mus)
  env.reset()
  tuning_parameter_sequence = []

  estimated_means_list = []
  estimated_vars_list = []
  actions_list = []
  rewards_list = []

  for t in range(T):

    if tune:
      if posterior_sample:
        reward_means = []
        for rep in range(monte_carlo_reps):
          draws = env.sample_from_posterior()
          means_for_each_action = []
          for a in range(env.number_of_actions):
            mean_a = draws[a]['mu_draw']
            means_for_each_action.append(mean_a)
          reward_means.append(means_for_each_action)
      else:
        reward_means = None

      sim_env = BernoulliMAB(list_of_reward_mus=env.estimated_means)
      pre_simulated_data = sim_env.generate_mc_samples_bernoulli(monte_carlo_reps, T, reward_means=reward_means)
      tuning_function_parameter, _ = opt.bayesopt(rollout.bernoulli_mab_rollout_with_fixed_simulations, policy,
                                               tuning_function,
                                               tuning_function_parameter, T, env, monte_carlo_reps,
                                               {'pre_simulated_data': pre_simulated_data},
                                               bounds, explore_, positive_zeta=positive_zeta)
      tuning_parameter_sequence.append([float(z) for z in tuning_function_parameter])

    action = policy(env.estimated_means, env.standard_errors, env.number_of_pulls, tuning_function,
                    tuning_function_parameter, T, t, env)

    # Compute regret
    regret = mu_opt - env.list_of_reward_mus[action]
    cumulative_regret += regret

  return {'cumulative_regret': cumulative_regret, 'zeta_sequence': tuning_parameter_sequence,
          'estimated_means': estimated_means_list, 'estimated_vars': estimated_vars_list,
          'rewards_list': rewards_list, 'actions_list': actions_list}


def run(policy_name, list_of_reward_mus=[0.3, 0.6], save=True, T=50, monte_carlo_reps=1000, posterior_sample=False):
  """

  :return:
  """
  replicates = 96*4
  num_cpus = int(mp.cpu_count())
  pool = mp.Pool(processes=num_cpus)
  episode_partial = partial(episode, policy_name, list_of_reward_mus=list_of_reward_mus, 
                            T=T, monte_carlo_reps=monte_carlo_reps,
                            posterior_sample=posterior_sample)
  num_batches = int(replicates / num_cpus)

  results = []
  for batch in range(num_batches):
    results_for_batch = pool.map(episode_partial, range(batch*num_cpus, (batch+1)*num_cpus))
    results += results_for_batch

  # results = pool.map(episode_partial, range(replicates))
  cumulative_regret = [np.float(d['cumulative_regret']) for d in results]
  zeta_sequences = [d['zeta_sequence'] for d in results]
  estimated_means = [d['estimated_means'] for d in results]
  estimated_vars = [d['estimated_vars'] for d in results]
  rewards = [d['rewards_list'] for d in results]
  actions = [d['actions_list'] for d in results]
  print(float(np.mean(cumulative_regret)), float(np.std(cumulative_regret))/np.sqrt(replicates))
  # Save results
  if save:
    results = {'T': float(T), 'list_of_reward_mus':list_of_reward_mus, 'mean_regret': float(np.mean(cumulative_regret)), 
               'se_regret': float(np.std(cumulative_regret))/np.sqrt(replicates),
               'regret list': [float(r) for r in cumulative_regret],
               'zeta_sequences': zeta_sequences, 'estimated_means': estimated_means, 'estimated_vars': estimated_vars,
               'rewards': rewards, 'actions': actions}

    base_name = 'bernoullimab-policy-{}-numAct-{}'.format(policy_name, len(list_of_reward_mus))
    prefix = os.path.join(project_dir, 'src', 'run', base_name)
    suffix = datetime.datetime.now().strftime("%y%m%d_%H%M%S")
    filename = '{}_{}.yml'.format(prefix, suffix)
    with open(filename, 'w') as outfile:
      yaml.dump(results, outfile)

  return


if __name__ == '__main__':
  episode('eps-decay', label=1, T=5, monte_carlo_reps=10, posterior_sample=True)

