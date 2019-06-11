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
import numpy as np
from functools import partial
import datetime
import yaml
import multiprocessing as mp


def episode(policy_name, label, std=0.1, list_of_reward_mus=[0.3,0.6], T=50, monte_carlo_reps=1000, posterior_sample=False):
  np.random.seed(label)

  # ToDo: Create factory function that encapsulates this behavior
  # posterior_sample = False
  bootstrap_posterior = False
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
#    if T==25:
#      tuning_function = lambda a, b, c: 0.7**b
#    elif T==50:
#      tuning_function = lambda a, b, c: 0.85**b
    tuning_function = lambda a, t, c: 0.5 / (t + 1)
#    tuning_function = tuned_bandit.expit_epsilon_decay
#    tuning_function = tuned_bandit.stepwise_linear_epsilon
    policy = tuned_bandit.mab_epsilon_greedy_policy
    tune = False
    tuning_function_parameter = np.array([0.05, 49.,    2.5 ])
    # Estimated optimal for normal mab with high variance on good arm
#    tuning_function_parameter = np.array([0.2, 0.164, 0.2, 0.193, 0.189, 
#                                          0.087, 0.069, 0.159, 0.09, 0.015])
  elif policy_name == 'eps-decay':
    tuning_function = tuned_bandit.expit_epsilon_decay
    policy = tuned_bandit.mab_epsilon_greedy_policy
    tune = True
    tuning_function_parameter = np.array([0.05, 45, 2.5])
    bounds = {'zeta0': (0.05, 1.0), 'zeta1': (1.0, 49.0), 'zeta2': (0.01, 2.5)}
    explore_ = {'zeta0': [1.0, 0.05, 1.0, 0.1, 0.1, 0.05, 4.43802103], 
                'zeta1': [50.0, 49.0, 1.0, 49.0, 1.0, 1.0,  85.04499728], 
                'zeta2': [0.1, 2.5, 1.0, 2.5, 2.5, 2.5, 0.09655535]}
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
    tuning_function = tuned_bandit.stepwise_linear_epsilon
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
  elif policy_name == 'ucb-tune-posterior-sample':
    bounds = {'zeta0': (0.8, 2.0), 'zeta1': (1.0, 49.0), 'zeta2': (0.01, 2.5)}
    explore_ = {'zeta0': [1.0, 1.0, 1.0, 0.9, 0.52814572, 10.49709491], 
                'zeta1': [25.0, 49.0, 1.0, 1.0, 49.09475206, 89.41417424], 
                'zeta2': [0.1, 2.5, 2.0, 2.5, 0.14538589, 0.09326324]}
    tuning_function = tuned_bandit.expit_epsilon_decay
    policy = tuned_bandit.normal_mab_ucb_policy
    tune = True
    tuning_function_parameter = np.array([1.0, 89.0, 5.0])
    posterior_sample = True
  elif policy_name == 'ucb':
#    tuning_function = lambda a, b, c: 0.05
    tuning_function = lambda a, b, c: 0.9
    policy = tuned_bandit.normal_mab_ucb_policy
    tune = False
    tuning_function_parameter = None
  elif policy_name == 'ucb-fixed-decay':
    tuning_function = lambda a, b, c: 0.9/(b+1.0)
    policy = tuned_bandit.normal_mab_ucb_policy
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
#    bounds = {'zeta0': (0.8, 2.0), 'zeta1': (1.0, 49.0), 'zeta2': (0.01, 2.5)}
#    explore_ = {'zeta0': [1.0, 1.0, 1.0], 'zeta1': [25.0, 49.0, 1.0], 'zeta2': [0.1, 2.5, 2.0]}
    ## Add the explore_ points which are close to 1/(t+1), and also enlarge the bounds ##
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

#  env = NormalMAB(list_of_reward_mus=[[1], [1.1]], list_of_reward_vars=[[1], [1]])
  # env = NormalMAB(list_of_reward_mus=[[0], [1]], list_of_reward_vars=[[1], [140]])
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

    if tune:
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
      print(sim_env.estimated_means)
      tuning_function_parameter = opt.bayesopt(rollout.mab_rollout_with_fixed_simulations, policy, tuning_function,
                                               tuning_function_parameter, T, env, monte_carlo_reps,
                                               {'pre_simulated_data': pre_simulated_data},
                                               bounds, explore_, positive_zeta=positive_zeta)
      tuning_parameter_sequence.append([float(z) for z in tuning_function_parameter])

    print('standard errors {}'.format(env.standard_errors))
    print('estimated vars {}'.format(env.estimated_vars))
    if policy_name == 'gittins':
      estimated_means = []
      for aa in range(env.number_of_actions):
        estimated_means.append(sum(env.draws_from_each_arm[aa])/env.number_of_pulls[aa])
      action = policy(estimated_means, env.standard_errors, env.number_of_pulls, tuning_function,
                    tuning_function_parameter, T, t, env)
    else:
      action = policy(env.estimated_means, env.standard_errors, env.number_of_pulls, tuning_function,
                    tuning_function_parameter, T, t, env)
    res = env.step(action)
    u = res['Utility']
    actions_list.append(int(action))
    rewards_list.append(float(u))

    # Compute regret
    regret = mu_opt - env.list_of_reward_mus[action]
    cumulative_regret += regret

  return {'cumulative_regret': cumulative_regret, 'zeta_sequence': tuning_parameter_sequence, 
          'estimated_means': estimated_means_list, 'estimated_vars': estimated_vars_list,
          'rewards_list': rewards_list, 'actions_list': actions_list}


def run(policy_name, std=0.1, list_of_reward_mus=[0.3,0.6],save=True, T=50, monte_carlo_reps=1000, posterior_sample=False):
  """

  :return:
  """
  replicates = 192
  num_cpus = int(mp.cpu_count())
  pool = mp.Pool(processes=num_cpus)
  episode_partial = partial(episode, policy_name, std=std, T=T, monte_carlo_reps=monte_carlo_reps,
                            posterior_sample=posterior_sample, list_of_reward_mus=list_of_reward_mus)
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
    results = {'T': float(T), 'mean_regret': float(np.mean(cumulative_regret)), 'std_regret': float(np.std(cumulative_regret)),
               'regret list': [float(r) for r in cumulative_regret],
               'zeta_sequences': zeta_sequences, 'estimated_means': estimated_means, 'estimated_vars': estimated_vars,
               'rewards': rewards, 'actions': actions, 'std': std}

    base_name = 'normalmab-postsample-{}-std-{}-{}-numAct-{}'.format(posterior_sample, std, policy_name, len(list_of_reward_mus))
    prefix = os.path.join(project_dir, 'src', 'run', base_name)
    suffix = datetime.datetime.now().strftime("%y%m%d_%H%M%S")
    filename = '{}_{}.yml'.format(prefix, suffix)
    with open(filename, 'w') as outfile:
      yaml.dump(results, outfile)

  return


if __name__ == '__main__':
#  episode('frequentist-ts-tuned', np.random.randint(low=1, high=1000))
#   run('eps-decay-fixed', save=False, std=1)
  # run('eps')
#   run('greedy', save=False)
  # run('eps-decay-bootstrap-sample', T=1, monte_carlo_reps=1)
  # run('ts-decay-posterior-sample', T=10, monte_carlo_reps=100)
#  run('ucb', std=0.1, T=50, save=False, monte_carlo_reps=1000)
#  run('ucb-fixed-decay', std=1, T=50, monte_carlo_reps=1000)
#   run('ts-fixed', T=50, monte_carlo_reps=1000)
#   run('frequentist-ts', T=50, std=1, monte_carlo_reps=1000, posterior_sample=True)
  run('gittins', T=50, std=0.1, list_of_reward_mus=[0.74, 0.15, 0.34, 0.48, 0.53, 0.23, 0.47, 0.51, 0.71, 0.42], monte_carlo_reps=1000, posterior_sample=True)
#   run('frequentist-ts-fixed-decay', list_of_reward_mus=[0.73, 0.56, 0.33, 0.04, 0.66], T=50, std=1, monte_carlo_reps=1000, posterior_sample=True)

#  run('frequentist-ts-tuned', T=50, std=0.1, list_of_reward_mus=[0.3,0.6], monte_carlo_reps=1000, posterior_sample=True)
#  run('eps-decay', T=50, std=0.1, list_of_reward_mus=[0.3,0.6],monte_carlo_reps=1000, posterior_sample=True)
#  run('ucb-tune-posterior-sample', std=0.1,list_of_reward_mus=[0.3,0.6], T=50, monte_carlo_reps=1000, posterior_sample=True)
#  run('frequentist-ts-tuned', T=50, std=1, list_of_reward_mus=[0.3,0.6], monte_carlo_reps=1000, posterior_sample=True)
#  run('eps-decay', T=50, std=1, list_of_reward_mus=[0.3,0.6],monte_carlo_reps=1000, posterior_sample=True)
#  run('ucb-tune-posterior-sample', std=1,list_of_reward_mus=[0.3,0.6], T=50, monte_carlo_reps=1000, posterior_sample=True)
#
#  run('frequentist-ts-tuned', T=50, std=0.1, list_of_reward_mus=[0.73, 0.56, 0.33, 0.04, 0.66], monte_carlo_reps=1000, posterior_sample=True)
#  run('eps-decay', T=50, std=0.1, list_of_reward_mus=[0.73, 0.56, 0.33, 0.04, 0.66],monte_carlo_reps=1000, posterior_sample=True)
#  run('ucb-tune-posterior-sample', std=0.1,list_of_reward_mus=[0.73, 0.56, 0.33, 0.04, 0.66], T=50, monte_carlo_reps=1000, posterior_sample=True)
#  run('frequentist-ts-tuned', T=50, std=1, list_of_reward_mus=[0.73, 0.56, 0.33, 0.04, 0.66], monte_carlo_reps=1000, posterior_sample=True)
#  run('eps-decay', T=50, std=1, list_of_reward_mus=[0.73, 0.56, 0.33, 0.04, 0.66],monte_carlo_reps=1000, posterior_sample=True)
#  run('ucb-tune-posterior-sample', std=1,list_of_reward_mus=[0.73, 0.56, 0.33, 0.04, 0.66], T=50, monte_carlo_reps=1000, posterior_sample=True)
#
#  run('frequentist-ts-tuned', T=50, std=0.1, list_of_reward_mus=[0.74, 0.15, 0.34, 0.48, 0.53, 0.23, 0.47, 0.51, 0.71, 0.42], 
#      monte_carlo_reps=1000, posterior_sample=True)
#  run('eps-decay', T=50, std=0.1, list_of_reward_mus=[0.74, 0.15, 0.34, 0.48, 0.53, 0.23, 0.47, 0.51, 0.71, 0.42],
#      monte_carlo_reps=1000, posterior_sample=True)
#  run('ucb-tune-posterior-sample', std=0.1, list_of_reward_mus=[0.74, 0.15, 0.34, 0.48, 0.53, 0.23, 0.47, 0.51, 0.71, 0.42], 
#      T=50, monte_carlo_reps=1000, posterior_sample=True)  
#  run('frequentist-ts-tuned', T=50, std=1, list_of_reward_mus=[0.74, 0.15, 0.34, 0.48, 0.53, 0.23, 0.47, 0.51, 0.71, 0.42], 
#      monte_carlo_reps=1000, posterior_sample=True)
#  run('eps-decay', T=50, std=1, list_of_reward_mus=[0.74, 0.15, 0.34, 0.48, 0.53, 0.23, 0.47, 0.51, 0.71, 0.42],
#      monte_carlo_reps=1000, posterior_sample=True)
#  run('ucb-tune-posterior-sample', std=1, list_of_reward_mus=[0.74, 0.15, 0.34, 0.48, 0.53, 0.23, 0.47, 0.51, 0.71, 0.42], 
#      T=50, monte_carlo_reps=1000, posterior_sample=True)
  






