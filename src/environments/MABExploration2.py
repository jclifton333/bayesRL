import sys
import pdb
import os
this_dir = os.path.dirname(os.path.abspath(__file__))
project_dir = os.path.join(this_dir, '..', '..')
sys.path.append(project_dir)


import matplotlib.pyplot as plt
from src.policies import tuned_bandit_policies as tuned_bandit
#from src.environments.policy_iteration import policy_iteration
#from src.environments.grid import Gridworld
import Bandit
  
import numpy as np
from functools import partial
import datetime
import yaml
import multiprocessing as mp
import sklearn
import matplotlib.pyplot as plt

from bayes_opt import BayesianOptimization
import time


def mab_epsilon_greedy_policy(estimated_means, standard_errors, number_of_pulls, tuning_function,
                              tuning_function_parameter, T, t, env, info):
  if info:
    R = standard_errors[1]/standard_errors[0]
    delta = estimated_means[1] - estimated_means[0]
    epsilon = tuning_function(T, t, tuning_function_parameter, R, delta)
  else:
    epsilon = tuning_function(T, t, tuning_function_parameter)
  greedy_action = np.argmax(estimated_means)
  if np.random.random() < epsilon:
    action = np.random.choice(2)
  else:
    action = greedy_action
  return action


def mab_epsilon_greedy_policy2(estimated_means, standard_errors, number_of_pulls, tuning_function,
                              tuning_function_parameter, T, t, env, info):
  if info:
    delta = abs(estimated_means[1] - estimated_means[0])
    epsilon = tuning_function(T, t, tuning_function_parameter, delta)
  else:
    epsilon = tuning_function(T, t, tuning_function_parameter)
  greedy_action = np.argmax(estimated_means)
  if np.random.random() < epsilon:
    action = np.random.choice(2)
  else:
    action = greedy_action
  return action


def mab_rollout_with_fixed_simulations(tuning_function_parameter, policy, time_horizon, tuning_function, env, 
                                       info, quantile, **kwargs):
  """
  Evaluate MAB exploration policy on already-generated data.
  :param kwargs: contain key pre_simlated_data, which is mc_rep-length list of dictionaries, which contain lists of
  length time_horizon of data needed to evaluate policy.
  :param tuning_function_parameter:
  :param policy:
  :param time_horizon:
  :param tuning_function:
  :param env:
  :return:
  """

  #percentile desired
  quant = 75

  pre_simulated_data = kwargs['pre_simulated_data']
  mean_cumulative_regret = 0.0
  optimal_reward = np.max(env.list_of_reward_mus)
  regrets = []
  for rep, rep_dict in enumerate(pre_simulated_data):
    initial_model = rep_dict['initial_model']
    estimated_means = initial_model['sample_mean_list']
    standard_errors = initial_model['standard_error_list']
    number_of_pulls = initial_model['number_of_pulls']
    estimated_vars = initial_model['sample_var_list']

    # Get obs sequences for this rep
    rewards_sequence = rep_dict['rewards']
    regrets_sequence = rep_dict['regrets']
    regret_for_rep = 0.0

    rewards_at_each_arm = [np.array([]) for _ in range(env.number_of_actions)]

    for t in range(time_horizon):
      # Draw context and draw arm based on policy
      action = policy(estimated_means, estimated_vars, None, tuning_function,
                tuning_function_parameter, time_horizon, t, env, info)
#      action = policy(estimated_means, standard_errors, None, tuning_function,
#                      tuning_function_parameter, time_horizon, t, env, info)

      # Get reward and regret
      reward = rewards_sequence[t, action]
      # regret = regrets_sequence[t, action]
      rewards_at_each_arm[action] = np.append(rewards_at_each_arm[action], reward)
      number_of_pulls[action] += 1
      expected_reward = env.list_of_reward_mus[action]
      regret_for_rep += (expected_reward-optimal_reward)

      # Update model
      sample_mean_at_action = (reward - estimated_means[action]) / number_of_pulls[action]
      estimated_means[action] = sample_mean_at_action
      standard_errors[action] = np.sqrt(np.mean((rewards_at_each_arm[action] - sample_mean_at_action)**2))
    if quantile:
      regrets.append(regret_for_rep)
    else:
      mean_cumulative_regret += (regret_for_rep - mean_cumulative_regret) / (rep + 1)
  if quantile:
    mean_cumulative_regret = np.percentile(regrets,quant)
  return mean_cumulative_regret


def bayesopt(rollout_function, policy, tuning_function, zeta_prev, time_horizon, env, mc_replicates,
             rollout_function_kwargs, bounds, explore_, info, quantile, positive_zeta=False):

  # Assuming 10 params!
  # def objective(zeta0, zeta1, zeta2, zeta3, zeta4, zeta5, zeta6, zeta7, zeta8, zeta9):
  # zeta = np.array([zeta0, zeta1, zeta2, zeta3, zeta4, zeta5, zeta6, zeta7, zeta8, zeta9])
  if info:
    def objective(zeta0, zeta1, zeta2, zeta3, zeta4):
      zeta = np.array([zeta0, zeta1, zeta2, zeta3, zeta4])
      return rollout_function(zeta, policy, time_horizon, tuning_function, env, info, quantile, **rollout_function_kwargs)
#    def objective(zeta0, zeta1, zeta2, zeta3, zeta4, zeta5, zeta6, zeta7, zeta8):
#      zeta = np.array([zeta0, zeta1, zeta2, zeta3, zeta4, zeta5, zeta6, zeta7, zeta8])
#      return rollout_function(zeta, policy, time_horizon, tuning_function, env, info, **rollout_function_kwargs)
  else:
    def objective(zeta0, zeta1, zeta2):
      zeta = np.array([zeta0, zeta1, zeta2])
      return rollout_function(zeta, policy, time_horizon, tuning_function, env, info, quantile, **rollout_function_kwargs)
  
  # bounds = {'zeta{}'.format(i): (lower_bound, upper_bound) for i in range(10)}
  explore_.update({'zeta{}'.format(i): [zeta_prev[i]] for i in range(len(zeta_prev))})
  bo = BayesianOptimization(objective, bounds)
  bo.explore(explore_)
  bo.maximize(init_points=10, n_iter=15)
  best_param = bo.res['max']['max_params']
  best_param = np.array([best_param['zeta{}'.format(i)] for i in range(len(bounds))])
  return best_param


def bayesopt_under_true_model(seed, info, quantile, mc_reps=1000, T=50):
  np.random.seed(seed)
  env = Bandit.NormalMAB(list_of_reward_mus=[0.3, 0.6], list_of_reward_vars=[0.1**2, 0.1**2])
  pre_simulated_data = env.generate_mc_samples(mc_reps, T)
  rollout_function_kwargs = {'pre_simulated_data': pre_simulated_data}

  rollout_function = mab_rollout_with_fixed_simulations
  policy = mab_epsilon_greedy_policy
  
  if info:
    bounds = {'zeta0': (0.05,2.0),'zeta1': (-5.0, 5.0),'zeta2': (-5.0,5.0),'zeta3': (-5.0,5.0),'zeta4': (-5.0,5.0),'zeta5': (-5.0,5.0),'zeta6': (-5.0,5.0),'zeta7': (-5.0,5.0), 'zeta8': (-5.0,5.0)}
    explore_ = {'zeta0': [0.05,0.1,0.0,1.0, 0.1],'zeta1': [0.0,0.0,0.0,0.0,-122.5],'zeta2': [0.0,0.0,0.0,0.0,0.0],'zeta3': [0.0,0.0,0.0,0.0,0.0],'zeta4': [0.0,0.0,0.0,0.0,0.0],'zeta5': [0.0,0.0,0.0,0.0,2.5],'zeta6': [0.0,0.0,0.0,0.0,0.0],'zeta7': [0.0,0.0,0.0,0.0,0.0], 'zeta8': [0.0,0.0,0.0,0.0,0.0]}
    tuning_function = tuned_bandit.information_expit_epsilon_decay
    def objective(zeta0, zeta1, zeta2, zeta3, zeta4, zeta5, zeta6, zeta7, zeta8):
      zeta = np.array([zeta0, zeta1, zeta2, zeta3, zeta4, zeta5, zeta6, zeta7, zeta8])
      return rollout_function(zeta, policy, T, tuning_function, env, info, quantile, **rollout_function_kwargs)
  else:
    bounds = {'zeta0': (0.05, 2.0), 'zeta1': (1.0, 49.0), 'zeta2': (0.01, 2.5)}
    explore_ = {'zeta0': [1.0, 0.05, 1.0, 0.1], 'zeta1': [50.0, 49.0, 1.0, 49.0], 'zeta2': [0.1, 2.5, 1.0, 2.5]}
    tuning_function = tuned_bandit.expit_epsilon_decay
    def objective(zeta0, zeta1, zeta2):
      zeta = np.array([zeta0, zeta1, zeta2])
      return rollout_function(zeta, policy, T, tuning_function, env, info, quantile, **rollout_function_kwargs)
  
  bo = BayesianOptimization(objective, bounds)
  bo.explore(explore_)
  bo.maximize(init_points=50, n_iter=50, alpha=1e-4)
#  bo.maximize(init_points=10, n_iter=15, alpha=1e-4)
  best_param = bo.res['max']['max_params']
  best_param = np.array([best_param['zeta{}'.format(i)] for i in range(len(bounds))])
  print(best_param)
  return best_param


def episode(policy_name, label, info, quantile, std=0.1, T=50, monte_carlo_reps=1000, posterior_sample=False,tune_start=-1,tune_stop=-1):
  np.random.seed(label)
  
  positive_zeta = False
  if policy_name == 'eps':
    eps = 0.1
    if info:
      tuning_function = lambda a, b, c, d, e: eps
    else:
      tuning_function = lambda a, b, c: eps # Constant epsilon
    policy = mab_epsilon_greedy_policy
    tune = False
    tuning_function_parameter = None
  elif policy_name == 'greedy':
    if info:
      tuning_function = lambda a, b, c, d, e: 0.00
    else:
      tuning_function = lambda a, b, c: 0.00  # Constant epsilon
    policy = mab_epsilon_greedy_policy
    tune = False
    tuning_function_parameter = None
  elif policy_name == "eps-decay":
    tune = True
    posterior_sample = True
    policy = mab_epsilon_greedy_policy2
    if info:
#      tuning_function_parameter = np.concatenate(([0.05],np.random.uniform(-0.5,0.5,8)))
#      bounds = {'zeta0': (0.05,2.0),'zeta1': (-5.0, 5.0),'zeta2': (-5.0,5.0),'zeta3': (-5.0,5.0),'zeta4': (-5.0,5.0),'zeta5': (-5.0,5.0),'zeta6': (-5.0,5.0),'zeta7': (-5.0,5.0), 'zeta8': (-5.0,5.0)}
#      explore_ = {'zeta0': [0.05,0.1,0.0,1.0, 0.1],'zeta1': [0.0,0.0,0.0,0.0,-122.5],'zeta2': [0.0,0.0,0.0,0.0,0.0],'zeta3': [0.0,0.0,0.0,0.0,0.0],'zeta4': [0.0,0.0,0.0,0.0,0.0],'zeta5': [0.0,0.0,0.0,0.0,2.5],'zeta6': [0.0,0.0,0.0,0.0,0.0],'zeta7': [0.0,0.0,0.0,0.0,0.0], 'zeta8': [0.0,0.0,0.0,0.0,0.0]}
#      tuning_function = tuned_bandit.information_expit_epsilon_decay
      tuning_function_parameter = np.concatenate(([0.05],np.random.uniform(-0.5,0.5,4)))
      bounds = {'zeta0': (0.05,2.0),'zeta1': (-5.0, 5.0),'zeta2': (-5.0,5.0),'zeta3': (-5.0,5.0), 'zeta4': (-5.0,5.0)}
      explore_ = {'zeta0': [0.05,0.1,0.0,1.0, 0.1],'zeta1': [0.0,0.0,0.0,0.0,-122.5],'zeta2': [0.0,0.0,0.0,0.0,0.0],
                  'zeta3': [0.0,0.0,0.0,0.0,0.0],'zeta4': [0.0,0.0,0.0,0.0,0.0]}
      tuning_function = tuned_bandit.information_expit_epsilon_decay2
    else:
      tuning_function_parameter = np.array([0.05, 1.0, 0.01]) 
      bounds = {'zeta0': (0.05, 2.0), 'zeta1': (1.0, 49.0), 'zeta2': (0.01, 2.5)}
      explore_ = {'zeta0': [1.0, 0.05, 1.0, 0.1], 'zeta1': [50.0, 49.0, 1.0, 49.0], 'zeta2': [0.1, 2.5, 1.0, 2.5]}
      tuning_function = tuned_bandit.expit_epsilon_decay
  elif policy_name == 'eps-fixed-decay':
    tune = False
    policy = mab_epsilon_greedy_policy
    if info:
      tuning_function = tuned_bandit.information_expit_epsilon_decay
      # obtained from quantile=false
      #tuning_function_parameter = np.array( [1.44604582, -1.04692838,  0.11755395, -2.35340744,  3.17611836, -3.83516477,
# -3.64502149 , 1.77037848 , 1.02438715])
      if quantile:
        # quant=75
        #tuning_function_parameter = np.array( [ 0,0,0,0,0,0,0,0,0])
        # quant=50
        tuning_function_parameter = np.array( [ 0.57164622,  4.62324166,  2.88626605, -3.49861048, -2.62096877, -1.86222763,
  2.89669445,  0.20207411,  2.21622908])
        # quant=25
        #tuning_function_parameter = np.array( [ 1.05237299, -0.25368232,  3.20527386, -4.67019178,  1.7334421,  -1.79493075,
  #1.66812503, -2.9827348,  4.49922889])
        # quant=10
        #tuning_function_parameter = np.array( [ 0.74892287, -3.25033099,  0.23110947, -4.87311224, -3.53732692, -4.62473731,
 #-0.83820035,  3.56393163, -2.76060575])
      else:
        tuning_function_parameter = np.array( [0.66225065, -4.8710143,  -0.79380959, -3.31694182, -3.22201959, -3.80354823,
  0.44470104,  2.75075175, -2.97245318])
    else:
      tuning_function = tuned_bandit.expit_epsilon_decay
      # obtained from quantile=false
      #tuning_function_parameter = np.array([0.05, 46.49084647, 2.5] )
      if quantile:
        # quant=75
        #tuning_function_parameter = np.array([0.15838401, 40.12500059, 2.21946538] )
        # quant=50
        tuning_function_parameter = np.array([0.05, 44.06308915, 2.5] )
        # quant=25
        #tuning_function_parameter = np.array([0.05, 46.77405996, 2.5] )
        # quant=10
        #tuning_function_parameter = np.array([0.08315987, 23.61019458,  2.44128153] )
      else:
        tuning_function_parameter = np.array([0.05, 47.22640834, 1.36518539] )

  env = Bandit.NormalMAB(list_of_reward_mus=[0.3, 0.6], list_of_reward_vars=[std**2, std**2])

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

    if tune and (t >= tune_start) and (t<tune_stop):
      print("########### Time: "+str(t)+"; Replicate: "+str(label)+" ############")
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

      sim_env = Bandit.NormalMAB(list_of_reward_mus=env.estimated_means, list_of_reward_vars=env.estimated_vars)
      pre_simulated_data = sim_env.generate_mc_samples(monte_carlo_reps, T, reward_means=reward_means,
                                                       reward_vars=reward_vars)

      tuning_function_parameter = bayesopt(mab_rollout_with_fixed_simulations, policy, tuning_function,
                                               tuning_function_parameter, T, env, monte_carlo_reps,
                                               {'pre_simulated_data': pre_simulated_data},
                                               bounds, explore_, info, quantile, positive_zeta=positive_zeta)
      tuning_parameter_sequence.append([float(z) for z in tuning_function_parameter])

#    print('standard errors {}'.format(env.standard_errors))
#    print('estimated vars {}'.format(env.estimated_vars))
    action = policy(env.estimated_means, env.estimated_vars, env.number_of_pulls, tuning_function,
              tuning_function_parameter, T, t, env, info)
#    action = policy(env.estimated_means, env.standard_errors, env.number_of_pulls, tuning_function,
#                    tuning_function_parameter, T, t, env, info)
    res = env.step(action)
    u = res['Utility']
    actions_list.append(int(action))
    rewards_list.append(float(u))

    # Compute regret
    regret = mu_opt - env.list_of_reward_mus[action]
    cumulative_regret += regret
  print("Cumulative regret: "+str(cumulative_regret))
  return {'cumulative_regret': cumulative_regret, 'zeta_sequence': tuning_parameter_sequence, 
          'estimated_means': estimated_means_list, 'estimated_vars': estimated_vars_list,
          'rewards_list': rewards_list, 'actions_list': actions_list}

    
def run(policy_name, info, quantile=False, save=True, mc_replicates=1000, T=50,tune_start=-1,tune_stop=-1):
  """

  :return:
  """

  replicates = 96
  num_cpus = int(mp.cpu_count())
  #num_cpus = 32
  #replicates = 20
  results = []
  pool = mp.Pool(processes=num_cpus)

  episode_partial = partial(episode, policy_name, monte_carlo_reps=mc_replicates, T=T, info=info, quantile=quantile,tune_start=tune_start,tune_stop=tune_stop)

  results = pool.map(episode_partial, range(replicates))
  #results = episode_partial(1)
  cumulative_regret = [np.float(d['cumulative_regret']) for d in results]
  zeta_sequences = [d['zeta_sequence'] for d in results]
  estimated_means = [d['estimated_means'] for d in results]
  estimated_vars = [d['estimated_vars'] for d in results]
  rewards = [d['rewards_list'] for d in results]
  actions = [d['actions_list'] for d in results]
#  if info:
#    posterior_betas = [d['posterior_betas']for d in results]
#    posterior_lambdas = [d['posterior_lambdas']for d in results]
#    posterior_mus = [d['posterior_mus']for d in results]
#  else:
#    posterior_betas = None
#    posterior_lambdas = None
#    posterior_mus = None
#  rewards = [list(d['rewards'].astype(float)) for d in results]
#  print(policy_name, cum_rewards)
  print(policy_name, info, quantile, 'rewards', float(np.mean(cumulative_regret)), 'se_rewards',float(np.std(cumulative_regret))/np.sqrt(replicates))
  # Save results
  if save:
    results = {'T': float(T), 'mean_regret': float(np.mean(cumulative_regret)), 'std_regret': float(np.std(cumulative_regret)),
               'regret list': [float(r) for r in cumulative_regret],
               'zeta_sequences': zeta_sequences, 'estimated_means': estimated_means, 'estimated_vars': estimated_vars,
               'rewards': rewards, 'actions': actions}

    base_name = 'info-mab-onlydelta-{}'.format(policy_name)
    prefix = os.path.join(project_dir, 'src', 'environments', base_name)
    suffix = datetime.datetime.now().strftime("%y%m%d_%H%M%S")
    filename = '{}_{}.yml'.format(prefix, suffix)
    with open(filename, 'w') as outfile:
      yaml.dump(results, outfile)

  return


if __name__ == '__main__':
  start_time = time.time()
#  check_coef_converge()
#  bayesopt_under_true_model(seed=0, info=False)
  #bayesopt_under_true_model(seed=1, info=True, quantile=True)
  #bayesopt_under_true_model(seed=0, info=True, quantile=False)
  #bayesopt_under_true_model(seed=0, info=False, quantile=True)
  #bayesopt_under_true_model(seed=0, info=False, quantile=False)
#  episode('eps-decay', 0, info=True, T=50)
#  episode('eps-fixed-decay', 2, T=50)
#  episode('eps', 0, info=True, T=50)
#  run('eps', save=False)
#  run('greedy', save=False, info=False)
  run('eps-decay', save=True, T=50, info=True, quantile=True,tune_start=10,tune_stop=11)
  run('eps-decay', save=True, T=50, info=True, quantile=False,tune_start=10,tune_stop=11)
#  run('eps-decay', save=True, T=50, info=True, quantile=False)
#  run('eps-decay', save=True, T=50, info=False, quantile=True)
#  run('eps-fixed-decay', save=False, T=50, info=True,quantile=True)
#  run('eps-fixed-decay', save=False, T=50, info=True,quantile=False)
#  run('eps-fixed-decay', save=False, T=50, info=False,quantile=True)
#  run('eps-fixed-decay', save=False, T=50, info=False,quantile=False)
#  run('eps-fixed-decay', save=False, T=50, info=False)
  #run('eps', save=False, T=50)
#  episode('eps', 1, T=50)
#  result = episode('eps', 0, T=1000)
#  print(result['cum_rewards'])
 # episode('eps-fixed-decay', 1, T=50)
#  num_processes = 4
#  num_replicates = num_processes
#  pool = mp.Pool(num_processes)
#  params = pool.map(bayesopt_under_true_model, range(num_processes))
#  params_dict = {str(i): params[i].tolist() for i in range(len(params))}
#  with open('bayes-opt-glucose.yml', 'w') as handle:
#    yaml.dump(params_dict, handle)
  #print(bayesopt_under_true_model(T=50,info=False))
  #print(bayesopt_under_true_model(T=50,info=True))
#  print(rollout_under_true_model(np.array([1.  ,50.,   0.1]), mdp_grid_epsilon_policy, 
#                             50, tuned_bandit.expit_epsilon_decay, 0.9, 20))
  elapsed_time = time.time() - start_time
  print("time {}".format(elapsed_time))
  
  
  