import sys
import pdb
import os
this_dir = os.path.dirname(os.path.abspath(__file__))
project_dir = os.path.join(this_dir, '..', '..')
sys.path.append(project_dir)


import matplotlib.pyplot as plt
from src.policies import tuned_bandit_policies as tuned_bandit
from src.environments.policy_iteration import policy_iteration
from src.environments.grid import Gridworld
  
import numpy as np
from functools import partial
import datetime
import yaml
import multiprocessing as mp

from bayes_opt import BayesianOptimization
import time


def mdp_grid_epsilon_policy(optimal_action, tuning_function, tuning_function_parameter, time_horizon, t, env):
  epsilon = tuning_function(time_horizon, t, tuning_function_parameter)
  if np.random.rand() < epsilon:
    optimal_action = np.random.choice(env.NUM_ACTION)
  return optimal_action

def mdp_grid_rollout(tuning_function_parameter, mdp_grid_epsilon_policy, 
                             time_horizon, tuning_function, env, 
                             gamma, mc_replicates):
  mean_cumulative_reward = 0
  nS, nA, nAdj = env.posterior_alpha.shape
  transition_prob = np.zeros((nS, nA, nAdj))
  for rep in range(mc_replicates):
    ## Sample from posterior distribution to get the transition probability and then get the sim environment
    for i in range(nS):
      for j in range(nA):
        transition_prob[i, j, :] = np.random.dirichlet(env.posterior_alpha[i, j, :])
    transitionMatrices = env.convert_to_transitionMatrices(transition_prob)
    sim_env = Gridworld(transitionMatrices=transitionMatrices)
    r = 0
    while sim_env.counter < sim_env.time_horizon:      
      s = sim_env.reset()
      for t in range(sim_env.maxT):
        update_transitionMatrices = sim_env.posterior_mean_model()
        mdp = Gridworld(transitionMatrices=update_transitionMatrices)
        _, _, Q = policy_iteration(mdp)
        optimal_action = np.argmax(Q[s, ])
        action = mdp_grid_epsilon_policy(optimal_action, tuning_function, tuning_function_parameter, 
                                         sim_env.time_horizon, sim_env.counter, sim_env)
        _, reward, done = sim_env.step(action)
        r += reward
        if done:
          break
      
    mean_cumulative_reward += (r - mean_cumulative_reward)/(rep+1)
  return mean_cumulative_reward

def bayesopt(rollout_function, policy, tuning_function, zeta_prev, time_horizon, env, mc_replicates,
             bounds, explore_, gamma):

  def objective(zeta0, zeta1, zeta2):
    zeta = np.array([zeta0, zeta1, zeta2])
    return rollout_function(zeta, policy, time_horizon, tuning_function, 
                            env, gamma, mc_replicates)

  explore_.update({'zeta{}'.format(i): [zeta_prev[i]] for i in range(len(zeta_prev))})
  bo = BayesianOptimization(objective, bounds)
  bo.explore(explore_)
  bo.maximize(init_points=10, n_iter=15, alpha=1e-4)
  best_param = bo.res['max']['max_params']
  best_param = np.array([best_param['zeta{}'.format(i)] for i in range(len(bounds))])
  return best_param


def rollout_under_true_model(tuning_function_parameter, mdp_grid_epsilon_policy, 
                             time_horizon, tuning_function,gamma, mc_replicates):
  env = Gridworld(time_horizon=time_horizon)
  mean_cumulative_reward = 0
  done=False
  for rep in range(mc_replicates):
    r = 0
    while env.counter < env.time_horizon:
      s = env.reset()
      actions =[]
      for t in range(env.maxT):
        update_transitionMatrices = env.posterior_mean_model()
        mdp = Gridworld(transitionMatrices=update_transitionMatrices)
        _, _, Q = policy_iteration(mdp)
        optimal_action = np.argmax(Q[s, ])
        action = mdp_grid_epsilon_policy(optimal_action, tuning_function, tuning_function_parameter, 
                                         env.time_horizon, env.counter, env)   
#        print(optimal_action)
        s, reward, done = env.step(action)
        actions.append(action)
        r += reward
        if done:
#          if t < env.maxT:
#            print(t, actions)
          break
    
    mean_cumulative_reward += (r - mean_cumulative_reward)/(rep+1)
  return mean_cumulative_reward

def bayesopt_under_true_model(T):
  rollout_function = rollout_under_true_model
  policy = mdp_grid_epsilon_policy    
  bounds = {'zeta0': (0.05, 2.0), 'zeta1': (1,75 ), 'zeta2': (0.01, 2.5)}
  tuning_function = tuned_bandit.expit_epsilon_decay
  
  def objective(zeta0, zeta1, zeta2):
    zeta = np.array([zeta0, zeta1, zeta2])
    return rollout_function(zeta, policy, T, tuning_function, 0.9, 10)
  
  # bounds = {'zeta{}'.format(i): (lower_bound, upper_bound) for i in range(10)}
#  explore_ = {'zeta0': [1.0, 0.05, 1.0, 0.1], 'zeta1': [50.0, 49.0, 1.0, 49.0], 'zeta2': [0.1, 2.5, 1.0, 2.5]}
  explore_ = {'zeta0': [1.0, 0.05, 1.0, 0.1], 'zeta1': [75.0, 75.0, 1.0, 75.0], 'zeta2': [0.1, 2.5, 1.0, 2.5]}
  bo = BayesianOptimization(objective, bounds)
  bo.explore(explore_)
  bo.maximize(init_points=10, n_iter=15, alpha=1e-4)
  best_param = bo.res['max']['max_params']
  best_param = np.array([best_param['zeta{}'.format(i)] for i in range(len(bounds))])
  return best_param


def episode(policy_name, label, mc_replicates=10, T=1000):
  np.random.seed(label)
  if policy_name == 'eps':
    tuning_function = lambda a, b, c: 0.1  # Constant epsilon
    tune = False
    tuning_function_parameter = None
  elif policy_name == "eps-decay":
    tuning_function = tuned_bandit.expit_epsilon_decay
    tune = True
    tuning_function_parameter = np.array([0.05, 1.0, 0.01]) 
    bounds = {'zeta0': (0.8, 2.0), 'zeta1': (1.0, 49.0), 'zeta2': (0.01, 2.5)}
    explore_ = {'zeta0': [1.0, 0.05, 1.0, 0.1], 'zeta1': [50.0, 49.0, 1.0, 49.0], 'zeta2': [0.1, 2.5, 1.0, 2.5]}
    rollout_function = mdp_grid_rollout
  elif policy_name == 'eps-fixed-decay':
    #tuning_function = lambda a, b, c: 0.1/float(b+1)
    #tune = False
    #tuning_function_parameter = None
    tuning_function = tuned_bandit.expit_epsilon_decay
    tune = False
    tuning_function_parameter = np.array([0.0500,  26.9969 ,  0.0100] )
#    tuning_function_parameter = np.array([ 2., 41.68182633, 2.5])

  policy = mdp_grid_epsilon_policy    
  gamma = 0.9
  env = Gridworld(time_horizon=T)
  r = 0
  time_horizon = T
  tuning_parameter_sequence = []
  rewards = []
  actions = []
  nEpi = 0
  while env.counter < env.time_horizon:
    s = env.reset()
    for t in range(env.maxT):
      if tune:
        tuning_function_parameter = bayesopt(rollout_function, policy, tuning_function, tuning_function_parameter, 
                                             time_horizon, env, mc_replicates, bounds, explore_, gamma)
        tuning_parameter_sequence.append([float(z) for z in tuning_function_parameter]) 
      update_transitionMatrices = env.posterior_mean_model()
      mdp = Gridworld(transitionMatrices=update_transitionMatrices)
      _, _, Q = policy_iteration(mdp)
      optimal_action = np.argmax(Q[s, ])
      action = mdp_grid_epsilon_policy(optimal_action, tuning_function, tuning_function_parameter, 
                                       env.time_horizon, env.counter, env)
      s, reward, done = env.step(action)
      rewards.append(reward)
      actions.append(action)
      r += reward
      if done:
#        if t < env.maxT:
#          print(t)
#          nEpi += 1 # the number of episodes where the agent gets the terminal state in less than maxT time steps
        break
#  print(nEpi)
  return {'rewards':rewards, 'cum_rewards': sum(rewards), 'zeta_sequence': tuning_parameter_sequence,
          'actions': actions}
    

def run(policy_name, save=True, mc_replicates=10, T=1000):
  """

  :return:
  """

#  replicates = 48
  num_cpus = int(mp.cpu_count())
#  num_cpus = 4
  replicates = 4
  results = []
  pool = mp.Pool(processes=num_cpus)

  episode_partial = partial(episode, policy_name, mc_replicates=mc_replicates, T=T)

  results = pool.map(episode_partial, range(replicates))
#  cumulative_regrets = [np.float(d['cumulative_regret']) for d in results]
  zeta_sequences = [list(d['zeta_sequence']) for d in results]
  actions = [list(d['actions']) for d in results]
  cum_rewards = [float(d['cum_rewards']) for d in results]
#  rewards = [list(d['rewards'].astype(float)) for d in results]
#  print(policy_name, cum_rewards)
  print(policy_name, 'rewards', float(np.mean(cum_rewards)), 'se_rewards',float(np.std(cum_rewards))/np.sqrt(replicates))
  # Save results
  if save:
    results = {'T':T, 'mc_replicates': mc_replicates, 'cum_rewards': cum_rewards, 
               'rewards': float(np.mean(cum_rewards)), 'se_rewards':float(np.std(cum_rewards)/np.sqrt(replicates)),
               'zeta_sequences': zeta_sequences, 'actions': actions}#, 'rewards':rewards}

    base_name = 'mdp-grid-{}'.format(policy_name)
    prefix = os.path.join(project_dir, 'src', 'environments', base_name)
    suffix = datetime.datetime.now().strftime("%y%m%d_%H%M%S")
    filename = '{}_{}.yml'.format(prefix, suffix)
    np.save('{}_{}'.format(prefix, suffix), results)
    with open(filename, 'w') as outfile:
      yaml.dump(results, outfile)

  return


if __name__ == '__main__':
  start_time = time.time()
#  check_coef_converge()
#  episode('eps-decay', 0, T=100)
#  episode('eps-fixed-decay', 0, T=1000)
#  run('eps-decay', T=25)
  run('eps-fixed-decay', save=False, T=150)
  run('eps', save=False, T=150)
#  episode('eps', 0, T=200)
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
#  print(bayesopt_under_true_model(T=150))
  elapsed_time = time.time() - start_time
  print("time {}".format(elapsed_time))
  
  
  
  
  
  
  
  
  