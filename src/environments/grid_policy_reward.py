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
import sklearn

from bayes_opt import BayesianOptimization
import time


def mdp_grid_epsilon_policy(optimal_action, tuning_function, tuning_function_parameter, time_horizon, t, env):
  epsilon = tuning_function(time_horizon, t, tuning_function_parameter)
  if np.random.rand() < epsilon:
    optimal_action = np.random.choice(env.NUM_ACTION)
  return optimal_action

def initial_reward_mat(nS, init_reward=75):
  return  np.ones((nS))*init_reward

def mdp_grid_rollout(tuning_function_parameter, mdp_grid_epsilon_policy, 
                             time_horizon, tuning_function, env, 
                             gamma, mc_replicates, reward_mat):
  mean_cumulative_reward = 0
  nS, nA, nAdj = env.posterior_alpha.shape
  transition_prob = np.zeros((nS, nA, nAdj))
  for rep in range(mc_replicates):
    ## Sample from posterior distribution to get the transition probability and then get the sim environment
    for i in range(nS):
      for j in range(nA):
        transition_prob[i, j, :] = np.random.dirichlet(env.posterior_alpha[i, j, :])
    transitionMatrices = env.convert_to_transitionMatrices(transition_prob)
#    transitionMatrices = env.transitionMatrices
    sim_env = Gridworld(transitionMatrices=transitionMatrices, time_horizon=time_horizon, reward_mat=reward_mat)
    r = 0
    update_reward_mat = initial_reward_mat(nS=nS)
    while sim_env.counter < sim_env.time_horizon:  
#      print(sim_env.counter)
      s = sim_env.reset()
      for t in range(sim_env.maxT):
        update_transitionMatrices = sim_env.posterior_mean_model()
        mdp = Gridworld(transitionMatrices=update_transitionMatrices, time_horizon=time_horizon, reward_mat=update_reward_mat)
        _, pis, Q = policy_iteration(mdp)
        optimal_action = pis[-1][s]
#        optimal_action = np.argmax(Q[s, ])
        action = mdp_grid_epsilon_policy(optimal_action, tuning_function, tuning_function_parameter, 
                                         sim_env.time_horizon, sim_env.counter, sim_env)
        s, reward, done = sim_env.step(action)
        update_reward_mat[s] = reward 
        r += reward
        if done:
          break
      
    mean_cumulative_reward += (r - mean_cumulative_reward)/(rep+1)
#    print(rep, r, mean_cumulative_reward)
  return mean_cumulative_reward

def bayesopt(rollout_function, policy, tuning_function, zeta_prev, time_horizon, env, mc_replicates,
             bounds, explore_, gamma, reward_mat):

  def objective(zeta0, zeta1, zeta2):
    zeta = np.array([zeta0, zeta1, zeta2])
    return rollout_function(zeta, policy, time_horizon, tuning_function, 
                            env, gamma, mc_replicates, reward_mat)

  explore_.update({'zeta{}'.format(i): [zeta_prev[i]] for i in range(len(zeta_prev))})
  bo = BayesianOptimization(objective, bounds)
  bo.explore(explore_)
  bo.maximize(init_points=10, n_iter=15, alpha=1e-4)
  best_param = bo.res['max']['max_params']
  best_param = np.array([best_param['zeta{}'.format(i)] for i in range(len(bounds))])
  return best_param


def rollout_under_true_model(tuning_function_parameter, mdp_grid_epsilon_policy, 
                             time_horizon, tuning_function,gamma, mc_replicates):
#  env = Gridworld(time_horizon=time_horizon)
#  print(time_horizon)
  mean_cumulative_reward = 0
#  done=False
  for rep in range(mc_replicates):
    env = Gridworld(time_horizon=time_horizon)
    r = 0
    update_reward_mat = initial_reward_mat(nS=env.NUM_STATE)
    while env.counter < env.time_horizon:  
#      print(env.counter, r)
      s = env.reset()
      for t in range(env.maxT):
        update_transitionMatrices = env.posterior_mean_model()
        mdp = Gridworld(transitionMatrices=update_transitionMatrices, time_horizon=time_horizon, reward_mat=update_reward_mat)
        _, pis, Q = policy_iteration(mdp)
#        optimal_action = np.argmax(Q[s, ])
        optimal_action = pis[-1][s]
        action = mdp_grid_epsilon_policy(optimal_action, tuning_function, tuning_function_parameter, 
                                         env.time_horizon, env.counter, env)
        s, reward, done = env.step(action)
        update_reward_mat[s] = reward 
        r += reward
        if done:
          break
#    print(r)  
    mean_cumulative_reward += (r - mean_cumulative_reward)/(rep+1)
#    print(rep, r, mean_cumulative_reward)
  return mean_cumulative_reward

def bayesopt_under_true_model(T):
  rollout_function = rollout_under_true_model
  policy = mdp_grid_epsilon_policy  
#  bounds = {'zeta0': (0.05, 2.0), 'zeta1': (1.0, 49.0 ), 'zeta2': (0.01, 2.5)}  
  bounds = {'zeta0': (0.05, 2.0), 'zeta1': (1.0, 75.0 ), 'zeta2': (0.01, 2.5)}
#  bounds = {'zeta0': (0.05, 2.0), 'zeta1': (1.0, 150.0 ), 'zeta2': (0.01, 2.5)}
  tuning_function = tuned_bandit.expit_epsilon_decay
  
  def objective(zeta0, zeta1, zeta2):
    zeta = np.array([zeta0, zeta1, zeta2])
    return rollout_function(zeta, policy, T, tuning_function, 0.9, 10)
  print(objective(0,1,0))
#  explore_ = {'zeta0': [1.0, 0.05, 1.0, 0.1, 0.05], 'zeta1': [50.0, 49.0, 1.0, 49.0, 1.0], 
#              'zeta2': [0.1, 2.5, 1.0, 2.5, 2.5]}
  explore_ = {'zeta0': [1.0, 0.05, 1.0, 0.1, 2.0, 0.33993149], 'zeta1': [75.0, 74.0, 1.0, 74.0, 75.0, 74.95364242], 
              'zeta2': [0.1, 2.5, 1.0, 2.5, 0.1, 0.07794554]}
#  explore_ = {'zeta0': [1.0, 0.05, 1.0, 0.1], 'zeta1': [150.0, 149.0, 1.0, 149.0], 'zeta2': [0.1, 2.5, 1.0, 2.5]}
  bo = BayesianOptimization(objective, bounds)
  bo.explore(explore_)
  bo.maximize(init_points=10, n_iter=15, alpha=1e-4)
  best_param = bo.res['max']['max_params']
  best_param = np.array([best_param['zeta{}'.format(i)] for i in range(len(bounds))])
  return best_param


def episode(policy_name, label, mc_replicates=10, T=1000):
  np.random.seed(label)
  if policy_name == 'eps':
    tuning_function = lambda a, b, c: 0.0 # Constant epsilon
    tune = False
    tuning_function_parameter = None
  elif policy_name == "eps-decay":
    tuning_function = tuned_bandit.expit_epsilon_decay
    tune = True
    tuning_function_parameter = np.array([0.05, 1.0, 0.01]) 
    explore_ = {'zeta0': [1.0, 0.05, 1.0, 0.1, 2.0, 0.33993149], 'zeta1': [75.0, 74.0, 1.0, 74.0, 75.0, 74.95364242], 
              'zeta2': [0.1, 2.5, 1.0, 2.5, 0.1, 0.07794554]}
    bounds = {'zeta0': (0.05, 2.0), 'zeta1': (1.0, 75.0 ), 'zeta2': (0.01, 2.5)}
#    bounds = {'zeta0': (0.05, 2.0), 'zeta1': (1.0, 49.0), 'zeta2': (0.01, 2.5)}
#    explore_ = {'zeta0': [1.0, 0.05, 1.0, 0.1], 'zeta1': [50.0, 49.0, 1.0, 49.0], 'zeta2': [0.1, 2.5, 1.0, 2.5]}
    rollout_function = mdp_grid_rollout
  elif policy_name == 'eps-fixed-decay':
    tuning_function = lambda a, b, c: 0.1/float(b+1)
    tune = False
    tuning_function_parameter = None
#    tuning_function = tuned_bandit.expit_epsilon_decay
#    tune = False
#    tuning_function_parameter = np.array([0.33993149, 74.95364242,  0.07794554])
#    tuning_function_parameter = np.array([5.47368649e+00, 1.23051916e+02, 6.69618550e-02])

  policy = mdp_grid_epsilon_policy    
  gamma = 0.9
  env = Gridworld(time_horizon=T)
  time_horizon = T
  tuning_parameter_sequence = []
  rewards = []
  actions = []
  posterior_alphas = []
  r = 0
  update_reward_mat = initial_reward_mat(nS=env.NUM_STATE)
  while env.counter < env.time_horizon:
#    print(env.counter, r)
    s = env.reset()
    acts=[]
    for t in range(env.maxT):
#      print(env.counter, t, sum(rewards))
      if tune:
        tuning_function_parameter = bayesopt(rollout_function, policy, tuning_function, tuning_function_parameter, 
                                             time_horizon, env, mc_replicates, bounds, explore_, gamma, update_reward_mat)
        tuning_parameter_sequence.append([float(z) for z in tuning_function_parameter]) 
      update_transitionMatrices = env.posterior_mean_model()
#      print("estimated {}, true {}".format(update_transitionMatrices[:,0,:], env.transitionMatrices[:,0,:]))
#      print("###########")
#      print(env.counter, update_transitionMatrices[1,:4,:], env.transitionMatrices[1,:4,:])
#      print(env.counter, sum(sum(abs(update_transitionMatrices[1,:3,:] - env.transitionMatrices[1,:3,:]))))
#      print(env.counter, sum(sum(abs(update_transitionMatrices[2,[3,7,10],:] - env.transitionMatrices[2,[3,7,10],:]))))
#      print(env.counter, sum(sum(sum(abs(update_transitionMatrices - env.transitionMatrices)))))
      
      mdp = Gridworld(transitionMatrices=update_transitionMatrices, time_horizon=time_horizon, reward_mat=update_reward_mat)
      _, pis, Q = policy_iteration(mdp)
#      optimal_action = np.argmax(Q[s, ])
      optimal_action = pis[-1][s]
      action = mdp_grid_epsilon_policy(optimal_action, tuning_function, tuning_function_parameter, 
                                       env.time_horizon, env.counter, env)
      s, reward, done = env.step(action)
      update_reward_mat[s] = reward
      rewards.append(reward)
      acts.append(optimal_action)
      actions.append(action)
      new_post_alpha = env.posterior_alpha.copy()
      posterior_alphas.append(new_post_alpha)
#      print("after: {}".format(posterior_alphas))
      r += reward
      if done:
#        print(acts)
#        print(env.counter, r)
        break
  print(sum(rewards))        
  return {'rewards':rewards, 'cum_rewards': sum(rewards), 'zeta_sequence': tuning_parameter_sequence,
          'actions': actions, 'posterior_alphas': posterior_alphas}
    

def run(policy_name, save=True, mc_replicates=10, T=1000):
  """

  :return:
  """

  replicates = 24
#  num_cpus = int(mp.cpu_count())
  num_cpus = 24
#  replicates = 24
  results = []
  pool = mp.Pool(processes=num_cpus)

  episode_partial = partial(episode, policy_name, mc_replicates=mc_replicates, T=T)

  results = pool.map(episode_partial, range(replicates))
#  cumulative_regrets = [np.float(d['cumulative_regret']) for d in results]
  zeta_sequences = [list(d['zeta_sequence']) for d in results]
  actions = [list(d['actions']) for d in results]
  cum_rewards = [float(d['cum_rewards']) for d in results]
  posterior_alphas = [d['posterior_alphas']for d in results]
#  rewards = [list(d['rewards'].astype(float)) for d in results]
#  print(policy_name, cum_rewards)
  print(policy_name, 'rewards', float(np.mean(cum_rewards)), 'se_rewards',float(np.std(cum_rewards))/np.sqrt(replicates))
  # Save results
  if save:
    results = {'T':T, 'mc_replicates': mc_replicates, 'cum_rewards': cum_rewards, 
               'rewards': float(np.mean(cum_rewards)), 'se_rewards':float(np.std(cum_rewards)/np.sqrt(replicates)),
               'zeta_sequences': zeta_sequences, 'actions': actions, 
               'posterior_alphas': posterior_alphas}#, 'rewards':rewards}

    base_name = 'mdp-grid-{}'.format(policy_name)
    prefix = os.path.join(project_dir, 'src', 'environments', base_name)
    suffix = datetime.datetime.now().strftime("%y%m%d_%H%M%S")
    filename = '{}_{}.yml'.format(prefix, suffix)
#    np.save('{}_{}'.format(prefix, suffix), results)
    with open(filename, 'w') as outfile:
      yaml.dump(results, outfile)

  return


if __name__ == '__main__':
  start_time = time.time()
#  check_coef_converge()
#  episode('eps-decay', 0, T=75)
#  episode('eps-fixed-decay', 2, T=50)
#  run('eps-decay', save=True, T=75, mc_replicates=50)
#  run('eps-fixed-decay', save=False, T=75)
#  run('eps', save=False, T=75)
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
  print(bayesopt_under_true_model(T=75))
#  print(rollout_under_true_model(np.array([1.  ,50.,   0.1]), mdp_grid_epsilon_policy, 
#                             50, tuned_bandit.expit_epsilon_decay, 0.9, 20))
  elapsed_time = time.time() - start_time
  print("time {}".format(elapsed_time))
  
  
  
  
  
  
  
  
  