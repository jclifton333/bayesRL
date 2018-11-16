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

from bayes_opt import BayesianOptimization
import time


def mab_rollout(tuning_function_parameter, mab_epsilon_policy, 
                             time_horizon, tuning_function, env, 
                             gamma, mc_reps, est_means, std_errs):
  mean_cumulative_reward = 0
  R = std_errs[1]/std_errs[0]
  delta = est_means[1] - est_means[0]
  a = 0
  expit_wrapper = partial(tuning_function, R=R, delta=delta)
  for rep in range(mc_reps):
    r = 0
    for j in range(time_horizon):
      a = mab_epsilon_policy(est_means, std_errs, 1, expit_wrapper, tuning_function_parameter, time_horizon, j, env)
      r += gamma**j*env.reward_dbn(a)[0]
    mean_cumulative_reward += (r - mean_cumulative_reward)/(rep+1)
    #print(rep, r, mean_cumulative_reward)
  return mean_cumulative_reward

# Modify this and bayesopt_under_true_model to work for MABs
def rollout_under_true_model(tuning_function_parameter, mab_epsilon_policy, 
                             time_horizon, tuning_function, gamma, mc_replicates):
#  env = Gridworld(time_horizon=time_horizon)
#  print(time_horizon)
  mean_cumulative_reward = 0
#  done=False
  for rep in range(mc_replicates):
    env = Bandit.NormalMAB(list_of_reward_mus=[[0.3],[0.6]])
    r = 0
    delta = env.list_of_reward_mus[1][0]-env.list_of_reward_mus[0][0]
    R = env.list_of_reward_vars[1][0]/env.list_of_reward_vars[0][0]
    if tuning_function==tuned_bandit.lm_expit_epsilon_decay:
      tuning_function = partial(tuning_function, R=R, delta=delta)
    for t in range(time_horizon):  
#      print(env.counter, r)
      action = mab_epsilon_policy(env.list_of_reward_mus,[],1, tuning_function, tuning_function_parameter, 
                                       time_horizon, t, env)
      reward = env.step(action)
      r += gamma**rep*reward['Utility'][0]
#  print(r)
    mean_cumulative_reward += (r - mean_cumulative_reward)/(rep+1)
#    print(rep, r, mean_cumulative_reward)
  return mean_cumulative_reward

def bayesopt_under_true_model(T):
  rollout_function = rollout_under_true_model
  policy = tuned_bandit.mab_epsilon_greedy_policy
  bounds = {'zeta0': (0.05,2.0),'zeta1': (-1.0,1.0),'zeta2': (-1.0,1.0),'zeta3': (-1.0,1.0),'zeta4': (-1.0,1.0),'zeta5': (-1.0,1.0),'zeta6': (-1.0,1.0),'zeta7': (-1.0,1.0), 'zeta8': (-1.0,1.0)}
  tuning_function = tuned_bandit.lm_expit_epsilon_decay
  
  def objective(zeta0, zeta1, zeta2, zeta3, zeta4, zeta5, zeta6, zeta7, zeta8):
    zeta = np.array([zeta0, zeta1, zeta2, zeta3, zeta4, zeta5, zeta6, zeta7, zeta8])
    return rollout_function(zeta, policy, T, tuning_function, 1, 10)

  # bounds = {'zeta{}'.format(i): (lower_bound, upper_bound) for i in range(10)}
#  explore_ = {'zeta0': [1.0, 0.05, 1.0, 0.1, 0.05], 'zeta1': [50.0, 49.0, 1.0, 49.0, 1.0], 
#              'zeta2': [0.1, 2.5, 1.0, 2.5, 2.5]}
  explore_ = {'zeta0': [0.05,1.0],'zeta1': [0.0,0.0],'zeta2': [0.0,0.0],'zeta3': [0.0,0.0],'zeta4': [0.0,0.0],'zeta5': [0.0,0.0],'zeta6': [0.0,0.0],'zeta7': [0.0,0.0], 'zeta8': [0.0,0.0]}
  bo = BayesianOptimization(objective, bounds)
  bo.explore(explore_)
  bo.maximize(init_points=100, n_iter=15, alpha=1e-4)
  best_param = bo.res['max']['max_params']
  best_param = np.array([best_param['zeta{}'.format(i)] for i in range(len(bounds))])
  return best_param

def mab_bayesopt(rollout_function, policy, tuning_function, zeta_prev, time_horizon, env, mc_replicates,
             bounds, explore_, gamma, est_means, std_errs):

  def objective(zeta0, zeta1, zeta2, zeta3, zeta4, zeta5, zeta6, zeta7, zeta8):
    zeta = np.array([zeta0, zeta1, zeta2, zeta3, zeta4, zeta5, zeta6, zeta7, zeta8])
    return rollout_function(zeta, policy, time_horizon, tuning_function, 
                            env, gamma, mc_replicates, est_means, std_errs)

  #print(objective(0,0,0,0,0,0,0,0,0))
  explore_.update({'zeta{}'.format(i): [zeta_prev[i]] for i in range(len(zeta_prev))})
  bo = BayesianOptimization(objective, bounds)
  bo.explore(explore_)
  bo.maximize(init_points=10, n_iter=15, alpha=1e-4)
  best_param = bo.res['max']['max_params']
  best_param = np.array([best_param['zeta{}'.format(i)] for i in range(len(bounds))])
  return best_param

def episode(policy_name, label, mc_replicates=10, T=1000):
  #np.random.seed(label)
  if policy_name == 'eps':
    tuning_function = lambda a, b, c: 0.0  # Constant epsilon
    tune = False
    tuning_function_parameter = None
  elif policy_name == "eps-decay":
    #tuning_function = tuned_bandit.expit_epsilon_decay
    tune = True
    #tuning_function_parameter = np.array([0.05, 1.0, 0.01]) 
    #bounds = {'zeta0': (0.05, 2.0), 'zeta1': (1.0, 49.0), 'zeta2': (0.01, 2.5)}
    #explore_ = {'zeta0': [1.0, 0.05, 1.0, 0.1], 'zeta1': [50.0, 49.0, 1.0, 49.0], 'zeta2': [0.1, 2.5, 1.0, 2.5]}
    tuning_function_parameter = np.concatenate(([0.05],np.random.uniform(-0.5,0.5,8)))
    bounds = {'zeta0': (0.05,2.0),'zeta1': (-1.0,1.0),'zeta2': (-1.0,1.0),'zeta3': (-1.0,1.0),'zeta4': (-1.0,1.0),'zeta5': (-1.0,1.0),'zeta6': (-1.0,1.0),'zeta7': (-1.0,1.0), 'zeta8': (-1.0,1.0)}
    explore_ = {'zeta0': [0.05,1.0],'zeta1': [0.0,0.0],'zeta2': [0.0,0.0],'zeta3': [0.0,0.0],'zeta4': [0.0,0.0],'zeta5': [0.0,0.0],'zeta6': [0.0,0.0],'zeta7': [0.0,0.0], 'zeta8': [0.0,0.0]}
    rollout_function = mab_rollout
    tuning_function = tuned_bandit.lm_expit_epsilon_decay
  elif policy_name == 'eps-fixed-decay':
    #tuning_function = lambda a, b, c: 0.1/float(b+1)
    #tune = False
    #tuning_function_parameter = None
    tuning_function = tuned_bandit.lm_expit_epsilon_decay
    tune = False
    tuning_function_parameter = np.array([ 0.4652, -0.6916, -0.4979, 0.3094, -0.7081, -0.8968, -0.1557, -0.8516, 0.9501 ] )
    #tuning_function_parameter = np.array([ 0.05     ,  43.46014702 , 2.5 ] )
#    tuning_function_parameter = np.array([ 2., 41.68182633, 2.5])

  policy = tuned_bandit.mab_epsilon_greedy_policy
  gamma = 1
  env = Bandit.NormalMAB(list_of_reward_mus=[[0.3],[0.6]])
  time_horizon = T
  tuning_parameter_sequence = []
  rewards = []
  actions = []
  act_count = [0,0]
  posterior_alphas = []
  posterior_betas = []
  posterior_lambdas = []  
  posterior_mus = []
  r = 0
  est_means = [0,0]
  sse = [1e-8,1e-8]
  if tune:
      #tuning_function_parameter = mab_bayesopt(rollout_function, policy, tuning_function, tuning_function_parameter, 
                                               #time_horizon, env, mc_replicates, bounds, explore_, gamma, est_means, np.divide(sse,np.add(act_count,-1)))
    tuning_function_parameter = bayesopt_under_true_model(time_horizon)            
    tuning_parameter_sequence.append([float(z) for z in tuning_function_parameter])
    
  for t in range(time_horizon):
#    print(env.counter, r)
#    acts=[]
#    for t in range(env.maxT):
#      print(env.counter, t, sum(rewards))
    #if tune and t <= 4:
    #  if t == 1 or t == 2:
    #    action = 1
    #  else:
    #    action = 0
    #else:
    #  if tune and t == 5:
    if t <= 4:
      if t == 1 or t == 2:
        action = 1
      else:
        action = 0
    else:
      if tuning_function == tuned_bandit.lm_expit_epsilon_decay:
        action = policy(est_means, sse, 1, partial(tuning_function,R=sse[1]*(act_count[0]-1)/(sse[0]*(act_count[1]-1)),delta=est_means[1]-est_means[0]), tuning_function_parameter, T,t,env)
      else:
        action = policy(est_means, sse, 1, tuning_function, tuning_function_parameter, T,t,env)
#      print("estimated {}, true {}".format(update_transitionMatrices[:,0,:], env.transitionMatrices[:,0,:]))
      
    print("###########")
#      print(env.counter, update_transitionMatrices[1,:4,:], env.transitionMatrices[1,:4,:])
#      print(env.counter, sum(sum(abs(update_transitionMatrices[1,:3,:] - env.transitionMatrices[1,:3,:]))))
#      print(env.counter, sum(sum(abs(update_transitionMatrices[2,[3,7,10],:] - env.transitionMatrices[2,[3,7,10],:]))))
#      print(env.counter, sum(sum(sum(abs(update_transitionMatrices - env.transitionMatrices)))))
      
    reward = env.step(action)
    if reward.__class__.__name__=='dict':
      reward = reward['Utility'][0]
    rewards.append(reward)
#    acts.append(action)
    actions.append(action)
    new_post_alphas = [env.posterior_params_dict[i]['alpha_post'] for i in env.posterior_params_dict.keys()]
    new_post_betas = [env.posterior_params_dict[i]['beta_post'] for i in env.posterior_params_dict.keys()]
    new_post_lambdas = [env.posterior_params_dict[i]['lambda_post'] for i in env.posterior_params_dict.keys()]
    new_post_mus = [env.posterior_params_dict[i]['mu_post'] for i in env.posterior_params_dict.keys()]
    posterior_alphas.append(new_post_alphas)
    posterior_betas.append(new_post_betas)
    posterior_lambdas.append(new_post_lambdas)
    posterior_mus.append(new_post_mus)
#      print("after: {}".format(posterior_alphas))
    r += reward
    #print(rewards)
    #print(est_means)
    #print(actions)
    act_count[action] += 1
    err = reward-est_means[action]
    est_means[action] += err/act_count[action]
    sse[action] += err * (reward-est_means[action])
#      if done:
#        print(acts)
#        print(env.counter, r)
#        break
  print(sum(rewards))
  print(tuning_parameter_sequence)
  return {'rewards':rewards, 'cum_rewards': sum(rewards), 'zeta_sequence': tuning_parameter_sequence,
          'actions': actions, 'posterior_alphas': posterior_alphas, 'posterior_betas': posterior_betas,
          'posterior_lambdas': posterior_lambdas, 'posterior_mus': posterior_mus}
    
def run(policy_name, save=True, mc_replicates=10, T=1000):
  """

  :return:
  """

#  replicates = 48
  num_cpus = int(mp.cpu_count())
#  num_cpus = 4
  replicates = 20
  results = []
  pool = mp.Pool(processes=num_cpus)

  episode_partial = partial(episode, policy_name, mc_replicates=mc_replicates, T=T)

  results = pool.map(episode_partial, range(replicates))
  #results = episode_partial(1)
#  cumulative_regrets = [np.float(d['cumulative_regret']) for d in results]
  zeta_sequences = [list(d['zeta_sequence']) for d in results]
  actions = [list(d['actions']) for d in results]
  cum_rewards = [float(d['cum_rewards']) for d in results]
  posterior_alphas = [d['posterior_alphas']for d in results]
  posterior_betas = [d['posterior_betas']for d in results]
  posterior_lambdas = [d['posterior_lambdas']for d in results]
  posterior_mus = [d['posterior_mus']for d in results]
#  rewards = [list(d['rewards'].astype(float)) for d in results]
#  print(policy_name, cum_rewards)
  print(policy_name, 'rewards', float(np.mean(cum_rewards)), 'se_rewards',float(np.std(cum_rewards))/np.sqrt(replicates))
  # Save results
  if save:
    results = {'T':T, 'mc_replicates': mc_replicates, 'cum_rewards': cum_rewards, 
               'rewards': float(np.mean(cum_rewards)), 'se_rewards':float(np.std(cum_rewards)/np.sqrt(replicates)),
               'zeta_sequences': zeta_sequences, 'actions': actions, 
               'posterior_alphas': posterior_alphas, 'posterior_betas': posterior_betas,
               'posterior_lambdas': posterior_lambdas, 'posterior_mus': posterior_mus}#, 'rewards':rewards}

    base_name = 'mab-{}'.format(policy_name)
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
#  episode('eps-decay', 0, T=75)
#  episode('eps-fixed-decay', 2, T=50)
#  run('eps-decay', save=True, T=10)
  run('eps-fixed-decay', save=False, T=75)
#  run('eps', save=False, T=50)
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
#  print(bayesopt_under_true_model(T=75))
#  print(rollout_under_true_model(np.array([1.  ,50.,   0.1]), mdp_grid_epsilon_policy, 
#                             50, tuned_bandit.expit_epsilon_decay, 0.9, 20))
  elapsed_time = time.time() - start_time
  print("time {}".format(elapsed_time))
  
  
  