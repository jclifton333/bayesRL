'''
To find the optimal epsilon function form in a 2-arm Bernoulli bandit problem
'''
import sys
import pdb
import os
this_dir = os.path.dirname(os.path.abspath(__file__))
project_dir = os.path.join(this_dir, '..', '..')
sys.path.append(project_dir)

from src.environments.Bandit import BernoulliMAB
from src.policies import tuned_bandit_policies as tuned_bandit
import numpy as np
from functools import partial
import datetime
import yaml
import multiprocessing as mp
import sklearn
import matplotlib.pyplot as plt
from bayes_opt import BayesianOptimization
import time


def bern_mab_epsilon_greedy_policy(estimated_means, number_of_pulls, tuning_function,
                              tuning_function_parameter, T, t):
#  s1, s2 = estimated_means * number_of_pulls # number of successes for each arm
#  n1, n2 = number_of_pulls
#  epsilon = tuning_function(T, s1, s2, n1, n2, tuning_function_parameter)
  epsilon = tuning_function(T, t, tuning_function_parameter)
  if estimated_means[0] == estimated_means[1]:
    action = np.random.choice(2) # break the tie
  else:
    greedy_action = np.argmax(estimated_means)
    if np.random.random() < epsilon:
      action = np.random.choice(2)
    else:
      action = greedy_action
  return action
#  return action, s1, s2, epsilon


def cum_regret_one_combination(p1, p2, tuning_function_parameter, T=50, replicates=100):
#  tuning_function = tuned_bandit.bern_expit_epsilon_decay
  tuning_function = tuned_bandit.expit_epsilon_decay
  list_of_reward_mus = [p1, p2]
  env = BernoulliMAB(list_of_reward_mus=list_of_reward_mus)
  p_opt = np.max(list_of_reward_mus)
  
  cum_regret = 0
  for rep in range(replicates):
    np.random.seed(rep)
    env.reset()
    for a in range(env.number_of_actions):
      env.step(a)
      
    cum_regret_each_rep = 0
    for t in range(T):
      action = bern_mab_epsilon_greedy_policy(env.estimated_means, env.number_of_pulls, tuning_function,
                                  tuning_function_parameter, T, t)
#      action,_,_,_ = bern_mab_epsilon_greedy_policy(env.estimated_means, env.number_of_pulls, tuning_function,
#                                  tuning_function_parameter, T, t)
      res = env.step(action)
      regret = p_opt - list_of_reward_mus[action]
      cum_regret_each_rep += regret
    
    cum_regret += (cum_regret_each_rep - cum_regret)/float(rep+1)
  return cum_regret


def worst_regret(tuning_function_parameter): # to find the larget regret, i.e. the worst case
  largest_regret = -1
  p = np.linspace(0.05,0.95,10)
  cum_regret_list = []
  for p1 in p:
#    for p2 in p:
    for p2 in p[p<=p1]:
      cum_regret = cum_regret_one_combination(p1, p2, tuning_function_parameter)
      if largest_regret < cum_regret:
        largest_regret = cum_regret
        worst_p1, worst_p2 = p1, p2
        cum_regret_list.append(cum_regret)
  print(worst_p1, worst_p2)
  return -np.mean(cum_regret_list)
#  return -largest_regret
  

def bayesopt_max_min(seed):
  np.random.seed(seed)
  
#  bounds = {'zeta0': (0.05,2.0),'zeta1': (-5.0, 5.0),'zeta2': (-5.0,5.0),'zeta3': (-5.0,5.0),'zeta4': (-5.0,5.0),'zeta5': (-5.0,5.0),'zeta6': (-5.0,5.0),'zeta7': (-5.0,5.0), 'zeta8': (-5.0,5.0)}
#  explore_ = {'zeta0': [0.05,0.1,0.0,1.0, 0.1],'zeta1': [0.0,0.0,0.0,0.0,-122.5],'zeta2': [0.0,0.0,0.0,0.0,0.0],'zeta3': [0.0,0.0,0.0,0.0,0.0],'zeta4': [0.0,0.0,0.0,0.0,0.0],'zeta5': [0.0,0.0,0.0,0.0,2.5],'zeta6': [0.0,0.0,0.0,0.0,0.0],'zeta7': [0.0,0.0,0.0,0.0,0.0], 'zeta8': [0.0,0.0,0.0,0.0,0.0]}
#  bounds = {'zeta0': (0.05,2.0),'zeta1': (-50, 50),'zeta2': (0.01,2.0),'zeta3': (-1.0,1.0),
#            'zeta4': (-1.0,1.0),'zeta5': (-1.0,1.0),'zeta6': (-1.0,1.0)}
#  explore_ = {'zeta0': [0.05],'zeta1': [-50],'zeta2': [0.5],'zeta3': [0.5],'zeta4': [0.5],
#              'zeta5': [0.5],'zeta6': [0.5]}
#  def objective(zeta0, zeta1, zeta2, zeta3, zeta4, zeta5, zeta6):
#    zeta = np.array([zeta0, zeta1, zeta2, zeta3, zeta4, zeta5, zeta6])
#    return worst_regret(zeta)

  bounds = {'zeta0': (0.05, 2.0), 'zeta1': (1.0, 49.0), 'zeta2': (0.01, 2.5)}
  explore_ = {'zeta0': [1.0, 0.05, 1.0, 0.1], 'zeta1': [50.0, 49.0, 1.0, 49.0], 'zeta2': [0.1, 2.5, 1.0, 2.5]}

  def objective(zeta0, zeta1, zeta2):
    zeta = np.array([zeta0, zeta1, zeta2])
    return worst_regret(zeta)

  bo = BayesianOptimization(objective, bounds)
  bo.explore(explore_)
  bo.maximize(init_points=15, n_iter=15, alpha=1e-4)
  best_param = bo.res['max']['max_params']
  best_param = np.array([best_param['zeta{}'.format(i)] for i in range(len(bounds))])
  print(best_param)
  return best_param

def episode(policy_name, label, info=True, p1=0.7, p2=0.4, T=50):
  if policy_name == 'eps':
    eps = 0.05
    if info:
      tuning_function = lambda a, b, c, d, e: eps
    else:
      tuning_function = lambda a, b, c: eps # Constant epsilon
#    policy = mab_epsilon_greedy_policy
    tune = False
    tuning_function_parameter = None
  elif policy_name == 'greedy':
    if info:
      tuning_function = lambda a, b, c, d, e: 0.00
    else:
      tuning_function = lambda a, b, c: 0.00  # Constant epsilon
#    policy = mab_epsilon_greedy_policy
    tune = False
    tuning_function_parameter = None
  elif policy_name == "eps-decay":
    tune = True
    posterior_sample = True
#    policy = mab_epsilon_greedy_policy
    if info:
      tuning_function_parameter = np.concatenate(([0.05],np.random.uniform(-0.5,0.5,8)))
      bounds = {'zeta0': (0.05,2.0),'zeta1': (-5.0, 5.0),'zeta2': (-5.0,5.0),'zeta3': (-5.0,5.0),'zeta4': (-5.0,5.0),'zeta5': (-5.0,5.0),'zeta6': (-5.0,5.0),'zeta7': (-5.0,5.0), 'zeta8': (-5.0,5.0)}
      explore_ = {'zeta0': [0.05,0.1,0.0,1.0, 0.1],'zeta1': [0.0,0.0,0.0,0.0,-122.5],'zeta2': [0.0,0.0,0.0,0.0,0.0],'zeta3': [0.0,0.0,0.0,0.0,0.0],'zeta4': [0.0,0.0,0.0,0.0,0.0],'zeta5': [0.0,0.0,0.0,0.0,2.5],'zeta6': [0.0,0.0,0.0,0.0,0.0],'zeta7': [0.0,0.0,0.0,0.0,0.0], 'zeta8': [0.0,0.0,0.0,0.0,0.0]}
      tuning_function = tuned_bandit.information_expit_epsilon_decay
    else:
      tuning_function_parameter = np.array([0.05, 1.0, 0.01]) 
      bounds = {'zeta0': (0.05, 2.0), 'zeta1': (1.0, 49.0), 'zeta2': (0.01, 2.5)}
      explore_ = {'zeta0': [1.0, 0.05, 1.0, 0.1], 'zeta1': [50.0, 49.0, 1.0, 49.0], 'zeta2': [0.1, 2.5, 1.0, 2.5]}
      tuning_function = tuned_bandit.expit_epsilon_decay
  elif policy_name == 'eps-fixed-decay':
    tune = False
#    policy = mab_epsilon_greedy_policy
    if info:
      tuning_function = tuned_bandit.information_expit_epsilon_decay
#        tuning_function_parameter = np.array([ 0.86275253, -2.73708042,  3.02635897, -2.38429591 ,-3.99702655,  0.03036565,
#    2.57773559, -4.80310515, -0.88817786])
      tuning_function_parameter = np.array([1.4621,  -0.0236,   1.6827 ,   4.2271,
                                    0.4680, -3.6468,   3.5583 ,  -3.6252,   -1.1178 ])
    else:
      tuning_function = tuned_bandit.expit_epsilon_decay
#      tuning_function_parameter = np.array([2. , 44.60110916,  2.5] )
      tuning_function_parameter = np.array([0.90945399, 44.80884696,  2.22349283])
  
  list_of_reward_mus=[p1, p2]
  env = BernoulliMAB(list_of_reward_mus=list_of_reward_mus)
  p_opt = np.max(list_of_reward_mus)
  env.reset()
  for a in range(env.number_of_actions):
    env.step(a)
    
  cum_regret_each_rep = 0
  s1_list = []; s2_list=[]; epsilon_list = []
  for t in range(T):
    action = bern_mab_epsilon_greedy_policy(env.estimated_means, env.number_of_pulls, tuning_function,
                            tuning_function_parameter, T, t)

#    action,s1,s2,epsilon = bern_mab_epsilon_greedy_policy(env.estimated_means, env.number_of_pulls, tuning_function,
#                                tuning_function_parameter, T, t)
    res = env.step(action)
    regret = p_opt - list_of_reward_mus[action]
    cum_regret_each_rep += regret
#    s1_list.append(s1); s2_list.append(s2); epsilon_list.append(epsilon)
#    print(epsilon)
#    plt.plot(np.array(s1_list), epsilon_list)
#    plt.plot(np.array(s2_list), epsilon_list)
  return {'cumulative_regret': cum_regret_each_rep}


def run(policy_name, save=False, info=False, T=50):
  """

  :return:
  """

  replicates = 96
  num_cpus = int(mp.cpu_count())
#  num_cpus = 48
  #replicates = 20
  results = []
  pool = mp.Pool(processes=num_cpus)

  episode_partial = partial(episode, policy_name, info=info)

  results = pool.map(episode_partial, range(replicates))
  #results = episode_partial(1)
  cumulative_regret = [np.float(d['cumulative_regret']) for d in results]

  print(policy_name, info, 'regrets', float(np.mean(cumulative_regret)), 'se_regrets',float(np.std(cumulative_regret))/np.sqrt(replicates))
  # Save results
#  if save:
#    results = {'T': float(T), 'mean_regret': float(np.mean(cumulative_regret)), 'std_regret': float(np.std(cumulative_regret)),
#               'regret list': [float(r) for r in cumulative_regret],
#               'zeta_sequences': zeta_sequences, 'estimated_means': estimated_means, 'estimated_vars': estimated_vars,
#               'rewards': rewards, 'actions': actions}
#
#    base_name = 'info-mab-{}'.format(policy_name)
#    prefix = os.path.join(project_dir, 'src', 'environments', base_name)
#    suffix = datetime.datetime.now().strftime("%y%m%d_%H%M%S")
#    filename = '{}_{}.yml'.format(prefix, suffix)
#    with open(filename, 'w') as outfile:
#      yaml.dump(results, outfile)

  return


if __name__ == '__main__':
  start_time = time.time()
#  bayesopt_max_min(0)ma
  run("eps-fixed-decay")
  run("eps")
  run("greedy")
#  print(episode("eps-fixed-decay",0, info=False))
#  print(episode("eps",0, info=False))
#  print(episode("greedy",0, info=False))
  elapsed_time = time.time() - start_time
  print("time {}".format(elapsed_time))
  
#T=50; n1=10; n2=10; 
#s1_list = np.arange(n1); s2=5
#s2_list = np.arange(n2); s1=5
##zeta = [0.42538565, -31.4546137 ,   0.76901383  ,-0.19323311,  -0.32425063,
##   0.34286123,   0.10704383] ### 
##zeta = [2.00000000e+00 ,-3.64544979e+01,  1.00000000e-02 ,-1.00000000e+00,
##  1.00000000e+00, -1.00000000e+00,  1.00000000e+00]; ### p2<=p1
##                   
#eps_list=[]
#for s2 in s2_list:
#  eps_list.append(bern_expit_epsilon_decay(T, s1, s2, n1, n2, zeta))
#  temp = np.dot([1, T-n1-n2, s1, s2, n1, n2], zeta[1:])
#  print(temp, np.exp(-temp), expit(temp))
#  
#plt.plot(eps_list)  
#
#
#zeta = [2. , 44.60110916,  2.5] ### MinMax
#plt.plot(expit_epsilon_decay(T, np.arange(T), zeta))
#
#
