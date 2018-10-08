import sys
import pdb
import os
this_dir = os.path.dirname(os.path.abspath(__file__))
project_dir = os.path.join(this_dir, '..', '..')
sys.path.append(project_dir)


import matplotlib.pyplot as plt
from src.environments.Bandit import NormalCB, NormalUniformCB
from src.policies import tuned_bandit_policies as tuned_bandit
from src.policies import global_optimization as opt
from src.policies import rollout
import copy
import numpy as np
import src.policies.linear_algebra as la
from scipy.linalg import block_diag
from functools import partial
import datetime
import yaml
import multiprocessing as mp
from src.run.normal_contextual_bandit import episode
from src.analysis.stepwise_bayesopt import  bayes_optimize_zeta


if __name__ == '__main__':
#  with open("/Users/lili/Documents/labproject2017/aistat/bayesRL/src/run/results/normalcb-eps-decay_181003_123046.yml", 'r') as f:
#    doc = yaml.load(f)
#  # episode('eps', 50)
#  cumulative_regret_different_t = []
#  cumulative_regret_se_different_t = []
#  for t in [10, 20, 30, 40, 49]:
#    cumulative_regret = 0
#    cumulative_regret_se = []
#    for i in range(96):
#      tuning_function_parameter = doc['zeta_sequences'][i][t]
##      print(i, t, tuning_function_parameter)
#      res = episode('eps-decay-fixed', 0, tuning_function_parameter=tuning_function_parameter, T=50)
##      print(i, t, res['cumulative_regret'])
#      cumulative_regret += (res['cumulative_regret'] - cumulative_regret)/(i+1)
#      cumulative_regret_se = np.append(cumulative_regret_se,  cumulative_regret)
#    cumulative_regret_different_t = np.append(cumulative_regret_different_t, cumulative_regret)
#    cumulative_regret_se_different_t = np.append(cumulative_regret_se_different_t, np.std(cumulative_regret_se)/np.sqrt(96))
#    print(t, cumulative_regret_different_t , cumulative_regret_se_different_t )
  best_para = dict()
  for N in [5, 10, 15, 20, 25]:
    env = NormalCB(num_initial_pulls=N, list_of_reward_betas=[[-10, 0.4, 0.4, -0.4], [-9.8, 0.6, 0.6, -0.4]], context_mean=np.array([0.0, 0.0, 0.0]),
            context_var=np.array([[1.0,0,0], [0,1.,0], [0, 0, 1.]]), list_of_reward_vars=[1, 1])
    sigma_sq_hat_list = [env.sigma_hat_list[a]**2 for a in range(2)]
    p = bayes_optimize_zeta(0, num_initial_pulls=N, list_of_reward_betas=env.beta_hat_list, context_mean=np.mean(env.X[:,-3:], axis=0), 
                        context_var=np.cov(env.X[:,-3:], rowvar=False), list_of_reward_vars=sigma_sq_hat_list, mc_rep=1000, T=50)
    print(p)
    best_para[str(N)] = p
    
  

