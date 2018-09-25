import sys
"""
Functions for sensitive analysis the optimal value of \zeta is to the parameters of the model
"""
import pdb
import os
this_dir = os.path.dirname(os.path.abspath(__file__))
project_dir = os.path.join(this_dir, '..', '..')
sys.path.append(project_dir)

import pdb
import numpy as np
import copy
from scipy.linalg import block_diag
from src.environments.Bandit import NormalCB
import src.policies.tuned_bandit_policies as tuned_bandit
import pickle
import multiprocessing as mp


def episode(env_name, grid_dimension = 50, mc_rep = 500, zeta = np.array([0.3, -5, 1]), T = 10):

  if env_name == "BernoulliMAB":
    env = BernoulliMAB()
    p0_low = 0
    p0_up = 1
    p1_up = 1
  elif env_name == "NormalMAB":
    env = NormalMAB()
    p0_low = -2
    p0_up = 2
    p1_up = 2
    
  for p0 in np.linspace(p0_low, p0_up, grid_dimension):
    for p1 in np.linspace(p0, p1_up, grid_dimension):
      for zeta2 in np.linspace(0, 2, grid_dimension):
        mean_cum_reward = 0
        for rep in range(mc_rep):
          env.reset()
          for t in range(T):
            zeta[2] = zeta2
            epsilon = tuned_bandit.expit_epsilon_decay(T, t, zeta)
            if np.random.rand() < epsilon:
              action = np.random.choice(2)
            else:
              action = np.argmax([p0, p1])
            env.step(action)
          cum_reward = sum(env.U)
          mean_cum_reward += (cum_reward - mean_cum_reward)/(rep+1)
          a = {'p0':p0, 'p1':p1, 'zeta':zeta, 'mean_cum_reward':mean_cum_reward}
          with open('{}_analysi_p0{}_p1{}_zeta2{}.pickle'.format(env_name, p0, p1, zeta2), 'wb') as handle:
            pickle.dump(a, handle, protocol=pickle.HIGHEST_PROTOCOL)
  return a

#with open('analysis.pickle', 'rb') as handle:
#    b = pickle.load(handle)
