# import sys
import pdb
# import os
#
# this_dir = os.path.dirname(os.path.abspath(__file__))
# project_dir = os.path.join(this_dir, '..', '..')
# sys.path.append(project_dir)
#
# import matplotlib.pyplot as plt
# from src.environments.Bandit import NormalCB, NormalUniformCB
# from src.policies import tuned_bandit_policies as tuned_bandit
# from src.policies import rollout
# from src.environments.Glucose import Glucose
#
# import copy
import numpy as np
# import src.policies.linear_algebra as la
# from scipy.linalg import block_diag
# from functools import partial
# import datetime
# import yaml
# import multiprocessing as mp
#
# from sklearn.ensemble import RandomForestRegressor
# from sklearn.linear_model import LinearRegression
# from bayes_opt import BayesianOptimization
# from scipy.stats import wishart
# from scipy.stats import chi2
# import time


class Gridworld(object):
    '''
    A difficult gridworld task with 16 squares.

    Actions
    ------
    0 = N
    1 = E
    2 = S
    3 = W

    State numbering
    ---------------
    0  1  2  3
    4  5  6  7
    8  9  10 11
    12 13 14 15

    Transitions
    -----------
    15 is terminal.
    Transitions are deterministic in states [0, 1, 2, 3, 7, 11],
    uniformly randomly everywhere else (for every action).
    Rewards
    -------
    Reward is -1 for each transition except transitions to 15, which are positive.
    '''
    NUM_STATE = 16
    NUM_ACTION = 4
    PATH = [0, 1, 2, 3, 7, 11]  # States for which transitions are deterministic
    TERMINAL = [15]

    # These functions help construct the reward and transition matrices.
    @staticmethod
    def adjacent(s):
        '''
        Returns states adjacent to s in order [N, E, S, W]
        If s on boundary, s is adjacent to itself in that direction
          e.g. adjacent(0) = [0, 1, 4, 0]
        This results in double-counting boundary states when transition is uniformly random
        '''
        return [s - 4 * (s > 3), s + 1 * ((s + 1) % 4 != 0), s + 4 * (s < 12), s - 1 * (s % 4 != 0)]

    @classmethod
    def transition(cls, s, a):
        # Returns the normal deterministic transition from state s given a
        return cls.adjacent(s)[a]

    @staticmethod
    def reward(s):
        # Returns reward for transitioning to state s
        if s < 15:
            return -1
        else:
            return 1

    def __init__(self, time_horizon=1000, maxT=10, gamma=0.9, transitionMatrices=None):#, hardmax, epsilon=0.1, fixUpTo=None):
        '''
        Parameters
        ----------
        maxT: max number of steps in an episode
        time_horizon: max number of steps of all episodes
        gamma : discount factor
        epsilon : for epsilon - greedy
        transitionMatrices: an array of Gridworld.NUM_ACTION by Gridworld.NUM_STATE by Gridworld.NUM_STATE (dim=(4, 16, 16))
        '''
        self.current_state = None
        self.counter = 0
        self.maxT = maxT
        self.gamma = gamma
        self.transitionMatrices = transitionMatrices
        self.rewardMatrices = np.zeros((Gridworld.NUM_ACTION, Gridworld.NUM_STATE, Gridworld.NUM_STATE))
        self.posterior_alpha = np.ones((16, 4, 4)) #Initialize posterior, actually it's also the prior
        self.time_horizon = time_horizon
        
        # Construct transition and reward arrays
        if self.transitionMatrices is None:
            self.transitionMatrices = np.zeros((Gridworld.NUM_ACTION, Gridworld.NUM_STATE, Gridworld.NUM_STATE))
    
            for s in range(Gridworld.NUM_STATE):
                if s in Gridworld.PATH:
                    for a in range(Gridworld.NUM_ACTION):
                        s_next = self.transition(s, a)
                        self.transitionMatrices[a, s, s_next] = 1
                        self.rewardMatrices[a, s, s_next] = self.reward(s_next)
                elif s in Gridworld.TERMINAL:
                    for a in range(Gridworld.NUM_ACTION):
                        s_next = s
                        self.transitionMatrices[a, s, s_next] = 1
                        self.rewardMatrices[a, s, s_next] = self.reward(s_next)
                else:
                    for a in range(Gridworld.NUM_ACTION):
                        adjacent_states = self.adjacent(s)
                        uniform_transition_prob = 0.2
                        self.transitionMatrices[a, s, adjacent_states[a]] = 0.4
                        self.rewardMatrices[a, s, adjacent_states[a]] = self.reward(adjacent_states[a])
    #                    uniform_transition_prob = 1.0 / len(adjacent_states)
#                        pdb.set_trace()
                        adjacent_states.remove(adjacent_states[a])
                        for s_next in adjacent_states:
                            self.transitionMatrices[a, s, s_next] += uniform_transition_prob
                            self.rewardMatrices[a, s, s_next] = self.reward(s_next)
        else:
          # Construct rewardMatrices under an known transitionMatrices
            for s in range(Gridworld.NUM_STATE):
                if s in Gridworld.PATH:
                    for a in range(Gridworld.NUM_ACTION):
                        s_next = self.transition(s, a)
                        self.rewardMatrices[a, s, s_next] = self.reward(s_next)
                elif s in Gridworld.TERMINAL:
                    for a in range(Gridworld.NUM_ACTION):
                        s_next = s
                        self.rewardMatrices[a, s, s_next] = self.reward(s_next)
                else:
                    for a in range(Gridworld.NUM_ACTION):
                        adjacent_states = self.adjacent(s)
                        for s_next in adjacent_states:
                            self.rewardMatrices[a, s, s_next] = self.reward(s_next)
 
        #Create transition dictionary of form {s_0 : {a_0: [( P(s_0 -> s_0), s_0, reward), ( P(s_0 -> s_1), s_1, reward), ...], a_1:...}, s_1:{...}, ...}
        self.mdpDict= {}
        for s in range(self.NUM_STATE):
            self.mdpDict[s] = {} 
            for a in range(self.NUM_ACTION):
                self.mdpDict[s][a] = [(self.transitionMatrices[a, s, sp1], sp1, self.rewardMatrices[a, s, sp1]) for sp1 in range(self.NUM_STATE)]
  

    def reset(self):
        '''
        reset to the state 0
        :return:
        '''
        self.t = 0
        self.current_state = 0
        return self.current_state

    def step(self, a):
        self.counter += 1
        self.t += 1
#        pdb.set_trace()
        prob = self.transitionMatrices[a, self.current_state, :]
#        print(a, self.current_state)
        s = self.current_state
        self.current_state = np.random.choice(range(Gridworld.NUM_STATE), 1, p=prob)[0]
        reward = self.reward(self.current_state)
        if self.t == self.maxT or self.current_state == Gridworld.TERMINAL or self.counter == self.time_horizon:
            done = True
        else:
            done = False
        
        adjacent_index = self.adjacent(s).index(self.current_state) # the adjacent state of s, the index is belong to {0, 1, 2, 3}
        self.posterior_alpha[s, a, adjacent_index] += 1.0
#        transition_prob = self.posterior_alpha/np.sum(self.posterior_alpha, axis=2).reshape(16,4,1)
#        transitionMatrices = self.convert_to_transitionMatrices(transition_prob)
        return self.current_state, reward, done
      
    def convert_to_transitionMatrices(self, adj_transition):
        '''
        adj_transition: transition matrix (16, 4, 4), the last dimension only contains the four adjacent states
        :return: convert to the format of transitionMatrices (4, 16, 16)
        '''
        transitionMatrices = np.zeros((self.NUM_ACTION, self.NUM_STATE, self.NUM_STATE))
        for s in range(self.NUM_STATE):
            for a in range(self.NUM_ACTION):
                for s_adj in range(len(self.adjacent(s))):
                    transitionMatrices[a, s, self.adjacent(s)[s_adj]] += adj_transition[s, a, s_adj]
#        self.transitionMatrices = transitionMatrices
        return transitionMatrices
      
    def posterior_mean_model(self):
        # Use posterior mean as transition probability, to get the estimated model for doing policy iteration
        # return 
        transition_prob = self.posterior_alpha/np.sum(self.posterior_alpha, axis=2).reshape(16,4,1)
        transitionMatrices = self.convert_to_transitionMatrices(transition_prob)
        return transitionMatrices 




