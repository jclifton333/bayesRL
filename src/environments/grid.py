# import sys
# import pdb
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
# import numpy as np
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
    def reward( s):
        # Returns reward for transitioning to state s
        if s < 15:
            return -1
        else:
            return 1

    def __init__(self, maxT=50):#, hardmax, maxT, gamma=0.9, epsilon=0.1, fixUpTo=None):
        '''
        Parameters
        ----------
        maxT: max number of steps in episode
        gamma : discount factor
        epsilon : for epsilon - greedy
        '''
        self.current_state = None
        self.t = 0
        self.maxT = maxT

        # Construct transition and reward arrays
        self.transitionMatrices = np.zeros((Gridworld.NUM_ACTION, Gridworld.NUM_STATE, Gridworld.NUM_STATE))
        self.rewardMatrices = np.zeros((Gridworld.NUM_ACTION, Gridworld.NUM_STATE, Gridworld.NUM_STATE))

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
                    uniform_transition_prob = 1 / len(adjacent_states)
                    for s_next in adjacent_states:
                        self.transitionMatrices[a, s, s_next] += uniform_transition_prob
                        self.rewardMatrices[a, s, s_next] = self.reward(s_next)

        # # Initialize as FiniteMDP subclass
        # FiniteMDP.__init__(self, method, hardmax, maxT, gamma, epsilon, transitionMatrices, rewardMatrices,
        #                    Gridworld.TERMINAL)

    def reset(self):
        '''
        reset to the state 0
        :return:
        '''
        self.current_state = [0]
        return self.current_state

    def step(self, a):
        self.t += 1
        prob = self.transitionMatrices[a, self.current_state, :]
        self.current_state = np.random.choice(range(16), 1, p=prob)
        reward = self.reward(self.current_state)
        if self.t == self.maxT or self.current_state == Gridworld.TERMINAL:
            done = True
        else:
            done = False
        return self.current_state, reward, done


