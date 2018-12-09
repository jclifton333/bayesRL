import sys
import pdb
import os
# this_dir = os.path.dirname(os.path.abspath(__file__))
# project_dir = os.path.join(this_dir, '..', '..')
# sys.path.append(project_dir)

import copy
import numpy as np


# import src.policies.linear_algebra as la


class Glucose(object):
  NUM_STATE = 8
  MAX_STATE = 1000 * np.ones(NUM_STATE)
  MIN_STATE = np.zeros(NUM_STATE)
  NUM_ACTION = 2

  # Generative model parameters
  SIGMA_GLUCOSE = 25
  MU_GLUCOSE = 250
  INS_PROB = 0.3

  MU_ACTIVITY_MOD = 31
  SIGMA_ACTIVITY_MOD = 5
  COEF = np.array([10, 0.9, 0.1, -0.01, 0.0, 0.1, -0.01, -10, -4])
  SIGMA_NOISE = 5
  MU_FOOD = 0
  SIGMA_FOOD = 10
  MU_ACTIVITY = 0
  SIGMA_ACTIVITY = 10
  PROB_ACTIVITY = PROB_FOOD = 0.6

  # Coefficients correspond to
  # intercept, current glucose food activity, previous glucose food activity, current action, previous action

  # Test states
  HYPOGLYCEMIC = np.array([50, 0, 33, 50, 0, 0, 0, 0])
  HYPERGLYCEMIC = np.array([200, 0, 30, 200, 0, 0, 78, 0])

  def __init__(self, nPatients=1, x_initials=None, sx_initials=None):
    self.R = [[]] * nPatients  # List of rewards at each time step
    self.A = [[]] * nPatients  # List of actions at each time step
    self.X = [[]] * nPatients  # List of features (previous and current states) at each time step
    self.S = [[]] * nPatients
    self.Xprime_X_inv = None
    self.t = -1
    #    self.horizon = horizon
    self.current_state = [None] * nPatients
    self.last_state = [None] * nPatients
    self.last_action = [None] * nPatients
    self.nPatients = nPatients

    self.x_initials = x_initials
    self.sx_initials = sx_initials

  @staticmethod
  def reward_function(s_prev, s):
    """

    :param s_prev: state vector at previous time step
    :param s: state vector
    :return:
    """
    new_glucose = s[0]
    last_glucose = s_prev[0]

    # Reward from this timestep
    r1 = (new_glucose < 70) * (-0.005 * new_glucose ** 2 + 0.95 * new_glucose - 45) + \
         (new_glucose >= 70) * (-0.00017 * new_glucose ** 2 + 0.02167 * new_glucose - 0.5)

    # Reward from previous timestep
    r2 = (last_glucose < 70) * (-0.005 * last_glucose ** 2 + 0.95 * last_glucose - 45) + \
         (last_glucose >= 70) * (-0.00017 * last_glucose ** 2 + 0.02167 * last_glucose - 0.5)
    return r1  # + r2

  @staticmethod
  def reward_funciton_mHealth(glucose_news):
    """

    :param glucose_news: an array of new glucose values for all patients (dim: times by nPatients)
    :return:
    """
    r = np.zeros(glucose_news.shape)
    ind = (glucose_news < 70)
    r[ind] = -0.005 * glucose_news[ind] ** 2 + 0.95 * glucose_news[ind] - 45
    r[~ind] = -0.00017 * glucose_news[~ind] ** 2 + 0.02167 * glucose_news[~ind] - 0.5

    return np.mean(r, axis=0)

  @classmethod
  def generate_food_and_activity(cls):
    """

    :return:
    """
    food = np.random.normal(Glucose.MU_FOOD, Glucose.SIGMA_FOOD)
    food = np.multiply(np.random.random() < Glucose.PROB_FOOD, food)
    activity = np.random.normal(Glucose.MU_ACTIVITY, Glucose.SIGMA_ACTIVITY)
    activity = np.multiply(np.random.random() < Glucose.PROB_ACTIVITY, activity)
    return food, activity

  def reset(self):
    """

    :return:
    """
    # Reset obs history
    self.R = [[]] * self.nPatients  # List of rewards at each time step
    self.A = [[]] * self.nPatients  # List of actions at each time step
    self.X = [[]] * self.nPatients  # List of features (previous and current states) at each time step
    self.SX = [
                []] * self.nPatients  # List of states corresponding to the states in MDP, exclude current action in self.X, and 1-lag advance than self.X
    self.S = [[]] * self.nPatients

    # Generate first states for nPatients
    if self.x_initials is None:
      for i in range(self.nPatients):
        food, activity = self.generate_food_and_activity()
        glucose = np.random.normal(Glucose.MU_GLUCOSE, Glucose.SIGMA_GLUCOSE)
        action = np.random.choice(2)
        x = np.array([1, glucose, food, activity, glucose, food, activity, 0, action])
        self.current_state[i] = np.array([glucose, food, activity])
        self.last_state[i] = np.array([glucose, food, activity])
        self.last_action[i] = action
        self.X[i] = np.append(self.X[i], [x])
        self.S[i] = np.append(self.S[i], [np.array([glucose, food, activity])])
        self.SX[i] = np.append(self.SX[i], x[:-1])
      #      print(i, x)
      self.step(np.random.choice(2, size=self.nPatients))
    else:
      ### need to check again
      self.X = [self.x_initials[i,] for i in range(self.nPatients)]
      self.SX = [self.sx_initials[i,] for i in range(self.nPatients)]
      self.S = [self.sx_initials[i, 1:4] for i in range(self.nPatients)]
      for i in range(self.nPatients):
        self.X[i] = np.vstack((self.X[i], self.X[i]))
        self.S[i] = np.vstack((self.S[i], self.S[i]))
        self.SX[i] = np.vstack((self.SX[i], self.SX[i]))

      self.last_state = [self.sx_initials[i, 4:7] for i in range(self.nPatients)]
      self.current_state = [self.sx_initials[i, 1:4] for i in range(self.nPatients)]
      self.last_action = [self.sx_initials[i, 7] for i in range(self.nPatients)]
      self.A = [np.array(self.sx_initials[i, 7]) for i in range(self.nPatients)]
      self.R = [np.array(self.reward_function(self.sx_initials[i, 4:7], self.sx_initials[i, 1:4])) \
                for i in range(self.nPatients)]
    return

  def next_state_and_reward(self, action, i):
    """

    :param action:
    :return:
    """

    # Transition to next state
    #    print(self.current_state[i], self.last_action[i])
    sx = np.concatenate(([1], self.current_state[i], self.last_state[i],
                         [self.last_action[i]]))
    x = np.concatenate((sx, [action]))
    glucose = np.random.normal(np.dot(x, self.COEF), self.SIGMA_NOISE)
    food, activity = self.generate_food_and_activity()

    # Update current and last state and action info
    self.last_state[i] = copy.copy(self.current_state[i])
    self.current_state[i] = np.array([glucose, food, activity]).reshape(1, 3)[0]
    self.last_action[i] = action
    reward = self.reward_function(self.last_state[i], self.current_state[i])
    # current_x = np.concatenate()
    return x, reward

  @staticmethod
  def get_state_at_action(action, x):
    """
    Replace current action entry in x with action.
    :param action:
    :param x:
    :return:
    """
    new_x = copy.copy(x)
    new_x[-1] = action
    return new_x

  def get_state_history_as_array(self):
    """
    Return past states as an array with blocks [ lag 1 states, states]
    :return:
    """
    X_as_array = np.vstack(self.X)
    SX_as_array = np.vstack(self.SX)
    S_as_array = np.vstack(self.S)
    return X_as_array, SX_as_array, S_as_array

  def get_state_transitions_as_x_y_pair(self):
    """
    For estimating transition density.
    :return:
    """
    #    X = np.vstack(self.X[1:])
    #    Sp1 = np.vstack(self.S[1:])
    X = np.vstack([self.X[j][1:] for j in range(self.nPatients)])
    Sp1 = np.vstack([self.S[j][1:] for j in range(self.nPatients)])
    return X, Sp1

  def step(self, actions):
    '''
    actions: an array of actions for each patient
    '''
    self.t += 1
    #    done = self.t == self.horizon
    x_list = []
    mean_rewards_nPatients = 0
    for i in range(self.nPatients):
      #      pdb.set_trace()
      x, reward = self.next_state_and_reward(actions[i], i)
      x_list.append(x)
      #      print(i, reward)
      self.X[i] = np.vstack((self.X[i], x))
      self.R[i] = np.append(self.R[i], reward)
      self.A[i] = np.append(self.A[i], actions[i])
      self.S[i] = np.vstack((self.S[i], self.current_state[i]))
      self.SX[i] = np.vstack((self.SX[i],
                              np.concatenate(([1], self.current_state[i], self.last_state[i], [actions[i]]))))
      mean_rewards_nPatients += (reward - mean_rewards_nPatients) / (i + 1)
    return np.vstack(x_list), mean_rewards_nPatients  # , done

  def get_current_SX(self):
    ## current state feature for all patients
    current_sx = np.hstack((np.hstack((np.ones((self.nPatients, 1)), np.hstack((self.current_state, self.last_state)))),
                            np.array(self.last_action).reshape(self.nPatients, 1)))
    return current_sx

#  def get_Xprime_X_inv(self):
#    X, _ = self.get_state_transitions_as_x_y_pair()
#    x_new = self.get_state_history_as_array()[-1,]
#    if self.Xprime_X_inv is None:  # Can't do fast update
#      Xprime_X_new = np.dot(X.T, X)
#      self.Xprime_X_inv = np.linalg.inv(Xprime_X_new + 0.01*np.eye(X.shape[1]))
#    else:
#      # Compute Xprime_X_inv
#      self.Xprime_X_inv= la.sherman_woodbury(self.Xprime_X_inv, x_new, x_new)
#
