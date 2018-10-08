import pdb
import numpy as np
import copy


class Glucose(object):
  NUM_STATE = 8
  MAX_STATE = 1000 * np.ones(NUM_STATE)
  MIN_STATE = np.zeros(NUM_STATE)
  NUM_ACTION = 2

  # Generative model parameters
  MU_GLUCOSE = 250
  SIGMA_GLUCOSE = 25
  INS_PROB = 0.3
  MU_FOOD = 0
  SIGMA_FOOD = 10
  MU_ACTIVITY = 0
  SIGMA_ACTIVITY = 10
  MU_ACTIVITY_MOD = 31
  SIGMA_ACTIVITY_MOD = 5
  SIGMA_NOISE = 5
  # Coefficients correspond to
  # intercept, current glucose food activity, previous glucose food activity, current action, previous action
  COEF = np.array([10, 0.9, 0.1, -0.01, 0.0, 0.1, -0.01, -10, -4])

  # Test states
  HYPOGLYCEMIC = np.array([50, 0, 33, 50, 0, 0, 0, 0])
  HYPERGLYCEMIC = np.array([200, 0, 30, 200, 0, 0, 78, 0])

  def __init__(self, horizon):
    self.R = []  # List of rewards at each time step
    self.A = []  # List of actions at each time step
    self.X = []  # List of features (previous and current states) at each time step
    self.S = []
    self.t = -1
    self.horizon = horizon
    self.current_state = self.last_state = self.last_action = None

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
    r1 = (new_glucose < 70) * (-0.005 * new_glucose**2 + 0.95 * new_glucose - 45) + \
          (new_glucose >= 70) * (-0.00017 * new_glucose**2 + 0.02167 * new_glucose - 0.5)

    # Reward from previous timestep
    r2 = (last_glucose < 70)*(-0.005*last_glucose**2 + 0.95*last_glucose - 45) + \
         (last_glucose >= 70)*(-0.00017*last_glucose**2 + 0.02167*last_glucose - 0.5)
    return r1 + r2

  def generate_food_and_activity(self):
    """

    :return:
    """
    food = np.random.normal(Glucose.MU_FOOD, Glucose.SIGMA_FOOD)
    food = np.multiply(np.random.random() < 0.2, food)
    activity = np.random.normal(Glucose.MU_ACTIVITY, self.SIGMA_ACTIVITY)
    activity = np.multiply(np.random.random() < 0.2, activity)
    return food, activity

  def reset(self):
    """

    :return:
    """
    # Reset obs history
    self.R = []  # List of rewards at each time step
    self.A = []  # List of actions at each time step
    self.X = []  # List of features (previous and current states) at each time step
    self.S = []

    # Generate first states
    food, activity = self.generate_food_and_activity()
    glucose = np.random.normal(Glucose.MU_GLUCOSE, Glucose.SIGMA_GLUCOSE)
    x = np.array([1, glucose, food, activity, glucose, food, activity, 0, 0])
    self.current_state = self.last_state = np.array([glucose, food, activity])
    self.last_action = 0
    self.t = -1
    self.X.append(x)
    self.S.append(self.current_state)
    self.step(0)
    return

  def next_state_and_reward(self, action):
    """

    :param action:
    :return:
    """

    # Transition to next state
    x = np.concatenate(([1], self.current_state, self.last_state, [self.last_action], [action]))
    glucose = np.dot(x, Glucose.COEF) + np.random.normal(0, Glucose.SIGMA_NOISE)
    food, activity = self.generate_food_and_activity()

    # Update current and last state and action info
    self.last_state = copy.copy(self.current_state)
    self.current_state = np.array([glucose, food, activity])
    self.last_action = action

    reward = self.reward_function(self.last_state, self.current_state)
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
    new_x[-2] = action
    return new_x

  def get_state_history_as_array(self):
    """
    Return past states as an array with blocks [ lag 1 states, states]
    :return:
    """
    X_as_array = np.vstack(self.X)
    return X_as_array

  def get_state_transitions_as_x_y_pair(self):
    """
    For estimating transition density.
    :return:
    """
    X = np.vstack(self.X[:-1])
    Sp1 = np.vstack(self.S[1:])
    return X, Sp1

  def step(self, action):
    self.t += 1
    done = self.t == self.horizon
    x, reward = self.next_state_and_reward(action)
    self.X.append(x)
    self.R.append(reward)
    self.A.append(action)
    self.S.append(self.current_state)
    return x, reward, done


