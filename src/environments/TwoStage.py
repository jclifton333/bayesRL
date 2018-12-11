import numpy as np
from abc import ABCMeta, abstractmethod

ABC = ABCMeta('ABC', (object, ), {'__slots__': ()})


class TwoStage(ABC):
  def __init__(self, n_patients, policy1, policy2):
    self.n_patients = n_patients
    self.policy1 = policy1
    self.policy2 = policy2

  @abstractmethod
  def x1_dbn(self):
    pass

  @abstractmethod
  def x2_dbn(self, x1, action):
    pass

  @abstractmethod
  def y_dbn(self, x1, action1, x2, action2):
    pass

  def draw_x1(self):
    self.x1 = self.x1_dbn()

  def draw_x2(self):
    self.action1 = self.policy1(self.x1)
    self.x2 = self.x2_dbn(self.x1, self.action1)

  def draw_y(self):
    self.action2 = self.policy2(self.x1, self.action1, self.x2)
    self.y = self.y_dbn(self.x1, self.action1, self.x2)

  def run_two_stage(self):
    self.draw_x1()
    self.draw_x2()
    self.draw_y()


class SimpleTwoStage(TwoStage):
  def __init__(self, n_patients, policy1, policy2)
    """
    
    :param n_patients: 
    :param policy1: 
    :param policy2: 
    """
    TwoStage.__init__(n_patients, policy1, policy2)

    # Generative model parameters
    self.x1_mean = np.zeros(3)
    self.x1_cov = np.eye(3)
    self.B2_0 = np.array([[1, 2, 3],   # (x2 | x_1, a_1) mean = B2_0.x_1 + a_1 * B2_1.x_1
                         [-1, 2, 0]])
    self.B2_1 = np.array([[1, 0, 0],
                          [0, 1, 0]])
    self.x2_cov = np.eye(2)
    self.B_reward_0 = np.array([[1, 0],  # y mean is defined similarly to x2_mean
                                [1, 0]])
    self.B_reward_1 = np.array([[1, -1],
                                [-2, 3]])
    self.y_std = 1.0

  def x1_dbn(self):
    x1 = np.random.multivariate_normal(mean=self.x1_mean, cov=self.x1_cov, size=self.n_patients)
    return x1

  def x2_dbn(self, x1, action):
    x1_times_action = np.multiply(x1, action)
    x2_means = np.dot(self.B2_0, x1)
    x2_means += np.dot(self.B2_1, x1_times_action)
    x2 = np.array([
      np.random.multivariate_normal(x2_mean, self.x2_cov) for x2_mean in x2_means
    ])
    return x2

  def y_dbn(self, x1, action1, x2, action2):
    x2_times_action = np.multiply(x2, action2)
    y_means = np.dot(self.B_reward_0, x2)
    y_means += np.dot(self.B_reward_1, x2_times_action)
    y = np.array([
      np.random.normal(y_mean, self.y_std) for y_mean in y_means
    ])
    return y

# ToDo: Implement ComplicatedTwoStage



