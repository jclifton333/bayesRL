import numpy as np
from abc import ABCMeta, abstractmethod

ABC = ABCMeta('ABC', (object, ), {'__slots__': ()})


def generate_two_stage_data(policy_1, policy_2, x1_dbn, x2_dbn, y_dbn, n_patient):
  """

  :param policy_1: returns actions A1 given initial covariates X1
  :param policy_2: returns actions given history (X1, A1, X2)
  :param x1_dbn: Function that draws from distribution of initial covariates
  :param x2_dbn: Function that draws from conditional distribution of covariates given X1 and action
  :param y_dbn:  Function that draws from conditional distribution of reward given history (X1, A1, X2, A2)
  :param n_patient:
  :return:
  """

  x1 = x1_dbn(n_patient)
  actions_1 = policy_1(x1)
  x2 = x2_dbn(x1, actions_1)
  actions_2 = policy_2(x1, actions_1, x2)
  y = y_dbn(x1, actions_1, x2, actions_2)
  return {'y': y, 'x1': x1, 'actions_1': actions_1, 'x2': x2, 'actions_2': actions_2}


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
  def __init__(self, n_patients, policy1, policy2, x1_mean=np.zeros(3), x1_cov=np.eye(3)):
    TwoStage.__init__(n_patients, policy1, policy2)
    self.x1_mean = x1_mean
    self.x1_cov = x1_cov

  def x1_dbn(self):
    x1 = np.random.multivariate_normal(mean=self.x1_mean, cov=self.x1_cov, size=self.n_patients)
    return x1

  def x2_dbn(self, x1, action):
    pass

  def y_dbn(self, x1, action1, x2, action2):
    pass

# Complicated generative model



