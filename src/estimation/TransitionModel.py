"""
Classes for estimated transition model, to be used for tuning exploration.
"""
import numpy as np
import pymc3 as pm
from .dependent_density import dependent_density_regression, posterior_predictive_transition
from theano import shared


class GlucoseTransitionModel(object):
  def __init__(self, method='np'):
    """

    :param method: string in ['np', 'p', 'averaged']
    """
    self.models = None
    self.trace = None
    self.compare = None
    self.shared_x_np = None
    self.shared_x_p = None
    self.method = method

  def fit(self, X, y):
    # Update shared features
    if self.method == 'np':
      self.shared_x_np = shared(X)
    elif self.method == 'p':
      self.shared_x_p = shared(X[:, :3])
    elif self.method == 'averaged':
      self.shared_x_np = shared(X)
      self.shared_x_p = shared(X[:, :3])

    # Fit model
    model_, trace_, compare_ = dependent_density_regression(self.shared_x_np, y, X_p=self.shared_x_p)
    self.models = model_
    self.trace = trace_
    self.compare = compare_

  def draw_from_ppd(self, x):
    """

    :param x:
    :return:
    """

    # If performing model averaging, need to compute weights and draw from mixed ppd
    if self.method == 'averaged':
      weights_ = np.array(self.compare.weight.sort_index(ascending=True)).astype(float)
      ix_ = np.random.choice(len(weights_), p=weights_)
      self.shared_x_np.set_value(x)
      self.shared_x_p.set_value(x[:3])
      pp_sample = pm.sample_ppc(self.trace[ix_], model=self.model[ix_], samples=1, size=1)
    elif self.method == 'np':
      self.shared_x_np.set_value(x)
      pp_sample = pm.sample_ppc(self.trace[0], model=self.model[0], samples=1, size=1)
    elif self.method == 'p':
      self.shared_x_p.set_value(x[:3])
      pp_sample = pm.sample_ppc(self.trace[1], model=self.model[1], samples=1, size=1)

    return pp_sample


