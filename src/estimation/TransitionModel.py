"""
Classes for estimated transition model, to be used for tuning exploration.
"""
import sys
import pdb
import os
this_dir = os.path.dirname(os.path.abspath(__file__))
project_dir = os.path.join(this_dir, '..', '..')
sys.path.append(project_dir)
import numpy as np
import pymc3 as pm
import src.estimation.dependent_density as dd
from theano import shared


class GlucoseTransitionModel(object):
  def __init__(self, method='np'):
    """

    :param method: string in ['np', 'p', 'averaged']
    """
    assert method in ['np', 'p', 'averaged']

    self.model = None
    self.trace = None
    self.compare = None
    self.shared_x_np = None
    self.shared_x_p = None
    self.method = method

  def fit(self, X, y):
    # Update shared features
    if self.method == 'np':
      self.shared_x_np = shared(X)
      model_, trace_ = dd.dirichlet_mixture_regression(self.shared_x_np, y)
    elif self.method == 'p':
      self.shared_x_p = shared(X[:, :3])
      model_, trace_ = dd.normal_bayesian_regression(self.shared_x_p, y)
    elif self.method == 'averaged':
      self.shared_x_np = shared(X)
      self.shared_x_p = shared(X[:, :3])
      model_np, trace_np = dd.dirichlet_mixture_regression(self.shared_x_np, y)
      model_p, trace_p = dd.normal_bayesian_regression(self.shared_x_p, y)
      model_ = [model_p, model_np]
      trace_ = [trace_p, trace_np]
      self.compare = pm.compare({model_p: trace_p, model_np: trace_np}, method='BB-pseudo-BMA')

    self.model = model_
    self.trace = trace_
    # self.compare = compare_

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
      if ix_ == 1:
        pp_sample = pm.sample_ppc(self.trace[ix_], model=self.model[ix_])['obs'][0]
      elif ix_ == 0:
        # ToDo: Still don't understand why sample_ppc doesn't return correct shape here
        pp_sample = pm.sample_ppc(self.trace[ix_], model=self.model[ix_])['obs'][0, 0]
    elif self.method == 'np':
      self.shared_x_np.set_value(x)
      pp_sample = pm.sample_ppc(self.trace, model=self.model)['obs'][0]
    elif self.method == 'p':
      self.shared_x_p.set_value(x[:3])
      pp_sample = pm.sample_ppc(self.trace, model=self.model)['obs'][0, 0]

    return pp_sample


