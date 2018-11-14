"""
Classes for estimated transition model, to be used for tuning exploration.
"""
import numpy as np
from .dependent_density import dependent_density_regression, posterior_predictive_transition


class GlucoseTransitionModel(object):
  def __init__(self):
    self.models = None
    self.trace = None
    self.compare = None
    self.shared_x = None

  def fit(self, X, Sp1):
    model_, trace_, compare_ = dependent_density_regression(X, Sp1)

