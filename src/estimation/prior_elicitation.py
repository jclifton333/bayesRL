import numpy as np


def compute_kl_divergence_for_combined_ppd(alpha, p1, p2, p):
  """
  For choosing alpha to approximate elicited prior predictive density p as alpha*p1 + (1-alpha)*p2.

  :param alpha:
  :param p1:
  :param p2:
  :param p:
  :return:
  """
  pass


def optimize_kl_divergence_for_combined_ppd(p1, p2, p):
  pass


def combine_ppds(pm_parametric, pm_np, elicited_prior_predictive, samples=500):
  """

  :param pm_parametric: pymc3 model for parametric model
  :param pm_np: pymc3 model for np model
  :param elicited_prior_predictive: function that allows sampling from elicited prior predictive model
  :param samples: number of samples to draw from each prior predictive dbn
  :return: optimal alpha for combining parametric and nonparametric prior predictive distributions to fit
            elicited_prior_predictive
  """
  p = elicited_prior_predictive(samples)
  p1 = pm_parametric.sample_prior_predictive(samples=samples)
  p2 = pm_np.sample_prior_predictive(samples=samples)
  alpha = optimize_kl_divergence_for_combined_ppd(p1, p2, p)
  return alpha


# Elicited prior predictive distributions for x1, x2 (conditional dbn), and y (conditional dbn) in the two-stage
# environment

def elicited_prior_predictive_for_x1():
  pass


def elicited_prior_predictive_for_x2():
  pass


def elicited_prior_predictive_for_y():
  pass








