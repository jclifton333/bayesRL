import numpy as np
from scipy.special import expit


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

def elicited_prior_predictive_for_x1(samples):
  pass


def elicited_prior_predictive_for_x2(samples, basis_states):
  """
  Mixture of N normals.

  :param samples: Number of samples AT EACH BASIS STATE
  :param basis_states: States at which to sample from conditional dbn
  :return:
  """
  # ToDo: Random placeholder
  N_MIXTURE_COMPONENTS = 3
  x1_dim = basis_states.shape[1]
  logit_parameters = np.random.random(size=(N_MIXTURE_COMPONENTS, x1_dim))
  normal_mean_parameters = np.random.random(size=(N_MIXTURE_COMPONENTS, x1_dim))  # ToDo: Should be 3d
  sample_list = []

  for x1 in basis_states:
    sample_list_for_x1 = []
    mixture_component_weights = np.array([np.expit(np.dot(x1, logit_parameter)) for logit_parameter in
                                          logit_parameters])
    mixture_component_weights /= np.sum(mixture_component_weights)
    normal_means = np.array([np.dot(x1, normal_mean_parameter) for normal_mean_parameter in normal_mean_parameters])
    for sample in range(samples):
      mixture_component = np.random.choice(N_MIXTURE_COMPONENTS, p=mixture_component_weights)
      normal_mean = normal_means[mixture_component]
      x2_sample = np.random.multivariate_normal(normal_mean, cov=np.eye(normal_mean))
      sample_list_for_x1.append(x2_sample)
    sample_list.append(sample_list_for_x1)
  return sample_list


def elicited_prior_predictive_for_y(samples, basis_states):
  pass









