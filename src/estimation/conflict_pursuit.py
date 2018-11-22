"""
Choose models from a posterior which have a combination of high posterior density and conflicting optimal policies.

Pseudocode:
  m1 <- MAP model
  for k=2,3,...
    mk <- solve max_m ( \sum_{i < k} cross_regret(m, mi) + penalty(d(m, mi)) )

where cross_regret(m_a, m_b) = V( pi_opt(m_b) ; m_a) + V( pi_opt(m_a) ; m_b)
"""
import numpy as np


def solve_for_pi_opt(transition_model, time_horizon):
  """

  :param transition_model:
  :param time_horizon:
  :return:
  """
  pass


def evaluate_policy(transition_model, time_horizon, policy):
  pass


def conflict_pursuit(model, trace, posterior_density, time_horizon, exploration_parameters):
  """
  Conflict pursuit for TWO MODELS only.

  :param model: pymc3 model
  :param trace:
  :param posterior_density:
  :param time_horizon:
  :param exploration_parameters:
  :return:
  """
  NUMBER_OF_EVALUATIONS = 10

  # Get map estimate and corresponding policy
  map_parameters = model.find_MAP()
  pi_1 = solve_for_pi_opt(map_parameters)

  # Define CP objective
  def cp_objective(transition_model_parameters):
    candidate_policy = solve_for_pi_opt(transition_model_parameters, time_horizon)
    # ToDo: Cross regret is wrong!
    cross_regret_1 = evaluate_policy(map_parameters, time_horizon, candidate_policy)
    cross_regret_2 = evaluate_policy(transition_model_parameters, time_horizon, pi_1)
    transition_model_parameters_density = posterior_density(transition_model_parameters_density)
    return (cross_regret_1 + cross_regret_2) * transition_model_parameters_density

  # Optimize; currently just sample from trace rather than getting fancy
  best_value = -float("inf")
  best_param = exploration_parameters[0]

  for param in exploration_parameters:
    param_value = cp_objective(param)
    if param_value > best_value:
      best_param = param
      best_value = param_value
  for eval in range(NUMBER_OF_EVALUATIONS):
    param = trace.sample()
    param_value = cp_objective(param)
    if param_value > best_value:
      best_param = param
      best_value = param_value

  return param




