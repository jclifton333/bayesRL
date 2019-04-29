"""
Choose models from a posterior which have a combination of high posterior density and conflicting optimal policies.

Pseudocode:
  m1 <- MAP model
  for k=2,3,...
    mk <- solve max_m ( \sum_{i < k} cross_regret(m, mi) + penalty(d(m, mi)) )

where cross_regret(m_a, m_b) = V( pi_opt(m_b) ; m_a) + V( pi_opt(m_a) ; m_b)
"""
import sys
import pdb
import os
this_dir = os.path.dirname(os.path.abspath(__file__))
project_dir = os.path.join(this_dir, '..', '..')
sys.path.append(project_dir)
import numpy as np
import src.estimation.density_estimation as dd
import src.estimation.TransitionModel as tm
from src.policies.simulation_optimization_policies import solve_for_pi_opt, glucose_feature_function
from src.run.evaluate_mb_policy import evaluate_policy
from sklearn.ensemble import RandomForestRegressor
from src.environments.Glucose import Glucose
from theano import shared


def posterior_mean(trace):
  """
  Helper.

  :param trace:
  :return:
  """
  mean_parameter = { varname: trace[varname].mean(axis=0) for varname in trace[0].keys() }
  return mean_parameter


def devils_advocate(model, trace, posterior_density, time_horizon, initial_state, initial_x, exploration_parameters,
                    number_of_actions, rollout_policy, feature_function, transition_model_from_parameter):
  """
  Conflict pursuit for TWO MODELS only.

  :param model: pymc3 model
  :param trace:
  :param posterior_density:
  :param time_horizon:
  :param exploration_parameters:
  :param transition_model_from_parameter: function that takes parameter and returns transition distribution
  :param initial_state:
  :return:
  """
  NUMBER_OF_EVALUATIONS = 10

  # Get map estimate and corresponding policy
  # map_parameters = model.find_MAP()  # ToDo: Posterior mean instead?
  map_parameters = posterior_mean(trace)
  transition_model_1 = transition_model_from_parameter(map_parameters)
  pi_1 = solve_for_pi_opt(initial_state, initial_x, transition_model_1, time_horizon, number_of_actions, rollout_policy,
                          feature_function)
  value_of_pi_1 = evaluate_policy(initial_state, initial_x, transition_model_1, time_horizon, pi_1, feature_function)

  # Define CP objective
  def cp_objective(transition_model_parameters):
    # Get value of policy corresponding to this parameter setting
    transition_model_2 = transition_model_from_parameter(transition_model_parameters)
    candidate_policy = solve_for_pi_opt(initial_state, initial_x, transition_model_2, time_horizon, number_of_actions,
                                        rollout_policy, feature_function)
    value_of_candidate_policy = evaluate_policy(initial_state, initial_x, transition_model_2, time_horizon, candidate_policy,
                                                feature_function)

    # Compute cross-regrets
    cross_value_candidate_policy = evaluate_policy(initial_state, initial_x, transition_model_1, time_horizon,
                                                   candidate_policy, feature_function)
    cross_value_pi_1 = evaluate_policy(initial_state, initial_x, transition_model_2, time_horizon, pi_1,
                                       feature_function)
    transition_model_parameters_density = posterior_density(transition_model_parameters)

    cross_regret_1 = value_of_pi_1 - cross_value_candidate_policy
    cross_regret_2 = value_of_candidate_policy - cross_value_pi_1
    return (cross_regret_1 + cross_regret_2) * transition_model_parameters_density

  # Optimize; currently just sample from trace rather than getting fancy
  best_value = -float("inf")
  best_param = None

  counter = 0
  for param in exploration_parameters:
    print(counter)
    param_value = cp_objective(param)
    if param_value > best_value:
      best_param = param
      best_value = param_value
    counter += 1
  for _ in range(NUMBER_OF_EVALUATIONS):
    print(counter)
    param = np.random.choice(trace)
    param_value = cp_objective(param)
    if param_value > best_value:
      best_param = param
      best_value = param_value
    counter += 1
  return best_param


if __name__ == "__main__":
  # def glucose_feature_function(g, a):
  #   # Draw next state from ppd
  #   glucose_patient = estimator.draw_from_ppd(current_x[patient])
  #   food_patient, activity_patient = env.generate_food_and_activity()  # ToDo: This should be estimated, not given!
  #   # food = np.append(food, food_patient)
  #   # activity = np.append(activity, activity_patient)
  #   x_patient = np.array([[1.0, glucose_patient, food_patient, activity_patient, X_rep[patient][-1, 1],
  #                          X_rep[patient][-1, 2], X_rep[patient][-1, 3], X_rep[patient][-1, -1], action[patient]]])

  # Collect data
  np.random.seed(3)
  n_patients = 10
  T = 5
  env = Glucose(nPatients=n_patients)
  env.reset()

  for t in range(T):
    # Get posterior
    # X, Sp1 = env.get_state_transitions_as_x_y_pair()
    # X = shared(X)
    # y = Sp1[:, 0]
    # model_, trace_ = dd.dependent_density_regression(X, y)
    action = np.random.binomial(1, 0.3, n_patients)
    env.step(action)

  # Get posterior
  X, Sp1 = env.get_state_transitions_as_x_y_pair()
  y = Sp1[:, 0]
  estimator = tm.GlucoseTransitionModel()
  estimator.fit(X, y)
  model_, trace_ = estimator.model, estimator.trace

  # Dissent pursuit
  time_horizon_ = 3

  def rollout_policy_(s):
    return np.random.binomial(1, 0.3)

  def posterior_density_(p):
    return np.exp(model_.logp(p))

  def transition_model(glucose_parameter):
    return tm.transition_model_from_np_parameter(glucose_parameter, estimator.draw_from_food_and_activity_ppd)

  feature_function = glucose_feature_function
  devils_advocate(model_, trace_, posterior_density_, time_horizon_, env.S[-1][-1, :], env.X[-1][-1, :], [],
                  2, rollout_policy_, feature_function, transition_model)


