"""
Choose models from a posterior which have a combination of high posterior density and conflicting optimal policies.

Pseudocode:
  m1 <- MAP model
  for k=2,3,...
    mk <- solve max_m ( \sum_{i < k} cross_regret(m, mi) + penalty(d(m, mi)) )

where cross_regret(m_a, m_b) = V( pi_opt(m_b) ; m_a) + V( pi_opt(m_a) ; m_b)
"""
import numpy as np
from sklearn.ensemble import RandomForestRegressor


def solve_for_pi_opt(initial_state, transition_model, time_horizon, number_of_actions, rollout_policy,
                     feature_function, mc_rollouts=100, number_of_dp_iterations=2):
  """
  Solve for optimal policy using dynamic programming.

  :param transition_model:
  :param time_horizon:
  :param rollout_policy: policy for generating data only
  :param number_of_dp_iterations:
  :return:
  """
  # Generate data for fqi
  x_dim = len(feature_function(initial_state, 0))
  s_dim = len(initial_state)
  X = np.zeros((0, x_dim))
  S = np.zeros((0, s_dim))
  R = np.zeros(0)
  for rollout in range(mc_rollouts):
    s = initial_state
    for t in range(time_horizon):
      a = rollout_policy(s)
      x = feature_function(s, a)
      X = np.vstack((X, x))
      S = np.vstack((S, s))
      s, r = transition_model(x)
      R = np.append(R, r)

  # Do FQI
  reg = RandomForestRegressor()
  reg.fit(X, R)
  q = lambda x_: reg.predict(x_)
  for _ in range(number_of_dp_iterations):
    Q_ = R + np.array([
      np.max([q(feature_function(s, a)) for a in range(number_of_actions)]) for s in S[1:]])
    reg.fit(X[:-1], Q_)
    q = lambda x_: reg.predict(x_)

  def pi_opt(s_):
    return np.argmax([q(feature_function(s_, a_) for a_ in range(number_of_actions))])

  return pi_opt


def evaluate_policy(initial_state, transition_model, time_horizon, policy, feature_function):
  """

  :param initial_state:
  :param transition_model:
  :param time_horizon:
  :param policy:
  :param feature_function:
  :return:
  """
  MC_REPLICATES = 100
  returns = []

  for _ in range(MC_REPLICATES):
    s = initial_state
    return_ = 0.0
    for t in range(time_horizon):
      a = policy(s)
      x = feature_function(s, a)
      s, r = transition_model(x)
      return_ += r
    returns.append(return_)
  return np.mean(returns)


def conflict_pursuit(model, trace, posterior_density, time_horizon, initial_state, exploration_parameters,
                     number_of_actions, rollout_policy, feature_function):
  """
  Conflict pursuit for TWO MODELS only.

  :param model: pymc3 model
  :param trace:
  :param posterior_density:
  :param time_horizon:
  :param exploration_parameters:
  :param initial_state:
  :return:
  """
  NUMBER_OF_EVALUATIONS = 10

  # Get map estimate and corresponding policy
  map_parameters = model.find_MAP()
  pi_1 = solve_for_pi_opt(initial_state, map_parameters, time_horizon, number_of_actions, rollout_policy,
                          feature_function)
  value_of_pi_1 = evaluate_policy(initial_state, map_parameters, time_horizon, pi_1, feature_function)

  # Define CP objective
  def cp_objective(transition_model_parameters):
    # Get value of policy corresponding to this parameter setting
    candidate_policy = solve_for_pi_opt(transition_model_parameters, time_horizon)
    value_of_candidate_policy = evaluate_policy(initial_state, map_parameters, time_horizon, candidate_policy,
                                                feature_function)

    # Compute cross-regrets
    cross_value_candidate_policy = evaluate_policy(initial_state, map_parameters, time_horizon, candidate_policy)
    cross_value_pi_1 = evaluate_policy(initial_state, transition_model_parameters, time_horizon, pi_1)
    transition_model_parameters_density = posterior_density(transition_model_parameters)

    cross_regret_1 = value_of_pi_1 - cross_value_candidate_policy
    cross_regret_2 = value_of_candidate_policy - cross_value_pi_1
    return (cross_regret_1 + cross_regret_2) * transition_model_parameters_density

  # Optimize; currently just sample from trace rather than getting fancy
  best_value = -float("inf")
  best_param = exploration_parameters[0]

  for param in exploration_parameters:
    param_value = cp_objective(param)
    if param_value > best_value:
      best_param = param
      best_value = param_value
  for _ in range(NUMBER_OF_EVALUATIONS):
    param = trace.sample()
    param_value = cp_objective(param)
    if param_value > best_value:
      best_param = param
      best_value = param_value

  return best_param




