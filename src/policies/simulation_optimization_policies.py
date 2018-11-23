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