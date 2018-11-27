import numpy as np


def transition_distribution_bellman_error(q, transition_distribution, S_ref, A_ref, feature_function, reward_function,
                                          gamma, number_of_actions, mc_samples_per_state_action_pair=100):
  """
  Compute (approximate) bellman error for q-function against transition distribution on reference distribution.

  :param q: q function
  :param transition_distribution: function that samples from conditional distribution at (s, a) pairs
  :param S_ref: Array of reference states
  :param A_ref: Array of reference actions
  :return:
  """
  be = 0.0
  for s, a in zip(S_ref, A_ref):
    q_sa = q(feature_function(s, a))
    for _ in range(mc_samples_per_state_action_pair):
      s_next = transition_distribution(s, a)
      r_next = reward_function(s_next)
      qmax = np.max([q(feature_function(s_next, a_next)) for a_next in range(number_of_actions)])
      be += np.power(r_next + gamma*qmax - q_sa, 2)
  return be


