import numpy as np


def normal_mab_gittins_index_policy(estimated_means, standard_errors, number_of_pulls, tuning_function,
                                    tuning_function_parameter, T, t):
  """
  Approximate gittins index for normal MAB with flat priors.

  :param estimated_means:
  :param standard_errors:
  :param number_of_pulls:
  :param tuning_function:
  :param tuning_function_parameter:
  :param T:
  :param t:
  :return:
  """
  posterior_mean = estimated_means
  posterior_var = 1.0/number_of_pulls #this requires that the reward has know variance which is 1
  m = T-t #t: start from 0
  gittins_index = posterior_mean + np.sqrt(2.0*posterior_var*np.log(m*posterior_var/np.sqrt(np.log(m*posterior_var))))
  action = np.argmax(gittins_index)
  return action


