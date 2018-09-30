import scipy.stats
import numpy as np
def normal_mab_ucb_policy(estimated_means, standard_errors, number_of_pulls, tuning_function,
                      tuning_function_parameter, T, t, env):
  ## alpha (percentile): decrease from a little bit smaller than 1 to 1/2 at T
  ## scale and shift
  alpha = tuning_function(T, t, tuning_function_parameter)/40 + 0.5
  z = scipy.stats.norm.ppf(alpha)
  action = np.argmax(estimated_means + z * standard_errors)
  return action
  
  