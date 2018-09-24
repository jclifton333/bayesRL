def mab_gittins_index_policy(estimated_means, standard_errors, number_of_pulls, T, t):
  posterior_mean = estimated_means
  posterior_var = 1.0/number_of_pulls #this requires that the reward has know variance which is 1
  m = T-t #t: start from 0
  gettins_index = posterior_mean + np.sqrt(2.0*posterior_var*np.log(m*posterior_var/np.sqrt(np.log(m*posterior_var))))
  action = np.argmax(gettins_index)
  return action


