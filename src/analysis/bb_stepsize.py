def sgd_bb_stepsize(zeta_k, zeta_k_minus_one):  
  # calculate gradient of V(zeta_k)
  z = np.random.normal(loc=0, size=J)
  v_of_zeta_k = rollout_stepwise_linear_mab(zeta_k, env, J=J, mc_rep=mc_rep, T=T)
  v_of_zeta_k_plus_z = rollout_stepwise_linear_mab(zeta_k + z, env, J=J, mc_rep=mc_rep, T=T)
  gradient_v_zeta_k = (v_of_zeta_k_plus_z - v_of_zeta_k)/z
  
  # calculate gradient of V(zeta_{k-1})
  z = np.random.normal(loc=0, size=J)
  v_of_zeta_k = rollout_stepwise_linear_mab(zeta_k_minus_one, env, J=J, mc_rep=mc_rep, T=T)
  v_of_zeta_k_plus_z = rollout_stepwise_linear_mab(zeta_k_minus_one + z, env, J=J, mc_rep=mc_rep, T=T)
  gradient_v_zeta_k_minus_one = (v_of_zeta_k_plus_z - v_of_zeta_k)/z
  
  # calculate stepsize lambda_k
  s_k = zeta_k - zeta_k_minus_one
  y_k = gradient_v_zeta_k - gradient_v_zeta_k_minus_one
  lambda_k = linalg.norm(s_k)**2/np.dot(s_k, y_k)   
  return lambda_k