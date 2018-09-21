from src.environments.Bandit import NormalCB
from src.policies.tuned_bandit_policies import truncated_thompson_sampling
import numpy as np


def main():
  """
  For now this will just be for truncated thompson sampling policy.

  :return:
  """
  # Simulation settings
  replicates = 50
  T = 25
  env = NormalCB()

  # Run sims
  for replicate in range(replicates):

    # Initial pulls (so we can fit the models)
    for a in range(env.number_of_actions):
      for p in range(env.context_dimension):
        env.step(a)

    # Get initial linear model
    X, A, U = env.X, env.A, env.U
    linear_model_results = {'beta_hat_list': [], 'Xprime_X_inv_list': [], 'X_list': [], 'X_dot_y_list': [],
                            'sample_cov_list': []}
    for a in range(env.number_of_actions):  # Fit linear model on data from each actions
      # Get observations where action == a
      indices_for_a = np.where(A == a)
      X_a = X[indices_for_a, :]
      U_a = U[indices_for_a]

      # Fit linear model
      Xprime_X_inv_a = np.linalg.inv(np.dot(X_a.T, X_a))
      X_dot_y_a = np.dot(X_a, U_a)
      beta_hat_a = np.dot(Xprime_X_inv_a, X_dot_y_a)
      yhat_a = np.dot(X_dot_y_a, beta_hat_a)
      sigma_hat_a = np.sum((U_a - yhat_a)**2)
      sample_cov_a = sigma_hat_a * np.dot(X_a.T, np.dot(Xprime_X_inv_a, X_a))

      # Add to linear model results
      linear_model_results['beta_hat_list'].append(beta_hat_a)
      linear_model_results['Xprime_X_inv_list'].append(Xprime_X_inv_a)
      linear_model_results['X_list'].append(X_a)
      linear_model_results['X_dot_y_list'].append(X_dot_y_a)
      linear_model_results['sample_cov_list'].append(sample_cov_a)

    # Estimate context mean and variance
    estimated_context_mean = np.mean(X, axis=0)
    estimated_context_variance = np.cov(X)

    for t in range(T):
      zeta = truncated_thompson_sampling(env, linear_model_results, T, t, estimated_context_mean,
                                         estimated_context_variance, truncation_function, truncation_function_gradient,
                                         zeta)




