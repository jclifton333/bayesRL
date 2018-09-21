from src.environments.Bandit import NormalCB
from src.policies import tuned_bandit_policies as tuned_bandit
import numpy as np
from scipy.linalg import block_diag


def main():
  """
  For now this will just be for truncated thompson sampling policy.

  :return:
  """

  # Simulation settings
  replicates = 50
  T = 25
  truncation_function = tuned_bandit.expit_truncate
  truncation_function_gradient = tuned_bandit.expit_truncate_gradient
  zeta = np.array([-5, 0.3])  # Initial truncation function parameters
  env = NormalCB()

  rewards = np.zeros((replicates, T))

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
      # Get exploration parameter
      zeta = tuned_bandit.tune_truncated_thompson_sampling(linear_model_results, T, t, estimated_context_mean,
                                                           estimated_context_variance, truncation_function,
                                                           truncation_function_gradient, zeta)
      shrinkage = truncation_function(T-t, zeta)

      # Sample beta
      beta_hat = linear_model_results['beta_hat_list'].flatten()
      sample_cov_hat = block_diag(linear_model_results['sample_cov_list'])
      beta_tilde = np.random.multivariate_normal(beta_hat, sample_cov_hat * shrinkage)

      # Get reward
      beta_tilde = beta_tilde.reshape((env.number_of_actions, env.context_dimension))
      estimated_rewards = np.dot(env.current_context, beta_tilde)
      a = np.argmax(estimated_rewards)
      u = env.step(a)
      rewards[replicate, t] = u

  return rewards


