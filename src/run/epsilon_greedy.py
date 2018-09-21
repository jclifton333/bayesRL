import sys
import pdb
import os
this_dir = os.path.dirname(os.path.abspath(__file__))
project_dir = os.path.join(this_dir, '..', '..')
sys.path.append(project_dir)


from src.environments.Bandit import NormalCB
from src.policies import tuned_bandit_policies as tuned_bandit
import copy
import numpy as np
from scipy.linalg import block_diag
import matplotlib.pyplot as plt
import matplotlib as mpl

def main():
  """
  For now this will just be for truncated thompson sampling policy.
  :return:
  """

  # Simulation settings
  replicates = 100
  T = 25
  env = NormalCB()

  rewards = np.zeros((replicates, T))
  epsilon_decay = False
  epsilon_fix = 0.05
  # Run sims
  for replicate in range(replicates):
    env.reset()

    # Initial pulls (so we can fit the models)
    for a in range(env.number_of_actions):
      for p in range(env.context_dimension):
        env.step(a)

    # Get initial linear model
    X, A, U = env.X, env.A, env.U
    linear_model_results = {'beta_hat_list': [], 'Xprime_X_inv_list': [], 'X_list': [], 'X_dot_y_list': [],
                            'sample_cov_list': [], 'sigma_hat_list': []}
    for a in range(env.number_of_actions):  # Fit linear model on data from each actions
      # Get observations where action == a
      indices_for_a = np.where(A == a)
      X_a = X[indices_for_a]
      U_a = U[indices_for_a]

      Xprime_X_inv_a = np.linalg.inv(np.dot(X_a.T, X_a))
      X_dot_y_a = np.dot(X_a.T, U_a)
      beta_hat_a = np.dot(Xprime_X_inv_a, X_dot_y_a)
      yhat_a = np.dot(X_a, beta_hat_a)
      sigma_hat_a = np.sum((U_a - yhat_a)**2)
      sample_cov_a = sigma_hat_a * Xprime_X_inv_a

      # Add to linear model results
      linear_model_results['beta_hat_list'].append(beta_hat_a)
      linear_model_results['Xprime_X_inv_list'].append(Xprime_X_inv_a)
      linear_model_results['X_list'].append(X_a)
      linear_model_results['X_dot_y_list'].append(X_dot_y_a)
      linear_model_results['sample_cov_list'].append(sample_cov_a)
      linear_model_results['sigma_hat_list'].append(sigma_hat_a)

    for t in range(T):

      # Sample beta
      beta_hat = np.vstack(linear_model_results['beta_hat_list'])
   

      # Get reward
      x = copy.copy(env.curr_context)
      estimated_rewards = np.dot(beta_hat, env.curr_context)
      eps = np.random.rand()
      epsilon = epsilon_fix
#      if epsilon_decay == True:
#        if t > T/3.0:
#          epsilon = epsilon_fix**(t/(T/3.0))
              
      if eps < epsilon:
        a = np.random.choice(env.number_of_actions)
      else:
        a = np.argmax(estimated_rewards)
      step_results = env.step(a)
      reward = step_results['Utility']
      rewards[replicate, t] = reward

      # Update linear model
      linear_model_results = tuned_bandit.update_linear_model_at_action(a, linear_model_results, x, reward)

  return rewards


if __name__ == '__main__':
  main()


mpl.style.use('seaborn')
fig, ax = plt.subplots(figsize=(5, 5))
ax.plot(np.mean(rewards,axis=0))
ax.fill_between(range(T), np.mean(rewards,axis=0)- np.std(rewards,axis=0),  
                np.mean(rewards,axis=0)+ np.std(rewards,axis=0), 
                facecolor='m', alpha=0.5)
print(sum(rewards.T))
print(np.mean(rewards))


ax.set_title('Linear VS Loc_Linear'.format('seaborn'), color='C0')
ax.set_ylabel('Timesteps')
ax.set_xlabel('Episodes')
ax.plot(num_steps_Loc, 'C'+str(1), label='Linear')
ax.plot(range(100), 'C2', label='Local linear')
ax.legend()
plt.savefig('Linear_Loc.png')
