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
import pickle
    
beta1 = 1
# beta2_list = [-3, -2, 0.01, 0.1, 0.2, 0.5, 1, 2]
beta2_list = [1.0]


def main():
  """
  For now this will just be for truncated thompson sampling policy.
  :return:
  """

  # Simulation settings
<<<<<<< HEAD
  replicates = 1
  T = 3
=======
  replicates = 100
  T = 25
>>>>>>> c4d50fbbb6d3eabf544e6d29a6fafa7278349d6c
  for beta2 in beta2_list:
    env = NormalCB(list_of_reward_betas=[[1,1],[beta1+beta2, beta1-beta2]],
                   list_of_reward_vars=[[1],[1]])
    
  #  rewards = np.zeros((replicates, T))
    regrets = np.zeros((replicates, T))
    epsilon_fix = 0.05
    epsilon_decay = True
<<<<<<< HEAD
    zeta = np.array([0.2, -5, 1])
=======
    zeta = np.array([0.2, -2, 1])
>>>>>>> c4d50fbbb6d3eabf544e6d29a6fafa7278349d6c

    # Run sims
    for replicate in range(replicates):
      env.reset()
  
      # Initial pulls (so we can fit the models)
      for a in range(env.number_of_actions):
        for p in range(2*env.context_dimension):
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

      beta_hat_save = np.zeros((T, env.number_of_actions*env.context_dimension))
      beta_hat_save_mean = np.zeros((T, env.number_of_actions*env.context_dimension))
      for t in range(T):
        beta_hat = np.vstack(linear_model_results['beta_hat_list'])
        eachbeta_hat = np.hstack(linear_model_results['beta_hat_list'])
        beta_hat_save[t,] = eachbeta_hat
        
        # Get reward
        x = copy.copy(env.curr_context)
        estimated_rewards = np.dot(beta_hat, env.curr_context)
        eps = np.random.rand()
        epsilon = epsilon_fix
        if epsilon_decay:
<<<<<<< HEAD
          zeta = tuned_bandit.tune_epsilon_greedy(linear_model_results, T, t, np.mean(env.X, axis=0),
                                                  np.cov(env.X, rowvar=False), None, None, zeta)
=======
          param_grid = np.zeros((0, 3))
          for rand_param in range(20):
            kappa = np.random.uniform(low=0.05, high=0.3)
            zeta_0 = np.random.uniform(low=-3, high=-0.05)
            zeta_1 = np.random.uniform(low=1, high=5)
            param_grid = np.vstack((param_grid, np.array([kappa, zeta_0, zeta_1])))
          param_scores = tuned_bandit.grid_search(tuned_bandit.epsilon_greedy_rollout, param_grid,
                                                  linear_model_results, T, t, np.mean(env.X, axis=0),
                                                  np.cov(env.X, rowvar=False))
          bo = tuned_bandit.bayesopt(tuned_bandit.epsilon_greedy_rollout, zeta, linear_model_results, T, t, np.mean(env.X, axis=0),
                                     np.cov(env.X, rowvar=False))
          pdb.set_trace()
          zeta = param_grid[np.argmax(param_scores), :]
>>>>>>> c4d50fbbb6d3eabf544e6d29a6fafa7278349d6c
          epsilon = tuned_bandit.expit_epsilon_decay(T, t, zeta)
          print('epsilon {}'.format(epsilon))
#          if t > T/5.0:
#            epsilon = epsilon_fix**(t/(T/5.0))
                
        if eps < epsilon:
          a = np.random.choice(env.number_of_actions)
        else:
          a = np.argmax(estimated_rewards)
        step_results = env.step(a)
        reward = step_results['Utility']
  #      rewards[replicate, t] = reward
        
        # Get regret
  #      beta_true = env.list_of_reward_betas
  #      regret = np.dot(beta_true, x)-reward
  #      regrets[replicate, t] = regret      
             
        oracle_expected_reward = np.max((np.dot(x, env.list_of_reward_betas[0]), np.dot(x, env.list_of_reward_betas[1])))
        regret = oracle_expected_reward - np.dot(x, env.list_of_reward_betas[a])
        regrets[replicate, t] = regret
  
        # Update linear model
        linear_model_results = tuned_bandit.update_linear_model_at_action(a, linear_model_results, x, reward)
      
      beta_hat_save_mean += beta_hat_save
      print('cum regret {}'.format(np.sum(regrets[replicate, :])))
    beta_hat_save_mean = beta_hat_save_mean/replicates
    
    # Save data
    cum_regret = np.sum(regrets, axis=1)
    cum_regret_mean = np.mean(cum_regret)
    cum_regret_std = np.std(cum_regret)
    print(cum_regret_mean, cum_regret_std)
    
    a = {'cum_regret_mean': cum_regret_mean, 'cum_regret_std':cum_regret_std,
         'beta_hat_save_mean':beta_hat_save_mean, 'beta2': beta2+beta1}
    # with open('/Users/lili/Documents/labproject2017/aistat/plots_results/sigmoid0_05.pickle', 'wb') as handle:
    #   pickle.dump(a, handle, protocol=pickle.HIGHEST_PROTOCOL)
  return cum_regret


if __name__ == '__main__':
  main()


# mpl.style.use('seaborn')
# fig, ax = plt.subplots(figsize=(5, 5))
# ax.plot(np.mean(rewards,axis=0))
# ax.fill_between(range(T), np.mean(rewards,axis=0)- np.std(rewards,axis=0),
#                 np.mean(rewards,axis=0)+ np.std(rewards,axis=0),
#                 facecolor='m', alpha=0.5)
# mean_cum = np.mean(np.sum(rewards, axis=1))
# cum_regret = np.sum(rewards, axis=1)
# plt.hist(cum_regret)
# print(sum(rewards.T))
# print(np.std(sum(rewards[:,:25].T)))
# print(np.mean(rewards[:,:25]))
#
#
# ax.set_title('Linear VS Loc_Linear'.format('seaborn'), color='C0')
# ax.set_ylabel('Timesteps')
# ax.set_xlabel('Episodes')
# ax.plot(num_steps_Loc, 'C'+str(1), label='Linear')
# ax.plot(range(100), 'C2', label='Local linear')
# ax.legend()
<<<<<<< HEAD
# plt.savefig('Linear_Loc.png')
=======
# plt.savefig('Linear_Loc.png')
>>>>>>> c4d50fbbb6d3eabf544e6d29a6fafa7278349d6c
