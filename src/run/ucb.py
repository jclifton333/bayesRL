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
import scipy.stats

    
beta1 = 1
beta2_list = [-3, -2, 0.01, 0.1, 0.2, 0.5, 1, 2]
alpha_fix = 0.55
zeta1 = 0.08
alpha_tune = True

def main():
  """
  For now this will just be for truncated thompson sampling policy.
  :return:
  """

  # Simulation settings
  replicates = 100
  T = 10
  for beta2 in beta2_list:
    env = NormalCB(list_of_reward_betas=[[1,1],[beta1+beta2, beta1+beta2]], 
                   list_of_reward_vars=[[1],[1]])
    
  #  rewards = np.zeros((replicates, T))
    regrets = np.zeros((replicates, T))
    zeta = np.array([0.2, -5, 1])

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
        if alpha_tune:          
          alpha = 1/(1+np.exp(-zeta1*(T-t-1)))
          print(alpha)
        else:
          alpha = alpha_fix
          
        beta_hat = np.vstack(linear_model_results['beta_hat_list'])
        eachbeta_hat = np.hstack(linear_model_results['beta_hat_list'])
        beta_hat_save[t,] = eachbeta_hat
        
        # Get reward
        x = copy.copy(env.curr_context)
        estimated_rewards = np.dot(beta_hat, env.curr_context) 
        
        # Get UCB
        q_hat_0 = estimated_rewards[0]
        q_hat_1 = estimated_rewards[1]
        b = env.X[-1,]
        
        B_0 = env.X[np.where(env.A==0),][0]
        U_0 = env.U[np.where(env.A==0),][0]
        omega_hat_0 = np.zeros([env.context_dimension, env.context_dimension])
        Sigma_hat_0 = np.zeros([env.context_dimension, env.context_dimension])
        for i in range( int(len(env.A)-sum(env.A)) ):
          omega_hat_0 += np.outer(B_0[i],B_0[i].T) /(len(env.A)-sum(env.A))
          Sigma_hat_0 += np.outer(B_0[i],B_0[i].T)*(U_0[i]-np.dot(B_0[i],beta_hat[0]))**2/(len(env.A)-sum(env.A))
        sigma_hat_0 = np.dot(np.dot(b, np.matmul(np.matmul(np.linalg.inv(omega_hat_0), Sigma_hat_0),
                              np.linalg.inv(omega_hat_0))), b)
        
        B_1 = env.X[np.where(env.A==1),][0]
        U_1 = env.U[np.where(env.A==1),][0]
        omega_hat_1 = np.zeros([env.context_dimension, env.context_dimension])
        Sigma_hat_1 = np.zeros([env.context_dimension, env.context_dimension])
        for i in range( int(sum(env.A)) ):
          omega_hat_1 += np.outer(B_1[i],B_1[i].T) /sum(env.A)
          Sigma_hat_1 += np.outer(B_1[i],B_1[i].T)*(U_1[i]-np.dot(B_1[i],beta_hat[1]))**2/sum(env.A)
        sigma_hat_1 = np.dot(np.dot(b, np.matmul(np.matmul(np.linalg.inv(omega_hat_1), Sigma_hat_1),
                              np.linalg.inv(omega_hat_1))), b)
        
        kexi_0 = q_hat_0 + scipy.stats.norm.ppf(1-alpha)*sigma_hat_0/np.sqrt(t+1)
        kexi_1 = q_hat_1 + scipy.stats.norm.ppf(1-alpha)*sigma_hat_1/np.sqrt(t+1)        
        
        a = np.argmax([kexi_0, kexi_1])
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
#      print('cum regret {}'.format(np.sum(regrets[replicate, :])))
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
# plt.savefig('Linear_Loc.png')