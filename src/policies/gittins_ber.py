"""
Created on Sat Jan  5 15:50:18 2019

Finite-horizon Gittins index for Bernoulli bandit 
from the paper "On Bayesian index policies for sequential resource allocation" page13
(https://arxiv.org/pdf/1601.01190.pdf)
"""
import numpy as np

def values_tau(tau, Lambda, alpha = 1.0, beta = 1.0):
  '''
  alpha, beta: prior parameters of Beta disttribution
  tau: from time 0 to T-t to calculate value
  t: the current time step, t=1,...,T-1, in order to calculate gittins index to decide A_{t+1}
  output: the values of each state for time horizon = T 
  At time t which means the sum number of pulls from all arms equal to t 
  '''
  if tau == 0:
    value = 0.0
  elif tau == 1:
    value = alpha/(alpha+beta) - Lambda
  else:
    values_T = (alpha+np.arange(tau))/(alpha+beta+tau-1) - Lambda
    Expect_Y = (alpha+np.arange(tau-1))/(alpha+beta+tau-2)
    values_Tminus1 = Expect_Y-Lambda + Expect_Y*values_T[1:] + (1-Expect_Y)*values_T[:-1]
    values_table_list = [[]]*tau
    values_table_list[0] = values_T
    values_table_list[1] = values_Tminus1
    for tt in range(2, tau):
      Expect_Y = (alpha+np.arange(tau-tt))/(alpha+beta+tau-tt-1)
      values_table_list[tt] = Expect_Y-Lambda + Expect_Y*values_table_list[tt-1][1:] + \
                             (1-Expect_Y)*values_table_list[tt-1][:-1]
#    print(values_table_list)
    value = values_table_list[-1][0]
  return value

def values_over_tau(remain_t, Lambda, alpha, beta):
  '''
  remain_t: remaining time steps, T-t, (recall t=1,...,T-1)
  output: the supreme values of each state over tau
  '''
  values = []
  for tau in range(remain_t+1):
    values = np.append(values, values_tau(tau, Lambda, alpha, beta))
  if len(values) == 1:
    value_max = values
  else:
    value_max = np.max(values)
#  print(values)
  return value_max
 
def gittins_index(t, num_pulls_one_arm, s_t, values_for_lambdas_dict, 
                  lambdas):
  '''
  T: time horizon
  t: the current time point
  s_t: the number of successes at time t
  values_for_lambdas_list:  each element in the dictionary, corresponding to one Lambda, is values over each tau 
  output: Lambda, i.e. gittins_index
  '''
  threshold = 0.001
  ## choose the smallest Lambda which gives the value close enough to 0 ##
  for Lambda in lambdas:
    key_name = str(Lambda)+'_'+str(t)+'_'+str(num_pulls_one_arm)+'_'+str(s_t)
    v_lambda = values_for_lambdas_dict[key_name]
    if abs(v_lambda) < threshold:
#      print(t, num_pulls_one_arm, s_t, v_lambda, Lambda)
      return float(Lambda)

def generate_values_for_different_lambdas():
  alpha0 = beta0 = 1.0
  T=51
  lambdas = np.linspace(0.1, 0.9, 100)
  values_for_lambdas_dict = {}
  for t in range(1, T): ## t = the sum of number of pulls for each arm ##
    print(t)
    for num_pulls_one_arm in range(1, t+1):
      for s_t in range(num_pulls_one_arm+1):
        for Lambda in lambdas:
          alpha = alpha0 + s_t
          beta = beta0 + num_pulls_one_arm - s_t
          key_name = str(Lambda)+'_'+str(t)+'_'+str(num_pulls_one_arm)+'_'+str(s_t)
          values_for_lambdas_dict[key_name] = values_over_tau(T-t, Lambda, alpha, beta)
  return values_for_lambdas_dict, lambdas
    
   
if __name__ == '__main__':
  values_for_lambdas_dict, lambdas = generate_values_for_different_lambdas()
  results_mean = []
  for rep in range(192):
    np.random.seed(rep)
    env = BernoulliMAB(list_of_reward_mus=[0.3, 0.6])
#    env = BernoulliMAB(list_of_reward_mus=[0.73, 0.56, 0.33, 0.04, 0.66])
#    env = BernoulliMAB(list_of_reward_mus=[0.74, 0.15, 0.34, 0.48, 0.53, 0.23, 0.47, 0.51, 0.71, 0.42])
    env.reset()
    optimal_mu = np.max(env.list_of_reward_mus)
    # Initial pulls
    for a in range(env.number_of_actions):
      env.step(a)
    
    Gindex_each_arm = np.empty([env.number_of_actions])
    cumulative_regrets = 0.0
    for t in range(1, T):
      ## \epsilon-greedy ##
  #    if np.random.rand() < 0.05:
  #      action = np.random.choice(2)
  #    else:
  #      action = np.argmax(env.estimated_means)
  #    env.step(action)
      
      ## Gittins index ##
      number_of_successes = env.estimated_means*env.number_of_pulls
      for k in range(env.number_of_actions):
        Gindex_each_arm[k] = gittins_index(t, int(env.number_of_pulls[k]), 
                       int(number_of_successes[k]), values_for_lambdas_dict, lambdas)
      action = np.argmax(Gindex_each_arm)
      env.step(action)
      
      regret = optimal_mu - env.list_of_reward_mus[action]
      cumulative_regrets = cumulative_regrets + regret
    results_mean = np.append(results_mean, cumulative_regrets)
  print("Mean Regrets: ", np.mean(results_mean), 'se:', np.std(results_mean)/np.sqrt(len(results_mean)))
  
    
    
    
  
                           
                           