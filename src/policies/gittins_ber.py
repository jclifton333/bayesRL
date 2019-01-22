"""
Created on Sat Jan  5 15:50:18 2019

Finite-horizon Gittins index for Bernoulli bandit 
from the paper "On Bayesian index policies for sequential resource allocation" page13
(https://arxiv.org/pdf/1601.01190.pdf)
"""
import numpy as np

def values_tau(T, Lambda):
  '''
  output: the values of each state for time horizon = T 
  At time t which means pulling one arm t times, the number of states is t+1, which is the number of successes for one arm
  '''
  alpha = 1.0; beta = 1.0 ## prior parameters of Beta disttribution ##
  values_T = (alpha+np.arange(T+1))/(alpha+beta+T) - Lambda
  Expect_Y = (alpha+np.arange(T))/(alpha+beta+T)
  values_Tminus1 = Expect_Y-Lambda + Expect_Y*values_T[1:] + (1-Expect_Y)*values_T[:-1]
  values_table_list = [[]]*T
  values_table_list[0] = values_T
  values_table_list[1] = values_Tminus1
  for t in range(2, T):
    Expect_Y = (alpha+np.arange(T+1-t))/(alpha+beta+T+1-t)
    values_table_list[t] = Expect_Y-Lambda + Expect_Y*values_table_list[t-1][1:] + \
                           (1-Expect_Y)*values_table_list[t-1][:-1]
  return values_table_list

def values_over_tau(T, Lambda):
  '''
  T: time horizon
  output: the supreme values of each state over tau
  '''
  sup_values_list = values_tau(T, Lambda)
  for tau in range(2, T):
    temp = values_tau(tau, Lambda)
    for k in range(tau):
      ## [-tau:][k] is the same as [T-tau+k] ##
      sup_values_list[T-tau+k] = np.max(np.vstack((temp[k], sup_values_list[-tau:][k])), axis=0)
  return sup_values_list
 
def gittins_index(t, s_t, values_for_lambdas_dict):
  '''
  T: time horizon
  t: the current time point
  s_t: the number of successes at time t
  values_for_lambdas_list:  each element in the dictionary, corresponding to one Lambda, is values over each tau 
  output: Lambda, i.e. gittins_index
  '''
  threshold = 0.001
  ## choose the smallest Lambda which gives the value close enough to 0 ##
  for k in values_for_lambdas_dict.keys(): 
    v_k = values_for_lambdas_dict[k][-t][s_t]
    if abs(v_k) < threshold:
      return float(k)
      
    


T=50
lambdas = np.linspace(0.1, 0.8, 1000)
values_for_lambdas_dict = {}
for Lambda in lambdas:
  values_for_lambdas_dict[str(Lambda)] = values_over_tau(T, Lambda)

results_mean = []
for rep in range(100):
  np.random.seed(rep)
  env = BernoulliMAB()
  env.reset()
  # Initial pulls
  for a in range(env.number_of_actions):
    env.step(a)
  
  Gindex_each_arm = np.empty([env.number_of_actions])
  for t in range(2, T):
    if np.random.rand() < 0.05:
      action = np.random.choice(2)
    else:
      action = np.argmax(env.estimated_means)
    env.step(action)
    
#    number_of_successes = env.estimated_means*env.number_of_pulls
#    for k in range(env.number_of_actions):
#      Gindex_each_arm[k] = gittins_index(t, int(number_of_successes[k]), values_for_lambdas_dict)
#    action = np.argmax(Gindex_each_arm)
#    env.step(action)
    
    results_mean = np.append(results_mean, np.mean(env.U))

print("Mean utility: ", np.mean(results_mean))
    
   
if __name__ == '__main__':
  start_time = time.time()
#  check_coef_converge()
#  episode('eps-decay', 0, T=5)
#  episode('eps-fixed-decay', 0, T=5)
#  run('eps-decay', T= 25, mc_replicates=10, AR1=False)
#  run('eps-fixed-decay', T=50, mc_replicates=10, AR1=False)
  run('eps',save=False, T=50, mc_replicates=10, AR1=False)
#  episode('eps', 0, T=25)
#  result = episode('eps', 0, T=50)
  # print(result['actions'])
 # episode('eps-fixed-decay', 1, T=50)
#  num_processes = 4
#  num_replicates = num_processes
#  pool = mp.Pool(num_processes)
#  params = pool.map(bayesopt_under_true_model, range(num_processes))
#  params_dict = {str(i): params[i].tolist() for i in range(len(params))}
#  with open('bayes-opt-glucose.yml', 'w') as handle:
#    yaml.dump(params_dict, handle)
#  print(bayesopt_under_true_model(T=25))
  elapsed_time = time.time() - start_time
  print("time {}".format(elapsed_time))
  # episode('ts-decay-posterior-sample', 0, T=10, mc_replicates=100)
  # episode('ucb-tune-posterior-sample', 0, T=10, mc_replicates=100)
  # run('ts-decay-posterior-sample', T=10, mc_replicates=100)
  # run('ucb-tune-posterior-sample', T=10, mc_replicates=100)
  
  
  
  

                         
                         