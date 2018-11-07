from Bandit import NormalMAB
import numpy as np
import bayes_opt
from src.policies import tuned_bandit_policies as tuned_bandit

def lm_expit_epsilon_decay(T, t, zeta, R, delta):
  covari = np.kron([1, T-t], np.kron([1, R],[1,delta]))
  return zeta[0] * expit(np.dot(zeta[1:9]),covari)

def mab_rollout(tuning_function_parameter, mab_epsilon_policy, 
                             time_horizon, tuning_function, bandit, 
                             gamma, mc_reps, est_means, std_errs):
  mean_cumulative_reward = 0
  R = std_errs[1]/std_errs[0]
  delta = est_means[1] - est_means[0]
  a = 0
  expit_wrapper = partial(tuning_function, R=R, delta=delta)
  for rep in range(mc_reps):
    r = 0
    for j in range(time_horizon):
      a = mab_epsilon_policy(est_means, std_errs, 1, expit_wrapper, tuning_function_parameter, T, t, bandit)
      r += gamma**j*bandit.reward_dbn(a)      
    mean_cumulative_reward += (r - mean_cumulative_reward)/(rep+1)
#    print(rep, r, mean_cumulative_reward)
  return mean_cumulative_reward

def bayesopt(rollout_function, policy, tuning_function, zeta_prev, time_horizon, env, mc_replicates,
             bounds, explore_, gamma, est_means, std_errs):

  def objective(zeta0, zeta1, zeta2, zeta3, zeta4, zeta5, zeta6, zeta7, zeta8):
    zeta = np.array([zeta0, zeta1, zeta2, zeta3, zeta4, zeta5, zeta6, zeta7, zeta8])
    return rollout_function(zeta, policy, time_horizon, tuning_function, 
                            env, gamma, mc_replicates, est_means, std_errs)

  explore_.update({'zeta{}'.format(i): [zeta_prev[i]] for i in range(len(zeta_prev))})
  bo = BayesianOptimization(objective, bounds)
  bo.explore(explore_)
  bo.maximize(init_points=10, n_iter=15, alpha=1e-4)
  best_param = bo.res['max']['max_params']
  best_param = np.array([best_param['zeta{}'.format(i)] for i in range(len(bounds))])
  return best_param

def episode(policy_name, label, mc_replicates=10, T=1000):
  np.random.seed(label)
  if policy_name == 'eps':
    tuning_function = lambda a, b, c: 0.0  # Constant epsilon
    tune = False
    tuning_function_parameter = None
  elif policy_name == "eps-decay":
    tuning_function = tuned_bandit.expit_epsilon_decay
    tune = True
    tuning_function_parameter = np.array([0.05, 1.0, 0.01]) 
    bounds = {'zeta0': (0.05, 2.0), 'zeta1': (1.0, 49.0), 'zeta2': (0.01, 2.5)}
    explore_ = {'zeta0': [1.0, 0.05, 1.0, 0.1], 'zeta1': [50.0, 49.0, 1.0, 49.0], 'zeta2': [0.1, 2.5, 1.0, 2.5]}
    rollout_function = mdp_grid_rollout
  elif policy_name == 'eps-fixed-decay':
    #tuning_function = lambda a, b, c: 0.1/float(b+1)
    #tune = False
    #tuning_function_parameter = None
    tuning_function = tuned_bandit.expit_epsilon_decay
    tune = False
    tuning_function_parameter = np.array([ 0.05     ,  43.46014702 , 2.5 ] )
#    tuning_function_parameter = np.array([ 2., 41.68182633, 2.5])

  policy = mdp_grid_epsilon_policy    
  gamma = 0.9
  env = Gridworld(time_horizon=T)
  time_horizon = T
  tuning_parameter_sequence = []
  rewards = []
  actions = []
  posterior_alphas = []
  nEpi = 0
  r = 0
  reward_mat = []
  while env.counter < env.time_horizon:
#    print(env.counter, r)
    s = env.reset()
    acts=[]
    for t in range(env.maxT):
#      print(env.counter, t, sum(rewards))
      if tune:
        tuning_function_parameter = bayesopt(rollout_function, policy, tuning_function, tuning_function_parameter, 
                                             time_horizon, env, mc_replicates, bounds, explore_, gamma)
        tuning_parameter_sequence.append([float(z) for z in tuning_function_parameter]) 
      update_transitionMatrices = env.posterior_mean_model()
#      print("estimated {}, true {}".format(update_transitionMatrices[:,0,:], env.transitionMatrices[:,0,:]))
      print("###########")
#      print(env.counter, update_transitionMatrices[1,:4,:], env.transitionMatrices[1,:4,:])
#      print(env.counter, sum(sum(abs(update_transitionMatrices[1,:3,:] - env.transitionMatrices[1,:3,:]))))
#      print(env.counter, sum(sum(abs(update_transitionMatrices[2,[3,7,10],:] - env.transitionMatrices[2,[3,7,10],:]))))
#      print(env.counter, sum(sum(sum(abs(update_transitionMatrices - env.transitionMatrices)))))
      
      mdp = Gridworld(transitionMatrices=update_transitionMatrices, time_horizon=time_horizon)
      _, _, Q = policy_iteration(mdp)
      optimal_action = np.argmax(Q[s, ])
      action = mdp_grid_epsilon_policy(optimal_action, tuning_function, tuning_function_parameter, 
                                       env.time_horizon, env.counter, env)
      s, reward, done = env.step(action)
      rewards.append(reward)
      acts.append(action)
      actions.append(action)
      new_post_alpha = env.posterior_alpha.copy()
      posterior_alphas.append(new_post_alpha)
#      print("after: {}".format(posterior_alphas))
      r += reward
      if done:
        print(acts)
#        print(env.counter, r)
        break
  print(sum(rewards))        
  return {'rewards':rewards, 'cum_rewards': sum(rewards), 'zeta_sequence': tuning_parameter_sequence,
          'actions': actions, 'posterior_alphas': posterior_alphas}
    
def run(policy_name, save=True, mc_replicates=10, T=1000):
  """

  :return:
  """

#  replicates = 48
  num_cpus = int(mp.cpu_count())
#  num_cpus = 4
  replicates = 20
  results = []
  pool = mp.Pool(processes=num_cpus)

  episode_partial = partial(episode, policy_name, mc_replicates=mc_replicates, T=T)

  results = pool.map(episode_partial, range(replicates))
#  cumulative_regrets = [np.float(d['cumulative_regret']) for d in results]
  zeta_sequences = [list(d['zeta_sequence']) for d in results]
  actions = [list(d['actions']) for d in results]
  cum_rewards = [float(d['cum_rewards']) for d in results]
  posterior_alphas = [d['posterior_alphas']for d in results]
#  rewards = [list(d['rewards'].astype(float)) for d in results]
#  print(policy_name, cum_rewards)
  print(policy_name, 'rewards', float(np.mean(cum_rewards)), 'se_rewards',float(np.std(cum_rewards))/np.sqrt(replicates))
  # Save results
  if save:
    results = {'T':T, 'mc_replicates': mc_replicates, 'cum_rewards': cum_rewards, 
               'rewards': float(np.mean(cum_rewards)), 'se_rewards':float(np.std(cum_rewards)/np.sqrt(replicates)),
               'zeta_sequences': zeta_sequences, 'actions': actions, 
               'posterior_alphas': posterior_alphas}#, 'rewards':rewards}

    base_name = 'mdp-grid-{}'.format(policy_name)
    prefix = os.path.join(project_dir, 'src', 'environments', base_name)
    suffix = datetime.datetime.now().strftime("%y%m%d_%H%M%S")
    filename = '{}_{}.yml'.format(prefix, suffix)
    np.save('{}_{}'.format(prefix, suffix), results)
    with open(filename, 'w') as outfile:
      yaml.dump(results, outfile)

  return


if __name__ == '__main__':
  start_time = time.time()
#  check_coef_converge()
#  episode('eps-decay', 0, T=75)
#  episode('eps-fixed-decay', 2, T=50)
#  run('eps-decay', save=True, T=2)
#  run('eps-fixed-decay', save=False, T=75)
#  run('eps', save=False, T=50)
  episode('eps', 1, T=50)
#  result = episode('eps', 0, T=1000)
#  print(result['cum_rewards'])
 # episode('eps-fixed-decay', 1, T=50)
#  num_processes = 4
#  num_replicates = num_processes
#  pool = mp.Pool(num_processes)
#  params = pool.map(bayesopt_under_true_model, range(num_processes))
#  params_dict = {str(i): params[i].tolist() for i in range(len(params))}
#  with open('bayes-opt-glucose.yml', 'w') as handle:
#    yaml.dump(params_dict, handle)
#  print(bayesopt_under_true_model(T=75))
#  print(rollout_under_true_model(np.array([1.  ,50.,   0.1]), mdp_grid_epsilon_policy, 
#                             50, tuned_bandit.expit_epsilon_decay, 0.9, 20))
  elapsed_time = time.time() - start_time
  print("time {}".format(elapsed_time))
  
  
  