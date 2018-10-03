import sys
import pdb
import os
this_dir = os.path.dirname(os.path.abspath(__file__))
project_dir = os.path.join(this_dir, '..', '..')
sys.path.append(project_dir)


import matplotlib.pyplot as plt
from src.environments.Bandit import NormalCB, NormalUniformCB
from src.policies import tuned_bandit_policies as tuned_bandit
from src.policies import global_optimization as opt
from src.policies import rollout
import copy
import numpy as np
from scipy.linalg import block_diag
from functools import partial
import datetime
import yaml
import multiprocessing as mp


def episode(policy_name, label, list_of_reward_betas=[[-10, 0.4, 0.4, -0.4], [-9.8, 0.6, 0.6, -0.4]], context_mean=np.array([0.0, 0.0, 0.0]),
            context_var=np.array([[1.0,0,0], [0,1.,0], [0, 0, 1.]]), list_of_reward_vars=[1, 1], T=50,
            mc_replicates=1000, pre_simulate=True):
  np.random.seed(label)

  # ToDo: Create policy class that encapsulates this behavior
  posterior_sample = False
  bootstrap_posterior = False
  positive_zeta = False
  if policy_name == 'eps':
    tuning_function = lambda a, b, c: 0.1  # Constant epsilon
    policy = tuned_bandit.linear_cb_epsilon_greedy_policy
    tune = False
    tuning_function_parameter = None
  elif policy_name == 'random':
    tuning_function = lambda a, b, c: 1.0  # Constant epsilon
    policy = tuned_bandit.linear_cb_epsilon_greedy_policy
    tune = False
    tuning_function_parameter = None
  elif policy_name == 'eps-decay-fixed':
    tuning_function = lambda a, t, c: 0.5 / np.sqrt(t + 1)
    policy = tuned_bandit.linear_cb_epsilon_greedy_policy
    tune = False
    tuning_function_parameter = None
  elif policy_name == 'eps-decay':
    tuning_function = tuned_bandit.expit_epsilon_decay
    policy = tuned_bandit.linear_cb_epsilon_greedy_policy
    tune = True
    tuning_function_parameter = np.ones(10) * 0.025
    posterior_sample = True
    bounds = {'zeta0': (0.05, 1.0), 'zeta1': (1.0, 49.0), 'zeta2': (0.01, 2.5)}
    explore_ = {'zeta0': [1.0, 0.05, 1.0, 0.1], 'zeta1': [50.0, 49.0, 1.0, 49.0], 'zeta2': [0.1, 2.5, 1.0, 2.5]}
  elif policy_name == 'greedy':
    tuning_function = lambda a, b, c: 0.00  # Constant epsilon
    policy = tuned_bandit.linear_cb_epsilon_greedy_policy
    tune = False
    tuning_function_parameter = None
  elif policy_name == 'worst':
    tuning_function = lambda a, b, c: 0.00
    policy = ref.linear_cb_worst_policy
    tune = False
    tuning_function_parameter = None
  elif policy_name == 'ts':
    tuning_function = lambda a, b, c: 1.0  # No shrinkage
    policy = tuned_bandit.linear_cb_thompson_sampling_policy
    tune = False
    tuning_function_parameter = None
  elif policy_name == 'ts-decay-posterior-sample':
    tuning_function = tuned_bandit.stepwise_linear_epsilon
    policy = tuned_bandit.linear_cb_thompson_sampling_policy
    tune = True
    tuning_function_parameter = np.ones(10)*0.1
    posterior_sample = True
  elif policy_name == 'ts-decay-bootstrap-sample':
    tuning_function = tuned_bandit.stepwise_linear_epsilon
    policy = tuned_bandit.linear_cb_thompson_sampling_policy
    tune = True
    tuning_function_parameter = np.ones(10)*0.1
    posterior_sample = True
    bootstrap_posterior = True
  elif policy_name == 'ts-decay':
    tuning_function = tuned_bandit.stepwise_linear_epsilon
    policy = tuned_bandit.linear_cb_thompson_sampling_policy
    tune = True
    tuning_function_parameter = np.ones(10)*0.1
  elif policy_name == 'ucb-tune-posterior-sample':
    tuning_function = tuned_bandit.stepwise_linear_epsilon
    policy = tuned_bandit.linear_cb_ucb_policy
    tune = True
    tuning_function_parameter = np.ones(10) * 0.025
    posterior_sample = True
  # elif policy_name == 'ts-shrink':
  #   tuning_function = tuned_bandit.expit_truncate
  #   policy = tuned_bandit.thompson_sampling_policy
  #   tune = True
  #   tuning_function_parameter = np.array([-2, 1])
  else:
    raise ValueError('Incorrect policy name')

  env = NormalCB(list_of_reward_betas=list_of_reward_betas, context_mean=context_mean, context_var=context_var,
                list_of_reward_vars=list_of_reward_vars)
#  env = NormalUniformCB(list_of_reward_betas=[np.ones(10) + 0.05, np.ones(10)], list_of_reward_vars=[0.01, 25])
  cumulative_regret = 0.0
  env.reset()
  tuning_parameter_sequence = []
  rewards = []
  actions = []
  for t in range(T):
    X = env.X
    estimated_context_mean = np.mean(X, axis=0)
    estimated_context_variance = np.cov(X, rowvar=False)
    estimated_context_bounds = (np.min(X), np.max(X[:, 1:]))

    if tune:
      if pre_simulate:
        if posterior_sample:
          gen_model_parameters = []
          for rep in range(mc_replicates):
            if bootstrap_posterior:
              pass
            else:
#              draws = env.sample_from_posterior()
              draws = env.sample_from_sampling_dist()
            betas_for_each_action = []
            vars_for_each_action = []
            for a in range(env.number_of_actions):
              beta_a = draws[a]['beta_draw']
              var_a = draws[a]['var_draw']
              betas_for_each_action.append(beta_a)
              vars_for_each_action.append(var_a)
            param_dict = {'reward_betas': betas_for_each_action, 'reward_vars': vars_for_each_action,
                          'context_max': draws['context_mean']}
#                          'context_max': draws['context_max']}
            gen_model_parameters.append(param_dict)
        else:
          gen_model_parameters = None

#        sim_env = NormalUniformCB(list_of_reward_betas=env.beta_hat_list, list_of_reward_vars=env.sigma_hat_list,
#                                  context_bounds=estimated_context_bounds)
        sim_env = NormalCB(list_of_reward_betas=list_of_reward_betas, context_mean=context_mean, context_var=context_var,
                           list_of_reward_vars=list_of_reward_vars)
        pre_simulated_data = sim_env.generate_mc_samples(mc_replicates, T, gen_model_params=gen_model_parameters)
        tuning_function_parameter = opt.bayesopt(rollout.normal_cb_rollout_with_fixed_simulations, policy,
                                                 tuning_function, tuning_function_parameter, T,
                                                 sim_env, mc_replicates,
                                                 {'pre_simulated_data': pre_simulated_data},
                                                 bounds, explore_, positive_zeta=positive_zeta)
        tuning_parameter_sequence.append([float(z) for z in tuning_function_parameter])
      else:
        tuning_function_parameter = tuned_bandit.random_search(tuned_bandit.oracle_rollout, policy, tuning_function,
                                                               tuning_function_parameter,
                                                               linear_model_results, T, t, estimated_context_mean,
                                                               estimated_context_variance, env)

    x = copy.copy(env.curr_context)
#    print('time {} epsilosn {}'.format(t, tuning_function(T,t,tuning_function_parameter)))
    beta_hat = np.array(env.beta_hat_list)
#    print(beta_hat, env.sample_mean, env.sample_se, env.sampling_cov_list)
    action = policy(beta_hat, env.sampling_cov_list, x, tuning_function, tuning_function_parameter, T, t, env)
    res = env.step(action)
    cumulative_regret += -env.regret(action, x)
    actions.append(action)
    u = res['Utility']
    rewards.append(u)
#    cumulative_regret += u
#    print(t, beta_hat[1])
  return {'cumulative_regret': cumulative_regret, 'zeta_sequence': tuning_parameter_sequence,
          'rewards': rewards, 'actions': actions}


def run(policy_name, save=True, mc_replicates=1000, T=100):
  """

  :return:
  """

  # These were randomly generated acc to ?
  list_of_reward_betas=[[-10, 0.4, 0.4, -0.4], [-9.8, 0.6, 0.6, -0.4]] 
  context_mean=np.array([0.0, 0.0, 0.0])
  context_var=np.array([[1.0,0,0], [0,1.,0], [0, 0, 1.]])
  list_of_reward_vars=[1, 1]

  replicates = 16
  num_cpus = int(mp.cpu_count())
  results = []
  pool = mp.Pool(processes=num_cpus)

  episode_partial = partial(episode, policy_name, list_of_reward_betas=list_of_reward_betas, context_mean=context_mean,context_var=context_var,
                            list_of_reward_vars=list_of_reward_vars, mc_replicates=mc_replicates, T=T)

  results = pool.map(episode_partial, range(replicates))
  cumulative_regrets = [np.float(d['cumulative_regret']) for d in results]
  zeta_sequences = [d['zeta_sequence'] for d in results]
  actions = [d['actions'] for d in results]
  rewards = [d['rewards'] for d in results]
  print(policy_name, 'mean_regrets', float(np.mean(cumulative_regrets)), 'std_rewards',float(np.std(cumulative_regrets))/np.sqrt(replicates))
  # Save results
  if save:
    results = {'mean_regret': float(np.mean(cumulative_regrets)), 'std_regret': float(np.std(cumulative_regrets)),
               'beta_hat_list': [beta for beta in list_of_reward_betas],
               'context_mean': [float(c) for c in context_mean], 'regret list': [float(r) for r in cumulative_regrets],
               'reward_vars': list_of_reward_vars, 'actions': actions, 'rewards': rewards,
               'zeta_sequences': zeta_sequences}

    base_name = 'normalcb-{}'.format(policy_name)
    prefix = os.path.join(project_dir, 'src', 'run', base_name)
    suffix = datetime.datetime.now().strftime("%y%m%d_%H%M%S")
    filename = '{}_{}.yml'.format(prefix, suffix)
    with open(filename, 'w') as outfile:
      yaml.dump(results, outfile)

  return


if __name__ == '__main__':
  episode('eps-decay', 20)
#  episode('eps',0)
   run('eps')
#  run('greedy', T=50)
#  run('eps-decay-fixed', T=50)
#  run('eps', T=1000)
<<<<<<< HEAD
  # run('eps-decay', T=20)
=======
#  run('eps-decay', T=20)
>>>>>>> 9c49042a785ebfc27f4dff26c544570cd7d93e95
#  run('random', T=100)
#  run('eps-decay', T=10)
  # episode('ts-decay-posterior-sample', 0, T=10, mc_replicates=100)
  # episode('ucb-tune-posterior-sample', 0, T=10, mc_replicates=100)
  # run('ts-decay-posterior-sample', T=10, mc_replicates=100)
  # run('ucb-tune-posterior-sample', T=10, mc_replicates=100)

