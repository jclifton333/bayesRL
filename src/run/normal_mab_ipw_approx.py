import sys
import pdb
import os
this_dir = os.path.dirname(os.path.abspath(__file__))
project_dir = os.path.join(this_dir, '..', '..')
sys.path.append(project_dir)


from src.policies import tuned_bandit_policies as tuned_bandit
from src.policies import gittins_index_policies as gittins
from src.policies.ipw_regret_estimate import max_ipw_regret
from src.policies import rollout
from src.environments.Bandit import NormalMAB
import src.policies.global_optimization as opt
import numpy as np
from functools import partial
import datetime
import yaml
import multiprocessing as mp


def episode(label, std=0.1, list_of_reward_mus=[0.3,0.6], T=50, monte_carlo_reps=1000, posterior_sample=False):
  np.random.seed(label)
  env = NormalMAB(list_of_reward_mus=list_of_reward_mus, list_of_reward_vars=[std**2]*len(list_of_reward_mus))

  cumulative_regret = 0.0
  mu_opt = np.max(env.list_of_reward_mus)
  env.reset()
  tuning_parameter_sequence = []
  # Initial pulls
  for a in range(env.number_of_actions):
    env.step(a)

  estimated_means_list = []
  estimated_vars_list = []
  actions_list = []
  rewards_list = []
  for t in range(T):
    estimated_means_list.append([float(xbar) for xbar in env.estimated_means])
    estimated_vars_list.append([float(s) for s in env.estimated_vars])

    if tune:
      if posterior_sample:
        reward_means = []
        reward_vars = []
        for rep in range(monte_carlo_reps):
          if bootstrap_posterior:
            draws = env.sample_from_bootstrap()
          else:
            draws = env.sample_from_posterior()
          means_for_each_action = []
          vars_for_each_action = []
          for a in range(env.number_of_actions):
            mean_a = draws[a]['mu_draw']
            var_a = draws[a]['var_draw']
            means_for_each_action.append(mean_a)
            vars_for_each_action.append(var_a)
          reward_means.append(means_for_each_action)
          reward_vars.append(vars_for_each_action)
      else:
        reward_means = None
        reward_vars = None

      sim_env = NormalMAB(list_of_reward_mus=env.estimated_means, list_of_reward_vars=env.estimated_vars)
      pre_simulated_data = sim_env.generate_mc_samples(monte_carlo_reps, T, reward_means=reward_means,
                                                       reward_vars=reward_vars)
      print(sim_env.estimated_means)
      tuning_function_parameter = opt.bayesopt(max_ipw_regret, policy, tuning_function,
                                               tuning_function_parameter, T, env, monte_carlo_reps,
                                               {'pre_simulated_data': pre_simulated_data},
                                               bounds, explore_, positive_zeta=positive_zeta)
      tuning_parameter_sequence.append([float(z) for z in tuning_function_parameter])

    print('standard errors {}'.format(env.standard_errors))
    print('estimated vars {}'.format(env.estimated_vars))
    if policy_name == 'gittins':
      estimated_means = []
      for aa in range(env.number_of_actions):
        estimated_means.append(sum(env.draws_from_each_arm[aa])/env.number_of_pulls[aa])
      action = policy(estimated_means, env.standard_errors, env.number_of_pulls, tuning_function,
                    tuning_function_parameter, T, t, env)
    else:
      action = policy(env.estimated_means, env.standard_errors, env.number_of_pulls, tuning_function,
                    tuning_function_parameter, T, t, env)
    res = env.step(action)
    u = res['Utility']
    actions_list.append(int(action))
    rewards_list.append(float(u))

    # Compute regret
    regret = mu_opt - env.list_of_reward_mus[action]
    cumulative_regret += regret

  return {'cumulative_regret': cumulative_regret, 'zeta_sequence': tuning_parameter_sequence,
          'estimated_means': estimated_means_list, 'estimated_vars': estimated_vars_list,
          'rewards_list': rewards_list, 'actions_list': actions_list}