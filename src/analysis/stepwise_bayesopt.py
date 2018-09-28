import pdb
import os
import sys
this_dir = os.path.dirname(os.path.abspath(__file__))
project_dir = os.path.join(this_dir, '..', '..')
sys.path.append(project_dir)
import numpy as np
from bayes_opt import BayesianOptimization
import multiprocessing as mp
from src.environments.Bandit import NormalMAB, BernoulliMAB, NormalCB, NormalUniformCB
from src.policies.rollout import normal_cb_rollout_with_fixed_simulations
from src.policies.tuned_bandit_policies import linear_cb_epsilon_greedy_policy, stepwise_linear_epsilon
import src.analysis.stepwise as stepwise
import matplotlib.pyplot as plt
import yaml


def bayes_optimize_zeta(seed, mc_rep=100, T=1000):
  np.random.seed(seed)

  env = NormalUniformCB()
  # X = env.X
  # estimated_context_mean = np.mean(X, axis=0)
  # estimated_context_variance = np.cov(X, rowvar=False)
  # estimated_context_bounds = (np.min(X), np.max(X))
  sim_env = NormalUniformCB(list_of_reward_betas=env.list_of_reward_betas, list_of_reward_vars=env.list_of_reward_vars,
                            context_bounds=env.context_bounds)
  pre_simulated_data = sim_env.generate_mc_samples(mc_rep, T)
  rollout_function_kwargs = {'pre_simulated_data': pre_simulated_data}

  def objective(zeta0, zeta1, zeta2, zeta3, zeta4, zeta5, zeta6, zeta7, zeta8, zeta9):
    zeta = np.array([zeta0, zeta1, zeta2, zeta3, zeta4, zeta5, zeta6, zeta7, zeta8, zeta9])
    return normal_cb_rollout_with_fixed_simulations(zeta, linear_cb_epsilon_greedy_policy, T,
                                                    stepwise_linear_epsilon, sim_env, **rollout_function_kwargs)

  bounds = {'zeta{}'.format(i): (0.0, 0.2) for i in range(10)}
  bo = BayesianOptimization(objective, bounds)
  bo.maximize(init_points=5, n_iter=20, alpha=1e-4)
  best_param = bo.res['max']['max_params']
  best_param = np.array([best_param['zeta{}'.format(i)] for i in range(10)])
  return best_param


if __name__ == "__main__":
  # num_processes = mp.cpu_count()
  # num_replicates = num_processes
  # pool = mp.Pool(num_processes)
  # params = []
  # for batch in range(int(num_replicates / num_processes)):
  #   params += pool.map(bayes_optimize_zeta, range(batch*num_processes, (batch+1)*num_processes))
  # params_dict = {str(i): params[i].tolist() for i in range(len(params))}
  # with open('bayes-opt-presimulated-unif-100.yml', 'w') as handle:
  #   yaml.dump(params_dict, handle)

  # mc_rep = 100
  # time_horizon = 1000
  # number_of_plots = 10
  # times = np.linspace(0, 100, 100)
  # for rep in range(10):
  #   zeta_opt = bayes_optimize_zeta(rep, mc_rep=mc_rep, T=time_horizon)
  #   vals = [stepwise.stepwise_linear_epsilon(zeta_opt, 10, t) for t in times]
  #   plt.plot(times, vals)
  #   plt.savefig('bayes-opt-presimulated-unif-T-{}-mcrep-{}.png'.format(time_horizon, mc_rep))

  results = yaml.load(open("bayes-opt-presimulated-unif-100.yml"))
  times = np.linspace(0, 100, 100)
  for param in results.values():
    vals = [stepwise.stepwise_linear_epsilon(param, 10, t) for t in times]
    plt.plot(times, vals)
  plt.savefig('bayes-opt-presimulated-unif-T-{}-mcrep-{}.png'.format(100, 100))



