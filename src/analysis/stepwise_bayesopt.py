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
import src.policies.rollout as rollout
import src.policies.tuned_bandit_policies as policies
import src.analysis.stepwise as stepwise
import matplotlib.pyplot as plt
import yaml


def bayes_optimize_zeta(seed, mc_rep=1000, T=50):
  np.random.seed(seed)
  
  env = NormalCB(list_of_reward_betas=[[-10, 0.4, 0.4, -0.4], [-9.8, 0.6, 0.6, -0.4]], context_mean=np.array([0.0, 0.0, 0.0]),
            context_var=np.array([[1.0,0,0], [0,1.,0], [0, 0, 1.]]), list_of_reward_vars=[1, 1])
  # env = NormalMAB(list_of_reward_mus=[0, 1], list_of_reward_vars=[1, 140])
#  env = NormalMAB(list_of_reward_mus=[0.3, 0.6], list_of_reward_vars=[1**2, 1**2])
  # X = env.X
  # estimated_context_mean = np.mean(X, axis=0)
  # estimated_context_variance = np.cov(X, rowvar=False)
  # estimated_context_bounds = (np.min(X), np.max(X))
  # sim_env = NormalUniformCB(list_of_reward_betas=env.list_of_reward_betas, list_of_reward_vars=env.list_of_reward_vars,
  #                           context_bounds=env.context_bounds)
  sim_env = NormalCB(list_of_reward_betas=[[-10, 0.4, 0.4, -0.4], [-9.8, 0.6, 0.6, -0.4]], context_mean=np.array([0.0, 0.0, 0.0]),
            context_var=np.array([[1.0,0,0], [0,1.,0], [0, 0, 1.]]), list_of_reward_vars=[1, 1])
#  sim_env = NormalMAB(list_of_reward_mus=env.list_of_reward_mus, list_of_reward_vars=env.list_of_reward_vars)
  pre_simulated_data = sim_env.generate_mc_samples(mc_rep, T)
  rollout_function_kwargs = {'pre_simulated_data': pre_simulated_data}
  ans  =rollout.normal_cb_rollout_with_fixed_simulations(None, policies.linear_cb_epsilon_greedy_policy, T,
          lambda a, b, c: 0.05, sim_env, **rollout_function_kwargs)

  print(ans)
  # def objective(zeta0, zeta1, zeta2, zeta3, zeta4, zeta5, zeta6, zeta7, zeta8, zeta9):
  #   zeta = np.array([zeta0, zeta1, zeta2, zeta3, zeta4, zeta5, zeta6, zeta7, zeta8, zeta9])
  def objective(zeta0, zeta1, zeta2):
    zeta = np.array([zeta0, zeta1, zeta2])
#    return rollout.mab_rollout_with_fixed_simulations(zeta, policies.mab_frequentist_ts_policy, T,
#                                                      policies.expit_epsilon_decay, sim_env, **rollout_function_kwargs)
    return rollout.normal_cb_rollout_with_fixed_simulations(zeta, policies.linear_cb_epsilon_greedy_policy, T,
                                                     policies.expit_epsilon_decay, sim_env, **rollout_function_kwargs)

  # bounds = {'zeta{}'.format(i): (0.0, 1.0) for i in range(10)}
  # explore_ = {'zeta{}'.format(i): [0.0] for i in range(10)}
  explore_ = {'zeta0': [1.0, 1.0, 1.0], 'zeta1': [25.0, 49.0, 1.0], 'zeta2': [0.1, 2.5, 2.0]}
  bounds = {'zeta0': (0.8, 2.0), 'zeta1': (1.0, 49.0), 'zeta2': (0.01, 2.5)}
  bo = BayesianOptimization(objective, bounds)
  bo.explore(explore_)
  bo.maximize(init_points=10, n_iter=20, alpha=1e-4)
  best_param = bo.res['max']['max_params']
  best_param = np.array([best_param['zeta{}'.format(i)] for i in range(3)])
  return best_param


def plot_epsilon_sequences(fname):
  # results = yaml.load(open(fname))
  results = {0: np.array([0.8, 62.95, 1.0])}
  times = np.linspace(0, 50, 100)
  for param in results.values():
   vals = [policies.expit_epsilon_decay(50, t, param) for t in times]
   plt.plot(times, vals)
   plt.savefig("bayes-opt-presimulated-normal-cb-0.8max-1000.png")


def max_observed_epsilons(fname):
  results = yaml.load(open(fname))
  max_epsilons = []
  for episode_results in results['zeta_sequences']:
    observed_epsilon_sequence = []
    for t, param in enumerate(episode_results):
      eps = policies.stepwise_linear_epsilon(100, t, param)
      observed_epsilon_sequence.append(eps)
    max_epsilons.append(np.max(observed_epsilon_sequence))
  plt.hist(max_epsilons)
  plt.show()


def plot_epsilons_and_estimated_means(fname):
  results = yaml.load(open(fname))
  sample_mean_diffs = []
  epsilons = []
  arm0_sample_vars = []

  # epsilons_diff_1 = []
  arm1_sample_vars_1 = []
  # epsilons_diff_0 = []
  arm1_sample_vars_0 = []
  arm0_sample_vars_1 = []
  arm0_sample_vars_0 = []

  for episode_zetas, episode_means, episode_vars in \
    zip(results['zeta_sequences'], results['estimated_means'], results['estimated_vars']):
    param_0 = episode_zetas[80]
    means_0 = episode_means[80]
    var_arm1 = episode_vars[15][1]
    var_arm0 = episode_vars[15][0]
    diff = means_0[0] - means_0[1]
    eps = policies.stepwise_linear_epsilon(100, 80, param_0)
    epsilons.append(eps)
    sample_mean_diffs.append(diff)

    if diff > 0:
      arm1_sample_vars_0.append(var_arm1)
      arm0_sample_vars_0.append(var_arm0)
    elif diff < 0:
      arm1_sample_vars_1.append(var_arm1)
      arm0_sample_vars_1.append(var_arm0)

  print('var0 given diff > 0: {}'.format(np.mean(arm0_sample_vars_0)))
  print('var1 given diff > 0: {}'.format(np.mean(arm1_sample_vars_0)))
  print('var0 given diff < 0: {}'.format(np.mean(arm0_sample_vars_1)))
  print('var1 given diff < 0: {}'.format(np.mean(arm1_sample_vars_1)))

    # Look where arm 1 is incorrectly estimtaed best
  #   if diff < 0:
  #     sample_mean_diffs.append(diff)
  #     eps = policies.stepwise_linear_epsilon(100, 0, param_0)
  #     arm1_sample_vars_1.append(var_arm1)
  #     epsilons_diff_1.append(eps)
  #   elif diff > 0:
  #     sample_mean_diffs.append(diff)
  #     eps = policies.stepwise_linear_epsilon(100, 0, param_0)
  #     arm1_sample_vars_0.append(var_arm1)
  #     epsilons_diff_0.append(eps)

  plt.scatter(sample_mean_diffs, epsilons)
  # plt.scatter(arm1_sample_vars_1, epsilons_diff_1)
  # plt.scatter(arm1_sample_vars_0, epsilons_diff_0)
  plt.show()


def plot_approximate_epsilon_sequences_from_sims(fname):
  results = yaml.load(open(fname))
  times = np.linspace(1, 100, 100)

  # for episode_results in results['zeta_sequences']:
  #   observed_epsilon_sequence = []
  #   for t, param in enumerate(episode_results):
  #     eps = policies.stepwise_linear_epsilon(100, t, param)
  #     observed_epsilon_sequence.append(eps)
  #   plt.plot(times, observed_epsilon_sequence)
  # plt.savefig("observed-epsilons-zeta-normal-mab-1000.png")

  time_slices = [10, 50, 80]
  for time_slice in time_slices:
    params = [episode_results[time_slice] for episode_results in results['zeta_sequences']]
    plt.figure()
    for param in params:
      vals = [policies.stepwise_linear_epsilon(100, t, param) for t in times]
      plt.plot(times, vals)
    # plt.savefig("estimated-zeta-normal-mab-1000-timeslice-{}.png".format(time_slice))
    plt.show()
  return


if __name__ == "__main__":
  # num_processes = 4
  # num_replicates = num_processes
  # pool = mp.Pool(num_processes)
  # # params = []
  # # for batch in range(int(num_replicates / num_processes)):
  # #   params += pool.map(bayes_optimize_zeta, range(batch*num_processes, (batch+1)*num_processes))
  # params = pool.map(bayes_optimize_zeta, range(num_processes))
  # params_dict = {str(i): params[i].tolist() for i in range(len(params))}
  # with open('bayes-opt-presimulated-normal-mab-low-var-1000.yml', 'w') as handle:
  #   yaml.dump(params_dict, handle)

  p = bayes_optimize_zeta(0, T=50, mc_rep=1000)
  # print(p)

  # plot_epsilon_sequences("bayes-opt-presimulated-normal-mab-low-var-1000.yml")
  # plot_approximate_epsilon_sequences_from_sims("normalmab-10-eps-decay-posterior-sample_180930_054446.yml")
  # plot_approximate_epsilon_sequences_from_sims("normalmab-10-ts-decay-posterior-sample_181001_064428.yml")
  # max_observed_epsilons("normalmab-10-eps-decay_180929_164335.yml")
  # plot_epsilons_and_estimated_means("normalmab-10-eps-decay_180929_212908.yml")



