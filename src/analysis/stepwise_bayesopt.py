import pdb
import os
import sys
this_dir = os.path.dirname(os.path.abspath(__file__))
project_dir = os.path.join(this_dir, '..', '..')
sys.path.append(project_dir)
import numpy as np
from bayes_opt import BayesianOptimization
import multiprocessing as mp
from src.environments.Bandit import NormalMAB, BernoulliMAB
import src.analysis.stepwise as stepwise
import matplotlib.pyplot as plt
import yaml


def bayes_optimize_zeta(seed, mc_rep=100, T=100):
  np.random.seed(seed)

  env = NormalMAB()

  def objective(zeta0, zeta1, zeta2, zeta3, zeta4, zeta5, zeta6, zeta7, zeta8, zeta9):
    zeta = np.array([zeta0, zeta1, zeta2, zeta3, zeta4, zeta5, zeta6, zeta7, zeta8, zeta9])
    return stepwise.rollout_stepwise_linear_mab(zeta, env, J=10, mc_rep=mc_rep, T=T)

  bounds = {'zeta{}'.format(i): (0.0, 0.2) for i in range(10)}
  bo = BayesianOptimization(objective, bounds)
  bo.maximize(init_points=1, n_iter=1)
  best_param = bo.res['max']['max_params']
  best_param = np.array([best_param['zeta{}'.format(i)] for i in range(10)])
  return best_param


if __name__ == "__main__":
  num_processes = mp.cpu_count()
  pool = mp.Pool(num_processes)
  params = pool.map(bayes_optimize_zeta, range(num_processes))
  params_dict = {i: params[i] for i in range(params)}

  with open('bayes-opt-100.yml', 'w') as handle:
    yaml.dump(params_dict, handle)

 #  mc_rep = 100
 #  number_of_plots = 10
 #  times = np.linspace(0, 100, 100)
 #  for rep in range(10):
 #    zeta_opt = bayes_optimize_zeta(mc_rep=mc_rep)
 #    vals = [stepwise.stepwise_linear_epsilon(zeta_opt, 10, t) for t in times]
 #    plt.plot(times, vals)
 #    plt.savefig('bayes-opt-{}-mc-rep-{}.png'.format(mc_rep, rep))


