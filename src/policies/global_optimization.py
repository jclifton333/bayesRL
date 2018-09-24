import pdb
import numpy as np
# from bayes_opt import BayesianOptimization
from scipy.optimize import basinhopping


def bayesopt(rollout_function, policy, tuning_function, zeta_prev, linear_model_results, time_horizon, current_time,
             estimated_context_mean, estimated_context_variance):
  def objective(kappa, zeta0, zeta1):
    return rollout_function(np.array([kappa, zeta0, zeta1]), policy, linear_model_results, time_horizon, current_time,
                            estimated_context_mean, tuning_function, estimated_context_variance)

  # ToDo: Fix this shit!
  if len(zeta_prev) == 3:
    bo = BayesianOptimization(objective, {'kappa': (0.05, 0.3), 'zeta0': (-2, -0.05), 'zeta1': (1, 2)})
    bo.explore({'kappa': [zeta_prev[0]], 'zeta0': [zeta_prev[1]], 'zeta1': [zeta_prev[2]]})
    bo.maximize(init_points=5, n_iter=10)
    best_param = bo.res['max']['max_params']
    best_param = np.array([best_param['kappa'], best_param['zeta0'], best_param['zeta1']])
  elif len(zeta_prev == 2):
    bo = BayesianOptimization(objective, {'zeta0': (-2, -0.05), 'zeta1': (1, 2)})
    bo.explore({'zeta0': [zeta_prev[0]], 'zeta1': [zeta_prev[1]]})
    bo.maximize(init_points=5, n_iter=10)
    best_param = bo.res['max']['max_params']
    best_param = np.array([best_param['zeta0'], best_param['zeta1']])
  return best_param


def random_search(rollout_function, policy, tuning_function, zeta_prev, linear_model_results, time_horizon,
                  current_time,
                  estimated_context_mean, estimated_context_variance, env):
  # Generate context sequences
  NUM_REP = 1000
  context_sequences = []
  for rep in range(NUM_REP):
    context_sequence = []
    for t in range(time_horizon - current_time):
      context = env.draw_context()
      context_sequence.append(context)
    context_sequences.append(context_sequence)

  def objective(zeta):
    return rollout_function(zeta, policy, linear_model_results, time_horizon, current_time,
                            estimated_context_mean, tuning_function, estimated_context_variance, env,
                            context_sequences)

  truncation_values = []
  # ToDo: Fix this shit!
  if len(zeta_prev) == 3:
    bounds = [(0.05, 0.3), (-2, -0.05), (1, 2)]
  elif len(zeta_prev) == 2:
    bounds = [(-2, -0.05), (1, 2)]
  random_zetas = np.array([np.array([np.random.uniform(low=low_, high=high_) for low_, high_ in bounds])
                           for _ in range(10)])

  best_val = objective(zeta_prev)
  best_zeta = zeta_prev
  best_truncation_val = tuning_function(time_horizon, current_time, zeta_prev)
  for zeta_rand in random_zetas:
    val = objective(zeta_rand)
    truncation_val = tuning_function(time_horizon, current_time, zeta_rand)
    truncation_values.append(truncation_val)
    if val > best_val:
      best_truncation_val = truncation_val
      best_val = val
      best_zeta = zeta_rand
  pdb.set_trace()
  return best_zeta


def mab_grid_search(rollout_function, policy, tuning_function, zeta_prev, time_horizon,
                    current_time, env, nPatients, points_per_grid_dimension, monte_carlo_reps):
  # Optimization parameters
  # zeta0_bounds = (-2, -0.05)
  zeta0 = -5
  zeta1_bounds = (0, 2)
  kappa = 0.3

  def objective(zeta):
    return rollout_function(zeta, policy, time_horizon, current_time, tuning_function, env, nPatients, monte_carlo_reps)

  truncation_values = []
  best_val = objective(zeta_prev)
  best_zeta = zeta_prev
  best_truncation_val = tuning_function(time_horizon, current_time, zeta_prev)
  # for zeta0 in np.linspace(zeta0_bounds[0], zeta0_bounds[1], points_per_grid_dimension):
  for zeta1 in np.linspace(zeta1_bounds[0], zeta1_bounds[1], points_per_grid_dimension):
    print(zeta0, zeta1)
    zeta_rand = np.array([kappa, zeta0, zeta1])
    val = objective(zeta_rand)
    truncation_val = tuning_function(time_horizon, current_time, zeta_rand)
    truncation_values.append(truncation_val)
    if val > best_val:
      best_truncation_val = truncation_val
      best_val = val
      best_zeta = zeta_rand
  return best_zeta


def grid_search(rollout_function, policy, tuning_function, zeta_prev, linear_model_results, time_horizon, current_time,
                estimated_context_mean, estimated_context_variance, env, nPatients, points_per_grid_dimension,
                monte_carlo_reps):
  # Optimization parameters
  # zeta0_bounds = (-2, -0.05)
  zeta0 = -5
  zeta1_bounds = (0, 2)
  kappa = 0.3

  # Generate context sequences
  context_sequences = []
  for rep in range(monte_carlo_reps):
    context_sequence = []
    for t in range(time_horizon - current_time):
      context_sequence_at_time_t = []
      for j in range(nPatients):
        context = np.random.multivariate_normal(estimated_context_mean, estimated_context_variance)
        context_sequence_at_time_t.append(context)
      context_sequence.append(context_sequence_at_time_t)
    context_sequences.append(context_sequence)

  def objective(zeta):
    return rollout_function(zeta, policy, linear_model_results, time_horizon, current_time,
                            estimated_context_mean, tuning_function, estimated_context_variance, env,
                            nPatients, context_sequences)

  truncation_values = []
  best_val = objective(zeta_prev)
  best_zeta = zeta_prev
  best_truncation_val = tuning_function(time_horizon, current_time, zeta_prev)
  # for zeta0 in np.linspace(zeta0_bounds[0], zeta0_bounds[1], points_per_grid_dimension):
  for zeta1 in np.linspace(zeta1_bounds[0], zeta1_bounds[1], points_per_grid_dimension):
    print(zeta0, zeta1)
    zeta_rand = np.array([kappa, zeta0, zeta1])
    val = objective(zeta_rand)
    truncation_val = tuning_function(time_horizon, current_time, zeta_rand)
    truncation_values.append(truncation_val)
    if val > best_val:
      best_truncation_val = truncation_val
      best_val = val
      best_zeta = zeta_rand
  return best_zeta


def basinhop(rollout_function, policy, tuning_function, zeta_prev, linear_model_results, time_horizon, current_time,
             estimated_context_mean, estimated_context_variance):
  def objective(zeta):
    return rollout_function(zeta, policy, linear_model_results, time_horizon, current_time,
                            estimated_context_mean, tuning_function, estimated_context_variance)

  # ToDo: Fix this shit!
  if len(zeta_prev) == 3:
    bounds = [(0.05, 0.3), (-2, -0.05), (1, 2)]
  elif len(zeta_prev == 2):
    bounds = [(-2, -0.05), (1, 2)]
  minimizer_kwargs = dict(method="L-BFGS-B", bounds=bounds)
  res = basinhopping(objective, x0=zeta_prev, minimizer_kwargs=minimizer_kwargs)
  return res.x
