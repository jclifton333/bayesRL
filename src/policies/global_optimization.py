import pdb
import numpy as np
from bayes_opt import BayesianOptimization
from scipy.optimize import basinhopping


def bayesopt(rollout_function, policy, tuning_function, zeta_prev, time_horizon, env, mc_replicates,
             rollout_function_kwargs, bounds, explore_, positive_zeta=False):

  # Assuming 10 params!
  # def objective(zeta0, zeta1, zeta2, zeta3, zeta4, zeta5, zeta6, zeta7, zeta8, zeta9):
  # zeta = np.array([zeta0, zeta1, zeta2, zeta3, zeta4, zeta5, zeta6, zeta7, zeta8, zeta9])

  def objective(zeta0, zeta1, zeta2):
    zeta = np.array([zeta0, zeta1, zeta2])
    value = rollout_function(zeta, policy, time_horizon, tuning_function, env, **rollout_function_kwargs)
    return value

  # bounds = {'zeta{}'.format(i): (lower_bound, upper_bound) for i in range(10)}
  explore_.update({'zeta{}'.format(i): [zeta_prev[i]] for i in range(len(zeta_prev))})
  bo = BayesianOptimization(objective, bounds, verbose=False)
  bo.explore(explore_)
  # bo.maximize(init_points=10, n_iter=10)
  bo.maximize(init_points=1, n_iter=1)
  best_param = bo.res['max']['max_params']
  best_param = np.array([best_param['zeta{}'.format(i)] for i in range(len(bounds))])
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
  objective_values = []
  best_val = objective(zeta_prev)
  best_zeta = zeta_prev
  best_truncation_val = tuning_function(time_horizon, current_time, zeta_prev)
  # for zeta0 in np.linspace(zeta0_bounds[0], zeta0_bounds[1], points_per_grid_dimension):
  for zeta1 in np.linspace(zeta1_bounds[0], zeta1_bounds[1], points_per_grid_dimension):
    print(zeta0, zeta1)
    zeta_rand = np.array([kappa, zeta0, zeta1])
    val = objective(zeta_rand)
    objective_values.append(val)
    truncation_val = tuning_function(time_horizon, current_time, zeta_rand)
    truncation_values.append(truncation_val)
    if val > best_val:
      best_truncation_val = truncation_val
      best_val = val
      best_zeta = zeta_rand
  return best_zeta


def grid_search(rollout_function, policy, tuning_function, zeta_prev, time_horizon, current_time,
                estimated_context_mean, estimated_context_variance, env, nPatients, points_per_grid_dimension,
                monte_carlo_reps):
  # Optimization parameters
  # zeta0_bounds = (-2, -0.05)
  zeta0 = -5
  zeta1_bounds = (0, 2)
  kappa = 0.3

  # # Generate context sequences
  # context_sequences = []
  # for rep in range(monte_carlo_reps):
  #   context_sequence = []
  #   for t in range(time_horizon - current_time):
  #     context_sequence_at_time_t = []
  #     for j in range(nPatients):
  #       context = np.random.multivariate_normal(estimated_context_mean, estimated_context_variance)
  #       context_sequence_at_time_t.append(context)
  #     context_sequence.append(context_sequence_at_time_t)
  #   context_sequences.append(context_sequence)

  def objective(zeta):
    return rollout_function(zeta, policy, time_horizon, current_time,
                            estimated_context_mean, tuning_function, estimated_context_variance, env,
                            nPatients, monte_carlo_reps)

  truncation_values = []
  best_val = objective(zeta_prev)
  best_zeta = zeta_prev
  print(zeta_prev)
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


def stochastic_approximation_step(rollout_function, policy, tuning_function, zeta_k, lambda_k, env, J, monte_carlo_reps,
                                  estimated_context_mean, estimated_context_variance, nPatients, T, t):
  def objective(zeta):
    return rollout_function(zeta, policy, T, t,
                            estimated_context_mean, tuning_function, estimated_context_variance, env,
                            nPatients, monte_carlo_reps)

  z = np.random.normal(loc=0, size=J)
  v_of_zeta_k = objective(zeta_k)
  v_of_zeta_k_plus_z = objective(zeta_k + z)
  zeta_k_plus_one = zeta_k + lambda_k * z * (v_of_zeta_k_plus_z - v_of_zeta_k)
  return zeta_k_plus_one


def linear_cb_stochastic_approximation(rollout_function, policy, tuning_function, tuning_function_parameter, T, t,
                                       estimated_context_mean, estimated_context_variance, env, nPatients,
                                       points_per_grid_dimension, monte_carlo_reps):

  MAX_ITER = 20
  TOL = 1e-4
  it = 0
  diff = float('inf')

  J = tuning_function_parameter.size
  zeta = tuning_function_parameter

  while it < MAX_ITER and diff > TOL:
    # print(it)
    lambda_ = 0.01 / (it + 1)
    new_zeta = stochastic_approximation_step(rollout_function, policy, tuning_function, zeta, lambda_, env, J,
                                             monte_carlo_reps, estimated_context_mean, estimated_context_variance,
                                             nPatients, T, t)
    new_zeta = np.array([np.min((np.max((z, 0.0)), 0.05)) for z in new_zeta])  # Project zeta onto space of reasonable
                                                                               # values
    diff = np.linalg.norm(new_zeta - zeta) / np.linalg.norm(zeta)
    zeta = new_zeta
    # print(zeta)
    it += 1

  return zeta
