from policies.policies import fitted_q
from Glucose import Glucose
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
import numpy as np

def fitted_q(env, gamma, regressor, number_of_value_iterations):
  X = env.get_state_history_as_array()
  X, Xp1 = X[:-1, :], X[1:, :]
  target = np.hstack(env.R)

  # Fit one-step q fn
  reg = regressor()
  reg.fit(X, target)

  # Fit longer-horizon q fns
  for k in range(number_of_value_iterations):
    q_max, _ = maximize_q_function_at_block(reg.predict, Xp1, env)
    target += gamma * q_max
    reg.fit(X, target)

  # Maximize final q iterate to get next action
  _, list_of_optimal_actions = maximize_q_function_at_block(reg.predict, X, env)

  # Last entry of list gives optimal action at current state
  optimal_action = list_of_optimal_actions[-1]
  
  return optimal_action


def mdp_epsilon_policy(optimal_action, tuning_function, tuning_function_parameter, time_horizon,
                      time):
  epsilon = tuning_function(time_horizon, time, tuning_function_parameter)
  if np.random.rand() < epsilon:
    action = np.random.choice(2)
  else:
    action = optimal_action
  return action


def condition_dist_next_state(X, Y):
  '''
  Y: dependent variables matrix
  '''
  n = X.shape[0]
  reg = LinearRegression().fit(X, Y)
  
  # Maximum Likelihood Estimate of Error Covariance matrix (slide 61 on http://users.stat.umn.edu/~helwig/notes/mvlr-Notes.pdf)
  Sigma_hat = np.matmul(Y.T, Y-reg.predict(X))/n
  
  beta_hat = np.append(reg.intercept_, reg.coef_)
  R_sq = reg.score(X, Y)
  
  return {'Sigma_hat': Sigma_hat, 'beta_hat': beta_hat, 'R_sq': R_sq, 'reg': reg}

  
  
def mdp_glucose_each_rollout(tuning_function_parameter, mdp_epsilon_policy, time_horizon, current_time,
                      x_initial, tuning_function, env, Glucose_Approx, gamma=0.9, number_of_value_iterations=10):
  X, Sp1 = env.get_state_transitions_as_x_y_pair()
  multi_regr = condition_dist_next_state(X, Sp1)
  beta_hat = multi_regr['beta_hat']
  Sigma_hat = multi_regr['Sigma_hat']
  sim_env = Glucose_Approx(time_horizon, x_initial, beta_hat, Sigma_hat)
  r = 0
  for time in range(time_horizon):
    optimal_action = fitted_q(env, gamma, RandomForestRegressor, number_of_value_iterations=number_of_value_iterations)
    action = mdp_epsilon_policy(optimal_action, tuning_function, tuning_function_parameter, time_horizon, time)
    _, reward, done = sim_env.step(action)
    r += reward
  return r
    
    
    
    
    
    
    
    
  
