import scipy
from scipy.optimize import minimize
def fun(zeta):
  norm = 0
  for i in range(75):
    norm += abs(expit_epsilon_decay(75, i, zeta)-1.0/(1.0+i))
  return norm
bnds = ((0.05, 2.0), (1.0, 75.0 ), (0.01, 2.5))
res = minimize(fun, (2, 75, 0.06), method='L-BFGS-B', bounds=bnds)
ranges = (slice(0.05, 2.0, 0.1), slice(1.0, 75.0, 1), slice(0.01, 2.5, 0.1))
res2 = scipy.optimize.brute(fun, ranges=ranges)
bounds = [(0.05, 2.0), (1.0, 75.0 ), (0.01, 2.5)]
res3 = scipy.optimize.differential_evolution(fun, bounds=bounds)