import scipy
from scipy.optimize import minimize, basinhopping
from scipy.special import expit
def expit_epsilon_decay(T, t, zeta):
  return zeta[0] * expit(zeta[2]*(T - t - zeta[1]))


def fun(zeta, T=50):
  norm = 0
  for i in range(T):
    norm += abs(expit_epsilon_decay(T, i, zeta)-1.0/(1.0+i))
  return norm
T=50
bnds = ((0.05, 2.0), (1.0, T ), (0.01, 2.5))
res = minimize(fun, (2, T, 0.06), method='L-BFGS-B', bounds=bnds)
ranges = (slice(0.05, 2.0, 0.1), slice(1.0, T, 1), slice(0.01, 2.5, 0.1))
res2 = scipy.optimize.brute(fun, ranges=ranges)
bounds = [(0.00, 100.0), (1.0, T ), (0.00, 10)]
res3 = scipy.optimize.differential_evolution(fun, bounds=bounds, tol=0.000001)
print(res3.fun)
minimizer_kwargs = {"method":"L-BFGS-B"}
x0 = [ 5.00983034, 80.08297668,  0.09330464]
ret = basinhopping(fun, x0, minimizer_kwargs=minimizer_kwargs,
                   niter=1000)
print(ret)
plt.plot(expit_epsilon_decay(T=T, t=np.arange(T), zeta=res['x']))
#plt.plot(expit_epsilon_decay(T=T, t=np.arange(T), zeta=res2))
plt.plot(expit_epsilon_decay(T=T, t=np.arange(T), zeta=res3['x']))

plt.plot(expit_epsilon_decay(T=T, t=np.arange(T), 
                             zeta=np.array([10.17777864, 87.55187593,  0.09409438])))
plt.plot(1.0/(1+np.arange(T)))
plt.xlim(0,10)
print(fun(zeta=np.array([ 1.90980867, 49.94980088,  0.4])))