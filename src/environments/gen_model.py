'''
Functions to construct a generative model (MAB) whose greedy policy is not the true optimal policy:
number of arms = 2;
NormalMAB();
assume mu0 < mu1, so that the true optimal action is arm 1;
'''
import scipy.stats
import pdb


def gen_normal_mab(N=100, mu0=0, mu1=1, var0=1, greedy_subopt_prob=0.3):
  '''
  greedy_subopt_prob < 0.5;
  mu0 < mu1;
  return the parameters of generative model
  '''
  #To find the variate for which the probability is given, let's say the 
  #value which needed to provide a 98% probability, you'd use the 
  #PPF Percent Point Function
  Z = scipy.stats.norm.ppf(1 - greedy_subopt_prob, 0, 1)  
  var1 = (mu1 - mu0)**2*N/Z**2 - var0
  return {'N':100, 'mu0':mu0, 'mu1':mu1, 'var0':var0, 'var1': var1, 'greedy_subopt_prob':greedy_subopt_prob}


'''
Functions to construct a generative model (MAB) whose greedy policy is not the true optimal policy:
number of arms = 2;
NormalUniformCB();
'''
def gen_uniform_cb(N=100, p=10, greedy_subopt_prob=0.45):
  '''
  p: the dimension of context;
  greedy_subopt_prob < 0.5;
  assume sum(\beta0) > sum(\beta1), so that the true optimal action is arm 0;
  context iid ~ Unif(0,1) 
  return the parameters of generative model
  '''
  #To find the variate for which the probability is given, let's say the 
  #value which needed to provide a 98% probability, you'd use the 
  #PPF Percent Point Function
  beta0 = np.ones(p)+1
  beta1 = np.ones(p)
  var0 = 1
  X0 = np.random.rand(N, p)
  X1 = np.random.rand(N, p)
  var0_tilde = var0 * sum(sum(np.linalg.inv(np.matmul(X0.T, X0))))
  Z = scipy.stats.norm.ppf(1 - greedy_subopt_prob, 0, 1)  
  var1_tilde = (sum(beta0) - sum(beta1))**2/Z**2 - var0_tilde
  var1 = var1_tilde/sum(sum(np.linalg.inv(np.matmul(X1.T, X1))))
  return {'N':100, 'beta0':beta0, 'beta1':beta1, 'var0':var0, 'var1': var1, 'greedy_subopt_prob':greedy_subopt_prob}


pdb.set_trace()
