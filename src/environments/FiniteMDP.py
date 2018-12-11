import numpy as np

class FiniteMDP(object):
    def __init__(self, nA=2, nS=3, time_horizon=50, gamma=0.9, transitionMatrices=None, rewardMatrices=None):#, hardmax, epsilon=0.1, fixUpTo=None):
        '''
        Parameters
        ----------
        nA: the number of actions
        nS: the number of states
        gamma : discount factor
        epsilon : for epsilon - greedy
        transitionMatrices: an array of nA by nS by nS 
        rewardMatrices: an array of nA by nS by nS 
        '''
        self.current_state = None
        self.time_horizon = time_horizon
        self.nS = nS
        self.nA = nA
        self.NUM_STATE = nS
        self.NUM_ACTION = nA
        self.gamma = gamma
        self.t = 0
        self.posterior_alpha = np.ones((nA, nS, nS)) #Initialize posterior, actually it's also the prior
        
        #Construct transition and reward arrays, set random seed to keep the same true environments for each replicate
        if rewardMatrices is None:
            np.random.seed(12345) 
            self.rewardMatrices = np.random.uniform(low=0, high=10, size=(nA, nS, nS))
        else:
            self.rewardMatrices = rewardMatrices
        if transitionMatrices is None:
            np.random.seed(12345)
            self.transitionMatrices = np.random.dirichlet(alpha=np.random.poisson(5, nS), size=(nA, nS))
        else:
            self.transitionMatrices = transitionMatrices 
            
        #Create transition dictionary of form {s_0 : {a_0: [( P(s_0 -> s_0), s_0, reward), ( P(s_0 -> s_1), s_1, reward), ...], a_1:...}, s_1:{...}, ...}
        self.mdpDict= {}
        for s in range(self.nS):
            self.mdpDict[s] = {} 
            for a in range(self.nA):
                self.mdpDict[s][a] = [(self.transitionMatrices[a, s, sp1], sp1, self.rewardMatrices[a, s, sp1]) for sp1 in range(self.nS)]
  

    def reset(self):
        '''
        reset to the state 0
        :return:
        '''
        self.current_state = 0
        return self.current_state

    def step(self, a):
        self.t += 1
#        pdb.set_trace()
        prob = self.transitionMatrices[a, self.current_state, :]
#        print(a, self.current_state)
        s_prev = self.current_state
        self.current_state = np.random.choice(range(self.nS), 1, p=prob)[0]
        reward = self.rewardMatrices[a, s_prev, self.current_state]
        
        self.posterior_alpha[a, s_prev, self.current_state] += 1.0
#        transition_prob = self.posterior_alpha/np.sum(self.posterior_alpha, axis=2).reshape(16,4,1)
#        transitionMatrices = self.convert_to_transitionMatrices(transition_prob)
        return self.current_state, reward #, done
      
    def posterior_mean_model(self):
        # Use posterior mean as transition probability, to get the estimated model for doing policy iteration
        # return 
        transitionMatrices = self.posterior_alpha/np.sum(self.posterior_alpha, axis=2).reshape(self.nA, self.nS, 1)
        return transitionMatrices 




