#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 28 19:03:16 2019
Tune gamma
@author: lwu9
"""
from policy_ashkan import *
import pickle
import time 
import os

if __name__ == "__main__":
    numCores = 96
    Rep=numCores
    nPatients = 50
    env = Glucose(nPatients=nPatients)
    policy = "eps-greedy"
    epsilon = 0.05
    fixed_decay = 0
    rollouts=50
    T=10
    K=1
    regr = "RF" #  "linear" # 
    for gamma_type in ["tune","not_tune"]:
        if gamma_type != "tune":
            for gamma in np.arange(0,1.1,0.1):
#        for gamma in np.arange(0,1.1,0.2):
#                def wrap(rep):
##                    print(rep)
#                    return(rewards_policy(rep, gamma=gamma, gamma_type=gamma_type, K=K, env=env, T = T,
#                                    policy=policy, epsilon=epsilon, fixed_decay=fixed_decay, regr=regr, rollouts=rollouts, 
#                                    gammas=np.arange(0,1,0.1)))
                pl = Pool(numCores)
#                cs = pl.map(wrap, range(Rep))
                cs = pl.map(partial(rewards_policy, gamma=gamma, gamma_type=gamma_type, K=K, env=env, T = T,
                                    policy=policy, epsilon=epsilon, fixed_decay=fixed_decay, regr=regr, rollouts=rollouts, 
                                    gammas=np.arange(0,1.1,0.1)), np.arange(Rep,2*Rep))
                pl.close()
                results = [i["mean_rewards"]  for i in cs]
                print("mean rewards of "+policy+"-"+str(epsilon)+"-fixedDecay-"+str(fixed_decay)+"-gammaType-"+gamma_type+\
                      "-gamma-"+str(gamma) + "-K-"+str(K)+": " + str(round(np.mean(results),3))+", with sd: " +\
                      str(round(np.std(results),3)))
        else:
            start = time.process_time()
            gamma = None
            pl = Pool(numCores)
            cs = pl.map(partial(rewards_policy, gamma=gamma, gamma_type=gamma_type, K=K, env=env, T = T,
                                policy=policy, epsilon=epsilon, fixed_decay=fixed_decay, regr=regr, rollouts=rollouts, 
                                gammas=np.arange(0,1.1,0.1)), np.arange(Rep,2*Rep))
            pl.close()
            results = [i["mean_rewards"] for i in cs]
            gammas_used = np.vstack([i["gammas_used"] for i in cs])
            print("mean rewards of "+policy+"-"+str(epsilon)+"-fixedDecay-"+str(fixed_decay)+"-gammaType-"+gamma_type+\
                  "-gamma-"+str(gamma) + "-K-"+str(K)+": " + str(round(np.mean(results),3))+", with sd: " +\
                  str(round(np.std(results),3)))
            print(np.mean(np.vstack([i["gammas_used"] for i in cs]), axis=0))
            print(np.std(np.vstack([i["gammas_used"] for i in cs]), axis=0))
            elapsed = (time.process_time() - start)
            print("Time used:",elapsed)
            ## write results into txt
            with open("Ashkan2_"+str(regr)+"_K_"+str(K)+"_T_"+str(T)+".txt", "wb") as fp:
                pickle.dump(cs, fp)
#            ## read results from txt
#            os.chdir("/Users/lwu9/Documents/lab_proj/PE")
#            with open("Ashkan2_"+str(regr)+"_K_"+str(K)+"_T_"+str(T)+".txt", "rb") as fp:
#                b2 = pickle.load(fp)

#    for policy in ["eps-greedy", "random"]:
#        if policy=="eps-greedy":
#            for epsilon in [0, 0.05, 0.1]:
#                pl = Pool(96)
#                cs = pl.map(partial(rewards, policy=policy, epsilon=epsilon, 
#                                    nPatients = 100, T = 50), range(Rep))
#                pl.close()
#                results = [i for i in cs]
#                print("mean rewards of "+policy+"-"+str(epsilon)+": "+str(np.mean(results))+", with sd: "+\
#                      str(np.std(results)))
#        else:
#            epsilon = None
#            pl = Pool(96)
#            cs = pl.map(partial(rewards, policy=policy, epsilon=epsilon, 
#                                nPatients = 100, T = 50), range(Rep))
#            pl.close()        
#            results = [i for i in cs]
#            print("mean rewards of "+policy+"-"+str(epsilon)+": "+str(np.mean(results))+", with sd: "+\
#                  str(np.std(results)))


## to check whether rewards are the same using the chosen gammas with the simulations
## but they are different for the tuned, the same for the not-tuned
gamma_type = "prespecify"; policy = "eps-greedy"; epsilon = 0.05; fixed_decay = 0;T=5; K=1; regr = "RF" ; 
Rep=len(b) ## b is read from txt
env = Glucose(nPatients=nPatients); rs = []
for i in range(Rep):
    gammas = b[i]['gammas_used']
#    gammas = [0.1]*T
    r = rewards_policy(i, gamma=gamma, gamma_type=gamma_type, K=K, env=env, T = T, policy=policy, 
                        epsilon=epsilon, fixed_decay=fixed_decay, regr=regr, rollouts=rollouts,
                        gammas=gammas)
    rs = np.append(rs, r['mean_rewards'])
print(sum(rs)/len(rs), np.std(rs))


gamma_type = "tune"; policy = "eps-greedy"; epsilon = 0.05; fixed_decay = 0;T=5; K=1; rollouts=0; regr = "linear" ;
env = Glucose(nPatients=20);  
rr= rewards_policy(i, gamma=gamma, gamma_type=gamma_type, K=K, env=env, T = T, policy=policy, 
                        epsilon=epsilon, fixed_decay=fixed_decay, regr=regr, rollouts=rollouts,
                        gammas=np.arange(0,1.1,0.1))
print(rr)

gamma_type = "prespecify"; policy = "eps-greedy"; epsilon = 0.05; fixed_decay = 0;T=5; K=1; rollouts=10; regr = "linear" ;
env = Glucose(nPatients=20);  
rr= rewards_policy(i, gamma=gamma, gamma_type=gamma_type, K=K, env=env, T = T, policy=policy, 
                        epsilon=epsilon, fixed_decay=fixed_decay, regr=regr, rollouts=rollouts,
                        gammas=rr['gammas_used'])
print(rr)








