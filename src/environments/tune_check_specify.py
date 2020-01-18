#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 10 15:11:11 2020

@author: lwu9
"""

from policy_ashkan import *
import pickle
import time 
import os

def out_loop(k, gamma, gamma_type, K, env, T, policy, 
                                epsilon, fixed_decay, regr, rollouts,
                                tune):
    rs = 0.0
    Rep = len(tune)
    for i in range(Rep):
        gammas = tune[i]['gammas_used']
        r = rewards_policy(i+k*Rep, gamma=gamma, gamma_type=gamma_type, K=K, env=env, T = T, policy=policy, 
                                epsilon=epsilon, fixed_decay=fixed_decay, regr=regr, rollouts=rollouts,
                                gammas=gammas)
        rs += r['mean_rewards']
    return(rs/Rep)


def run_tuned_check_specify(numCores, Rep, K, env, T, policy, epsilon, fixed_decay, regr, rollouts):
    gamma_type = "tune_np"; gamma=None
    pl = Pool(numCores)
    tune = pl.map(partial(rewards_policy, gamma=gamma, gamma_type=gamma_type, K=K, env=env, T = T,
                        policy=policy, epsilon=epsilon, fixed_decay=fixed_decay, regr=regr, rollouts=rollouts, 
                        gammas=np.arange(0,1.1,0.1)), range(Rep))
    pl.close()
    # write results into txt
    with open("Ashkan_"+gamma_type+str(regr)+"_K_"+str(K)+"_T_"+str(T)+"_nPatients_"+str(env.nPatients)+".txt", "wb") as fp:
        pickle.dump(tune, fp)
    results = [i["mean_rewards"] for i in tune]
    gammas_used = np.vstack([i["gammas_used"] for i in tune])
    print(tune)
    print("mean rewards of "+policy+"-"+str(epsilon)+"-fixedDecay-"+str(fixed_decay)+"-gammaType-"+gamma_type+\
          "-gamma-"+str(gamma) + "-K-"+str(K)+": " + str(round(np.mean(results),3))+", with sd: " +\
          str(round(np.std(results),3)))
    print(np.mean(np.vstack([i["gammas_used"] for i in tune]), axis=0))
    print(np.std(np.vstack([i["gammas_used"] for i in tune]), axis=0))


    gamma_type = "not_tune"
    for gamma in np.arange(0,1.1,0.1):
        pl = Pool(numCores)
#                cs = pl.map(wrap, range(Rep))
        cs = pl.map(partial(rewards_policy, gamma=gamma, gamma_type=gamma_type, K=K, env=env, T = T,
                            policy=policy, epsilon=epsilon, fixed_decay=fixed_decay, regr=regr, rollouts=rollouts, 
                            gammas=np.arange(0,1.1,0.1)),range(Rep))
        pl.close()
        results = [i["mean_rewards"]  for i in cs]
        print("mean rewards of "+policy+"-"+str(epsilon)+"-fixedDecay-"+str(fixed_decay)+"-gammaType-"+gamma_type+\
              "-gamma-"+str(gamma) + "-K-"+str(K)+": " + str(round(np.mean(results),3))+", with sd: " +\
              str(round(np.std(results),3)))
#        
#        
##    with open("Ashkan_tune"+str(regr)+"_K_"+str(K)+"_T_"+str(T)+".txt", "rb") as fp:
##        tune = pickle.load(fp)    
#    ## check with prespecify    
#    gamma_type = "prespecify"; gamma=None
#    pl = Pool(numCores)
#    results_spe = pl.map(partial(out_loop, gamma=gamma, gamma_type=gamma_type, K=K, env=env, T = T,
#                        policy=policy, epsilon=epsilon, fixed_decay=fixed_decay, regr=regr, rollouts=rollouts, tune=tune), range(Rep))
#    pl.close()
#    with open("Ashkan_"+gamma_type+str(regr)+"_K_"+str(K)+"_T_"+str(T)+"_nPatients_"+str(env.nPatients)+".txt", "wb") as fp:
#        pickle.dump(results_spe, fp)
#    print("prespecify_out_loop:")
#    print(np.mean(results_spe), np.std(results_spe))
#
##    print("mean rewards of "+policy+"-"+str(epsilon)+"-fixedDecay-"+str(fixed_decay)+"-gammaType-"+gamma_type+\
##          "-gamma-"+str(gamma) + "-K-"+str(K)+": " + str(round(np.mean(results_spe),3))+", with sd: " +\
##          str(round(np.std(results_spe),3)))
#    ## write results into txt




if __name__ == "__main__":
    numCores = 86
    Rep=numCores
    nPatients = 50
    env = Glucose(nPatients=nPatients)
    policy = "eps-greedy"
    epsilon = 0.05
    fixed_decay = 0
    rollouts=50
    T=10
    K=1
    regr = "linear" 
    run_tuned_check_specify(numCores, Rep, K, env, T, policy, epsilon, fixed_decay, regr, rollouts)
