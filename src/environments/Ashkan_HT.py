#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 15 11:11:02 2020
#### do the test for Ashkan model
@author: lwu9
"""

from policy_ashkan import *
import copy
import pickle

def reward_t_T(gamma, env, env_tilde, T, epsilon, K, regr):
    ## to calculate cumulative rewards from t to T under a model sampled from the sampling dbn of the estimated model:
    ## assign the current states to the env_tilde
    env_tilde.R = copy.deepcopy(env.R); env_tilde.S = copy.deepcopy(env.S)
    env_tilde.X = copy.deepcopy(env.X); env_tilde.A = copy.deepcopy(env.A)
    env_tilde.current_s = copy.deepcopy(env.current_s)
    env_tilde.prev_u_t = copy.deepcopy(env.prev_u_t)
    env_tilde.t = copy.deepcopy(env.t)
    cum_rewards = 0
    for t in np.arange(env_tilde.t, T):
        epsilon_t = epsilon
        if regr=="RF":
            est_opt_actions = fitted_q_iteration_mHealth(env_tilde, gamma, K)
        elif regr=="linear":
            est_opt_actions = linear_rule(env_tilde, gamma, K)
        actions_random = np.random.binomial(1,0.5,env.nPatients)
        random_prob = np.random.rand(env.nPatients) 
        est_opt_actions[random_prob < epsilon_t] = actions_random[random_prob < epsilon_t]
        done,_,rewards = env_tilde.step(est_opt_actions)
        cum_rewards += rewards
        if done:
            break
    return cum_rewards

def ht(gamma0, gamma1, env, T, epsilon, K, regr, B, N, alpha):
    ## gamma0, gamma1: the baseline and proposed schedules
    model_boot = GlucoseTransitionModel()
    model_boot.bootstrap_and_fit_conditional_densities(env.X)
    cum_reward0 = []; cum_reward1 = []
    for b in range(B):
        env_tilde0 = Glucose(nPatients=env.nPatients, sigma_eps = model_boot.sigma_eps, 
                             mu0 = model_boot.mu0, sigma_B0_X0_Y0 = model_boot.sigma_B0_X0_Y0, 
                             mu_B0_X0 = model_boot.mu_B0_X0, prob_L_given_trts = model_boot.prob_L_given_trts,
                             tau_trts=model_boot.tau_trts, death_prob_coef=model_boot.death_prob_coef)
        env_tilde1 = Glucose(nPatients=env.nPatients, sigma_eps = model_boot.sigma_eps, 
                             mu0 = model_boot.mu0, sigma_B0_X0_Y0 = model_boot.sigma_B0_X0_Y0, 
                             mu_B0_X0 = model_boot.mu_B0_X0, prob_L_given_trts = model_boot.prob_L_given_trts,
                             tau_trts=model_boot.tau_trts, death_prob_coef=model_boot.death_prob_coef)
        cum_reward0 = np.append(cum_reward0, reward_t_T(gamma0, env, env_tilde0, T, epsilon, K, regr))
        cum_reward1 = np.append(cum_reward1, reward_t_T(gamma1, env, env_tilde1, T, epsilon, K, regr))
    rej_prop = np.mean((cum_reward0 - cum_reward1)<0) ## P(delta < 0)
    rej_indicator = (rej_prop < alpha)
    
    ## calculate true value under the true model
    value0 = 0.0; value1 = 0.0
    for i in range(N):
        env_true0 = Glucose(nPatients=env.nPatients)
        env_true1 = Glucose(nPatients=env.nPatients)
        value0 += reward_t_T(gamma0, env, env_true0, T, epsilon, K, regr)
        value1 += reward_t_T(gamma1, env, env_true1, T, epsilon, K, regr)
    value0 /= N; value1 /= N 
    rej_true = ((value0 - value1)<0) ## indicator
    return({"value0":value0, "value1":value1, "cum_reward0_list":cum_reward0, 
            "cum_reward1_list":cum_reward1, "rej_indicator":rej_indicator, "rej_true":rej_true})
    
def ht_whole_traj(rep, gamma0, gamma1, env, T, epsilon, K, regr, B=100, N=200, alpha=0.05):
    np.random.seed(rep)
    env.reset()
    mean_rewards = 0
    actions = np.random.binomial(1,0.5,env.nPatients)
    done, _, rewards = env.step(actions)
    gammas_used = []
    rej_indicator = False
    ht_list = []
    for t in range(T):
        if done:
            break
        else:
            print("time step {}".format(t))
            if not rej_indicator:
                ht_results = ht(gamma0, gamma1, env, T, epsilon, K, regr, B, N, alpha)
                ht_list = np.append(ht_list, ht_results)
                rej_indicator = ht_results["rej_indicator"]     
                if rej_indicator:
                    gamma = gamma1
                else:
                    ## if not reject, follow the baseline gamma0 at t
                    gamma = gamma0
            else:
                ## if reject, follow the proposed from t and after, no need do the test anymore
                gamma = gamma1
            gammas_used = np.append(gammas_used, gamma)
            ## eps-greedy policy:
            epsilon_t = epsilon
            if regr=="RF":
                est_opt_actions = fitted_q_iteration_mHealth(env, gamma, K)
            elif regr=="linear":
                est_opt_actions = linear_rule(env, gamma, K)
            actions_random = np.random.binomial(1,0.5,env.nPatients)
            random_prob = np.random.rand(env.nPatients) 
            est_opt_actions[random_prob < epsilon_t] = actions_random[random_prob < epsilon_t]
            done,_,rewards = env.step(est_opt_actions)
            mean_rewards += (rewards - mean_rewards) / (t+1)
    return({"mean_rewards":mean_rewards, "gammas_used":gammas_used, "ht_list":ht_list})
                
                
if __name__ == "__main__":
    numCores = 96
    Rep=numCores
    nPatients = 100
    env = Glucose(nPatients=nPatients)
    policy = "eps-greedy"
    epsilon = 0.05; T=10; K=1
    regr = "RF" 
    gamma0=0.1; gamma1=0.9
#    results = ht_whole_traj(0, gamma0, gamma1, env, T, epsilon, K, regr, B=10, N=50, alpha=0.05)
    pl = Pool(numCores)
    results = pl.map(partial(ht_whole_traj, gamma0=gamma0, gamma1=gamma1, env=env, T = T,
                          epsilon=epsilon, K=K, regr=regr, B=100, N=200, alpha=0.05), range(Rep))
    pl.close()
    ## write results into txt
    with open("Ashkan_HT_"+"gamma0_"+str(gamma0)+"_gamma1_"+str(gamma1)+"_regr_"+str(regr)+"_K_"+str(K)+"_T_"+str(T)+"_nPatients_"+str(env.nPatients)+".txt", "wb") as fp:
        pickle.dump(results, fp)

    t1_error=[]; t2_error=[]; total_t1_error = 0.0; total_num_ht = 0
    ht_results = [i['ht_list'] for i in results]
    for d_each_rep in ht_results:
        for d_each_t in d_each_rep:
            total_num_ht += 1
            if d_each_t['rej_true']:
                ## H1 is true
                t2_error.append(1-d_each_t['rej_indicator'])
            else:
                ## H0 is true
                t1_error.append(d_each_t['rej_indicator']) 
                total_t1_error += d_each_t['rej_indicator']
    print("T1 error: {}, T2 error: {}, total T1 error: {}".format(np.mean(t1_error), 
          np.mean(t2_error), total_t1_error/total_num_ht))

