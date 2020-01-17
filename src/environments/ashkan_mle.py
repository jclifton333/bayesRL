#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 30 16:21:25 2019

@author: lwu9
"""
import numpy as np
from scipy.optimize import minimize
import pdb
from sklearn.ensemble import RandomForestRegressor

class GlucoseTransitionModel():

  def __init__(self):
    """
    :param test: if True, this is just for testing - only draw one sample in MCMC
    """
    # Covariates and subsequent glucoses; hang on to these after model is fit for plotting
    self.X_ = None
    self.y_ = None
    
  def fit_conditional_densities(self, X, weight=None):
    if weight is None:
      weight = [np.ones(X[j].shape[0]) for j in range(len(X))]
    self.weight = weight
    self.X_ = X
    self.nPatients = len(X)
    x0 = np.ones(15)/2.0 # initialize the parameters
    x0[8:11] = np.zeros(3)
    bnds = ((0.0001,None), (0, 20), (0.0001,None), (None, None), (0,1), (0,1), (0,1), (0,1), (-20,20),(-2,2),(-2,2),
            (0.0001,0.9999), (0.0001,0.9999), (0.0001,0.9999), (0.0001,0.9999))
#    bnds = ((0.0001,2), (7.7,7.7), (1,1), (0,0), (0.14,0.14), (0.2,0.2), (0.02, 0.02), (0.14,0.14), (-10,-10),(0.08,0.08),(0.5,0.5),
#            (0.2,0.2), (0.2,0.2), (0.2,0.2), (0.35,0.35))
    res = minimize(self.joint_loglikeli, x0, method='SLSQP', bounds=bnds)
    self.sigma_eps, self.mu0, self.sigma_B0_X0_Y0, self.mu_B0_X0 = res.x[:4]
    self.tau_trts = res.x[4:8]
    self.death_prob_coef = res.x[8:11]
    self.prob_L_given_trts = res.x[-4:]
    self.res = res
    
  def loglikeli_L_given_trt(self, X_each_patient, prob_L_given_trts, weight):
    ## loglikelihood of L_t|trt_{t-1} for each patient
    ## X_each_patient: 2-d array of states for one patient with dim time steps by 7, the last col is actions
    ind_a = np.where(X_each_patient[:,6] == 1)[0][:4]
    if len(ind_a) == 0:
        loglikeli = 0
    else:
        loglikeli = sum(weight[ind_a+1]*(X_each_patient[ind_a+1, 1] * np.log(prob_L_given_trts[:len(ind_a)]) +  \
                    (1-X_each_patient[ind_a+1, 1]) * np.log((1-prob_L_given_trts)[:len(ind_a)])))
    return loglikeli
  
  def loglikeli_death(self, X_each_patient, death_prob_coef, weight):
    y_each_patient = X_each_patient[:, 2]
    x = death_prob_coef[0]+death_prob_coef[1]*y_each_patient[:-1]**2*(y_each_patient[:-1]>7)+\
        death_prob_coef[2]*X_each_patient[:-1, 0]
    x = np.clip(x, -100, 100) ## clip the range of x in case to generate nan values
    prob_c = self.exp_helper(x)
    prob_c[prob_c==0] = 10**(-9); prob_c[prob_c==1] = 1-10**(-9); ## in case log(0) gives nan
    loglikeli = sum(weight[1:]*(X_each_patient[1:, 5] * np.log(prob_c) + \
                    (1-X_each_patient[1:, 5]) * np.log(1-prob_c)))
    return loglikeli
  
  def loglikeli_B_X(self, X_each_patient, sigma_eps, weight):
    mean_B = X_each_patient[:-1, 3]/np.sqrt(1+sigma_eps**2)
    mean_X = X_each_patient[:-1, 4]/np.sqrt(1+sigma_eps**2)
    loglikeli = sum(weight[1:]*(-np.log(self.sigma_helper(sigma_eps))/2 - (X_each_patient[1:, 3]-mean_B)**2/(2*self.sigma_helper(sigma_eps)) + \
                    -np.log(self.sigma_helper(sigma_eps))/2 - (X_each_patient[1:, 4]-mean_X)**2/(2*self.sigma_helper(sigma_eps))))
    return loglikeli

  def A1c_multiply_part_each_patient(self, X_each_patient, tau_trts, mu0):
    y_each_patient = X_each_patient[:, 2]
    K_t = (y_each_patient[:-1] > 7) * (X_each_patient[:-1, 0] < 4) * (X_each_patient[:-1, 6] != 0) * \
            (X_each_patient[1:, 1] != 1)
    tau_t = np.zeros(len(K_t))
    trt = X_each_patient[1:, 0][X_each_patient[1:, 0] != 0].astype(int) ## treatments from time 1 to (t-1)
    tau_t[X_each_patient[1:, 0] != 0] = tau_trts[trt-1]
#    print( K_t * tau_t)
    cum_multiply = np.cumprod(1 - K_t * tau_t)
    return np.append(mu0, cum_multiply * mu0)

  def loglikeli_A1c(self, X_each_patient, sigma_eps, tau_trts, mu0, weight):
    y_each_patient = X_each_patient[:, 2]
    cum_multiply = self.A1c_multiply_part_each_patient(X_each_patient, tau_trts, mu0)
    cond_mean = (y_each_patient[:-1] - cum_multiply[:-1]) / np.sqrt(1+sigma_eps**2) + cum_multiply[1:]
    loglikeli = sum(weight[1:]*(-np.log(self.sigma_helper(sigma_eps))/2 - \
                    (y_each_patient[1:] - cond_mean)**2 / (2*self.sigma_helper(sigma_eps))))
    return loglikeli

  def joint_loglikeli(self,parameters):
    ## taus: paramenters of treatment effect
    ## the function need to be miniized
    sigma_eps, mu0, sigma_B0_X0_Y0, mu_B0_X0 = parameters[:4]
    tau_trts = parameters[4:8]
    death_prob_coef = parameters[8:11]
    prob_L_given_trts = parameters[-4:]
    log_likelihood = 0
    #pdb.set_trace()
    for i in range(self.nPatients):
      w_each_patient = self.weight[i]
      X_each_patient = self.X_[i]
      y_each_patient = X_each_patient[:, 2]
      log_likelihood += self.loglikeli_L_given_trt(X_each_patient, prob_L_given_trts, w_each_patient) + \
                        self.loglikeli_death(X_each_patient, death_prob_coef, w_each_patient) + \
                        self.loglikeli_A1c(X_each_patient, sigma_eps, tau_trts, mu0, w_each_patient) + \
                        self.loglikeli_B_X(X_each_patient, sigma_eps, w_each_patient) + \
                        w_each_patient[0]*(-np.log(sigma_B0_X0_Y0)-(X_each_patient[0,3]-mu_B0_X0)**2/(2*sigma_B0_X0_Y0**2) + \
                        -np.log(sigma_B0_X0_Y0)-(X_each_patient[0,4]-mu_B0_X0)**2/(2*sigma_B0_X0_Y0**2) + \
                        -np.log(sigma_B0_X0_Y0)-(y_each_patient[0]-mu0)**2/(2*sigma_B0_X0_Y0**2))
      #pdb.set_trace()
      if ~np.isfinite(log_likelihood):
        print(i, log_likelihood)
        pdb.set_trace()
    return -log_likelihood

  def bootstrap_and_fit_conditional_densities(self, X):
    weight = [np.random.exponential(size=X[j].shape[0]) for j in range(len(X))]
    self.fit_conditional_densities(X, weight)
        
  def exp_helper(self, x):
    return np.exp(x)/(1.0+np.exp(x))
  
  def sigma_helper(self, sigma):
    return float(sigma**2/(1.0+sigma**2))
 
  def np_A1c(self, X, weight, regressor=RandomForestRegressor):
      ## output: esimated E(A1c_{t+1}|N_t, L_t, A1c_t, BP_t, Weight_t, A_t) and esitmated conditional variance. 
      ## The conditioning variables are corresponding to X_each_patient[:,[0,1,2,3,4,6]] ## the 5th is death indicator, no need to be included
      x = X[0][:-1,:]
      y = X[0][1:,2]
      sample_weight = weight[0][:-1]
      for i in np.arange(1, len(X)):
          x = np.vstack((x, X[i][:-1,:]))
          y = np.append(y, X[i][1:,2])
          sample_weight = np.append(sample_weight, weight[i][:-1])
      regression = regressor(n_estimators=50, min_samples_leaf=1)
      regression.fit(x, y, sample_weight=sample_weight)
      y_pred = regression.predict(x)
      sd = np.std(y_pred-y)
      self.sd = sd
      self.regression = regression
  
  def joint_loglikeli_except_A1c(self,parameters):
    ## taus: paramenters of treatment effect
    ## the function need to be miniized
    sigma_eps, mu0, sigma_B0_X0_Y0, mu_B0_X0 = parameters[:4]
    death_prob_coef = parameters[8:11]
    prob_L_given_trts = parameters[-4:]
    log_likelihood = 0
    #pdb.set_trace()
    for i in range(self.nPatients):
      w_each_patient = self.weight[i]
      X_each_patient = self.X_[i]
      y_each_patient = X_each_patient[:, 2]
      log_likelihood += self.loglikeli_L_given_trt(X_each_patient, prob_L_given_trts, w_each_patient) + \
                        self.loglikeli_death(X_each_patient, death_prob_coef, w_each_patient) + \
                        self.loglikeli_B_X(X_each_patient, sigma_eps, w_each_patient) + \
                        w_each_patient[0]*(-np.log(sigma_B0_X0_Y0)-(X_each_patient[0,3]-mu_B0_X0)**2/(2*sigma_B0_X0_Y0**2) + \
                        -np.log(sigma_B0_X0_Y0)-(X_each_patient[0,4]-mu_B0_X0)**2/(2*sigma_B0_X0_Y0**2) + \
                        -np.log(sigma_B0_X0_Y0)-(y_each_patient[0]-mu0)**2/(2*sigma_B0_X0_Y0**2))
      #pdb.set_trace()
      if ~np.isfinite(log_likelihood):
        print(i, log_likelihood)
        pdb.set_trace()
    return -log_likelihood

  def fit_conditional_densities_except_A1c(self, X, weight=None):
    if weight is None:
      weight = [np.ones(X[j].shape[0]) for j in range(len(X))]
    self.weight = weight
    self.X_ = X
    self.nPatients = len(X)
    x0 = np.ones(11)/2.0 # initialize the parameters
    x0[4:7] = np.zeros(3)
    bnds = ((0.0001,None), (0, 20), (0.0001,None), (None, None), (-20,20),(-2,2),(-2,2),
            (0.0001,0.9999), (0.0001,0.9999), (0.0001,0.9999), (0.0001,0.9999))
#    bnds = ((0.0001,2), (7.7,7.7), (1,1), (0,0), (0.14,0.14), (0.2,0.2), (0.02, 0.02), (0.14,0.14), (-10,-10),(0.08,0.08),(0.5,0.5),
#            (0.2,0.2), (0.2,0.2), (0.2,0.2), (0.35,0.35))
    res = minimize(self.joint_loglikeli, x0, method='SLSQP', bounds=bnds)
    self.sigma_eps, self.mu0, self.sigma_B0_X0_Y0, self.mu_B0_X0 = res.x[:4]
    self.death_prob_coef = res.x[4:7]
    self.prob_L_given_trts = res.x[-4:]
    self.res = res 

  def bootstrap_and_fit_conditional_densities_except_A1c_np(self, X):
    weight = [np.random.exponential(size=X[j].shape[0]) for j in range(len(X))]
    self.fit_conditional_densities_except_A1c(X, weight)    
    self.np_A1c(X, weight)
      
      
      
      
      
      
      
      
      
      
      




