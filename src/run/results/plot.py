#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Oct  2 12:17:49 2018

Functions for plots of results of normalMAB
list_of_reward_mus=[0.3, 0.6], 
action=1: regret=0
action=0: regret=0.3
"""
import yaml
import os

def expit_epsilon_decay(T, t, zeta):
  return zeta[0] * expit(zeta[2]*(T - t - zeta[1]))

os.chdir("/Users/lili/Documents/labproject2017/aistat/bayesRL/src/run/results")
names = ["normalmab-postsample-True-eps-decay_181002_044328", "normalmab-postsample-False-eps05_181002_090501", "normalmab-postsample-False-eps10_181002_090744",
         "normalmab-postsample-True-frequentist-ts-tuned_181002_030543", "normalmab-10-frequentist-ts_181001_223917", 
         "normalmab-postsample-True-ucb-tune-posterior-sample_181002_094331", "normalmab-postsample-False-ucb_181002_095950"]
         
names = ["normalmab-postsample-True-std-0.1-eps-decay_181002_135253", "normalmab-postsample-False-std-0.1-eps05_181002_140624", "normalmab-postsample-False-std-0.1-eps10_181002_140433",
         "normalmab-postsample-True-std-0.1-frequentist-ts-tuned_181002_122545", "normalmab-postsample-True-std-0.1-frequentist-ts_181002_123446",
         "normalmab-postsample-True-std-0.1-ucb-tune-posterior-sample_181002_180302", "normalmab-postsample-False-std-0.1-ucb_181002_180547"]
tuneds= [0,3,5]
cum_regret_ave_over_rep_allmethods = np.zeros((len(names), 50))
zeta_seq = []*3
mean_regrets = np.zeros((len(names)))
se_regrets = np.zeros((len(names)))
for i in range(len(names)):
  print(names[i])
  with open(names[i]+".yml", 'r') as f:
    doc = yaml.load(f)
    regret = (1-np.array(doc['actions']))*0.3
    cum_regret = np.cumsum(regret, axis=1)
    cum_regret_ave_over_rep = np.mean(cum_regret, axis=0)
    cum_regret_ave_over_rep_allmethods[i, ] = cum_regret_ave_over_rep
    mean_regrets[i] = doc['mean_regret']
    se_regrets[i] = doc['std_regret']/np.sqrt(96)

print(np.round(mean_regrets, 2))
print(np.round(se_regrets, 2))

#NAMES = ["Tuned "+r"$\epsilon-greedy$", r"$\epsilon-greedy$ (0.05)", r"$\epsilon-greedy$ (0.1)", "Tuned Thompson Sampling", 'Thompson Sampling', "Tuned UCB", "UCB"]
##index_list = [3, 0, 1, 2, 6]
#fig, ax = plt.subplots(figsize=(8, 6))
#ax.set_title('Mean Cumulative Regrets ('+r'$\sigma=1$)'.format('seaborn'), color='C0')
#ax.set_xlabel('Timesteps')
#ax.set_ylabel('Mean cumulativ regrets')
##ax.set_ylim([0,210]) 
#
#for i in range(len(NAMES)):  
#  ax.plot(cum_regret_ave_over_rep_allmethods[i, ], 'C'+str(i), label=NAMES[i])
##ax.plot(nSteps['num_steps'], 'C'+str(1), 
##        label='Obj val = '+str(round(test,5))+', Lmax='+str(Lmax)+', length='+str(res.shape[0])+', ksi='+str(ksi)+', nCutoff='+str(100/thresh)+', K_V='+str(K))        
#ax.legend()
#fig.savefig('sigma1.eps', format='eps', dpi=1000)


NAMES = [["Tuned "+r"$\epsilon$-greedy", r"$\epsilon$-greedy ("+r"$\epsilon=0.05$)", r"$\epsilon$-greedy ("+r"$\epsilon=0.1$)"], ["Tuned Thompson Sampling", 'Thompson Sampling'], 
          ["Tuned UCB", "UCB ("+r'$\alpha=0.05$)']]
k = [0, 3, 5]
#index_list = [3, 0, 1, 2, 6]
fig, ax = plt.subplots(1, 3, figsize=(15, 5), sharey=True)
fig.suptitle('Mean Cumulative Regrets ('+r'$\sigma=0.1$)'.format('seaborn'), color='C0')
#fig.supxlabel('Timesteps')
#ax.set_title('Mean Cumulative Regrets ('+r'$\sigma=1$)'.format('seaborn'), color='C0')
#ax[0].set_xlabel('Timesteps')
ax[0].set_ylabel('Mean cumulativ regrets')
#ax.set_ylim([0,210]) 
for j in range(3):
  for i in range(len(NAMES[j])):  
#    ax[j].set_ylim([0,5.8]) 
    ax[j].plot(cum_regret_ave_over_rep_allmethods[k[j]+i, ], 'C'+str(i), label=NAMES[j][i])
#ax.plot(nSteps['num_steps'], 'C'+str(1), 
#        label='Obj val = '+str(round(test,5))+', Lmax='+str(Lmax)+', length='+str(res.shape[0])+', ksi='+str(ksi)+', nCutoff='+str(100/thresh)+', K_V='+str(K))        
  ax[j].legend(loc=2)
#  ax[j].set_xlabel('Timesteps')
fig.text(0.5, 0.04, 'Timesteps', ha='center')
fig.savefig('sigma01.eps', format='eps', dpi=1000)

'''
zeta's
'''
fig, ax = plt.subplots(figsize=(8,6))
ax.set_xlabel('Timesteps')
#ax.set_title(r'$1-\alpha$ Vaules in Tuned UCB'.format('seaborn'), color='C0')
#ax.set_ylabel(r'$1-\alpha$')
#ax.set_title('Shrinkage Values in Tuned TS'.format('seaborn'), color='C0')
#ax.set_ylabel('Shrinkage value')
ax.set_title(r'$\epsilon$ Values in Tuned $\epsilon$-greedy'.format('seaborn'), color='C0')
ax.set_ylabel(r'$\epsilon$')
#ax.set_ylim([0,210]) 
reps = [30, 90]
times = [15, 45]
#reps = [5, 25, 85]
#times=[15]
#times = [15, 45]
for j in reps:
  for k in times:
    if j==90 and k==45:
      break
#    ts = [0.5+0.5*expit_epsilon_decay(T, i, doc['zeta_sequences'][j][k]) for i in range(T)]
    ts = [expit_epsilon_decay(T, i, doc['zeta_sequences'][j][k]) for i in range(T)]
    ax.plot(ts, label='('+r'$\hat{\mu}_1,\hat{\sigma}_1^2$'+') = ('+str(round(doc['estimated_means'][j][k][0],2)) + ", "+str(round(doc['estimated_vars'][j][k][0],2)) +"), ("+\
            r'$\hat{\mu}_2,\hat{\sigma}_2^2$'+') = ('+str(round(doc['estimated_means'][j][k][1],2)) + ", "+str(round(doc['estimated_vars'][j][k][1],2)) +")")
#ax.legend(bbox_to_anchor=(0, 1.02, 1., .102), loc=3, ncol=1, mode="expand", borderaxespad=1)
#ax.legend(loc=6)
ax.legend(loc=[0.2, 0.5])
fig.savefig('eps2.eps', format='eps', dpi=1000)

os.chdir('/Users/lili/Documents/labproject2017/aistat/bayesRL/src/run')
with open("normalcb-eps_181003_015127.yml", 'r') as f:
  doc = yaml.load(f)
plt.plot(doc['beta_hat_list'][0][0])