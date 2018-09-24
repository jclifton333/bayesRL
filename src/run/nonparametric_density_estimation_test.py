"""
Module for nonparametric density estimation for context density
and reward given context density.

Code is sourced from https://docs.pymc.io/notebooks/dependent_density_regression.html
"""

import pymc3 as pm
import numpy as np
import pandas as pd
from matplotlib import animation as ani, pyplot as plt
import seaborn as sns
from theano import shared, tensor as tt

def norm_cdf(z):
    return 0.5 * (1 + tt.erf(z / np.sqrt(2)))

def stick_breaking(v):
    return v * tt.concatenate([tt.ones_like(v[:, :1]),
                               tt.extra_ops.cumprod(1 - v, axis=1)[:, :-1]],
                              axis=1)

def standardize(x):
    return (x - x.mean()) / x.std()

SEED = 972915
DATA_URI = 'http://www.stat.cmu.edu/~larry/all-of-nonpar/=data/lidar.dat'
df = (pd.read_csv(DATA_URI, sep=' *', engine='python')
        .assign(std_range=lambda df: standardize(df.range),
                std_logratio=lambda df: standardize(df.logratio)))

N, _ = df.shape
K = 20

std_range = df.std_range.values[:, np.newaxis]
std_logratio = df.std_logratio.values[:, np.newaxis]

x_lidar = shared(std_range, broadcastable=(False, True))

with pm.Model() as model:
    alpha = pm.Normal('alpha', 0., 5., shape=K)
    beta = pm.Normal('beta', 0., 5., shape=K)
    v = norm_cdf(alpha + beta * x_lidar)
    w = pm.Deterministic('w', stick_breaking(v))

with model:
    gamma = pm.Normal('gamma', 0., 10., shape=K)
    delta = pm.Normal('delta', 0., 10., shape=K)
    mu = pm.Deterministic('mu', gamma + delta * x_lidar)

with model:
    tau = pm.Gamma('tau', 1., 1., shape=K)
    obs = pm.NormalMixture('obs', w, mu, tau=tau, observed=std_logratio)

SAMPLES = 10
BURN = 2

with model:
    step = pm.Metropolis()
    trace = pm.sample(SAMPLES, step, chains=1, tune=BURN, random_seed=SEED)


fig, ax = plt.subplots(figsize=(8, 6))

ax.bar(np.arange(K) + 1,
       trace['w'].mean(axis=0).max(axis=0))

ax.set_xlim(1 - 0.5, K + 0.5)
ax.set_xticks(np.arange(0, K, 2) + 1)
ax.set_xlabel('Mixture component')

ax.set_ylabel('Largest posterior expected\nmixture weight')
plt.show()

# plot credible interval
PP_SAMPLES = 5000

lidar_pp_x = np.linspace(std_range.min() - 0.05, std_range.max() + 0.05, 100)
x_lidar.set_value(lidar_pp_x[:, np.newaxis])

with model:
    pp_trace = pm.sample_posterior_predictive(trace, PP_SAMPLES, random_seed=SEED)

fig, ax = plt.subplots(figsize=(8, 6))
blue, *_ = sns.color_palette()
ax.scatter(df.std_range, df.std_logratio,
           c=blue, zorder=10,
           label=None);

low, high = np.percentile(pp_trace['obs'], [2.5, 97.5], axis=0)
ax.fill_between(lidar_pp_x, low, high,
                color='k', alpha=0.35, zorder=5,
                label='95% posterior credible interval');

ax.plot(lidar_pp_x, pp_trace['obs'].mean(axis=0),
        c='k', zorder=6,
        label='Posterior expected value');

ax.set_xticklabels([]);
ax.set_xlabel('Standardized range');

ax.set_yticklabels([]);
ax.set_ylabel('Standardized log ratio');

ax.legend(loc=1);
ax.set_title('LIDAR Data');
plt.show()

pass