#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 25 08:43:02 2018

@author: lili
"""
import yaml
from mpl_toolkits import mplot3d


#name = "NormalMAB"
name = "BernoulliMAB"
with open('/Users/lili/Downloads/{}-opt-zeta-vs-model.yml'.format(name), 'r') as f:
    doc = yaml.load(f)
x = doc['p0']    
y = doc['p1'] 

z = doc['best_score']
ax = plt.axes(projection='3d')
ax.scatter3D(x, y, z);
ax.set_xlabel('p0')
ax.set_ylabel('p1')
ax.set_zlabel('best_score');
ax.set_title(name)


z = doc['best_zeta2']
ax = plt.axes(projection='3d')
ax.scatter3D(x, y, z);
ax.set_xlabel('p0')
ax.set_ylabel('p1')
ax.set_zlabel('best_zeta2');
ax.set_title(name)