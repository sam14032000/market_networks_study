# -*- coding: utf-8 -*-
"""
Created on Thu Nov 19 15:24:06 2020

@author: saksh

Basic funcs used in main.py file

Additional libraries to be installed:
    1) python-lovain
    2) networkx
"""

import pandas as pd

import numpy as np
np.random.seed(1337) #random state used throughout the notebook for reproducibility
from math import log

import matplotlib.pyplot as plt
import community as louvain

plt.style.use('classic')

def louvain_community(G_cache = G_cache, resolution = 1):
  comm_cache = {}
  comm_cache_mod = {}
  for timestamp in list(G_cache):
      G = G_cache[timestamp]
      comm_iter = louvain.best_partition(G, weight = 'weight', random_state = 1337, resolution = resolution)
      comm_count = max(comm_iter.values())+1
      comm_cache_mod_list = [[] for i in range(comm_count)]
      for node in list(comm_iter):
        i = comm_iter[node]
        comm_cache_mod_list[i].append(node)
    
      comm_cache[timestamp] = comm_iter
      comm_cache_mod[timestamp] = comm_cache_mod_list

  return comm_cache, comm_cache_mod

def variation_of_information(X, Y):
  n = float(sum([len(x) for x in X]))
  sigma = 0.0
  for x in X:
    p = len(x) / n
    for y in Y:
      q = len(y) / n
      r = len(set(x) & set(y)) / n
      if r > 0.0:
        sigma += r * (log(r / p, 2) + log(r / q, 2))
  return abs(sigma)/log(n)

def pd_fill_diagonal(corr_df, value=0): 
  arr = corr_df.values
  np.fill_diagonal(arr, value)
  corr_df = pd.DataFrame(arr, index=corr_df.index, columns=corr_df.columns)
  return corr_df