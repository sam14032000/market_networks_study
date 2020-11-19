# -*- coding: utf-8 -*-
"""
Created on Thu Nov 19 15:27:23 2020

@author: saksh

Main execution file for market_networks paper; Reccommended to use market_networks(phase_3).ipynb for a more thorough analysis
Adjust the file path in import_csv according to position of file
"""

#init
import pandas as pd

import numpy as np
np.random.seed(1337) #random state used throughout the notebook for reproducibility
from math import log

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import matplotlib.cm as cm
import seaborn as sns
from datetime import datetime
import networkx as nx
import community as louvain
from collections import Counter
import random
from preprocess_funcs import louvain_community, variation_of_information, pd_fill_diagonal

plt.style.use('classic')

#dataset import
sp500 = pd.read_csv('/content/drive/My Drive/collab_files/^GSPC.csv', header = 0, index_col = 'Date')
sp500.index = pd.to_datetime(sp500.index, format = '%d-%m-%y')
sp500 = sp500[1:]
#sp500 = sp500.resample('W').mean()
#sp500.head()
print(len(sp500))

#import nifty50 data
nifty = pd.read_csv('/content/drive/My Drive/collab_files/^NSEI.csv', header = 0, index_col = 'Date')
nifty.index = pd.to_datetime(nifty.index, format = '%d-%m-%y')
nifty = nifty.reindex(index = sp500.index, method = 'bfill')
nifty.fillna(method = 'bfill', inplace=True)
#nifty = nifty.resample('W').mean()
#nifty.head()
print(len(nifty))

sing_sti = pd.read_csv('/content/drive/My Drive/collab_files/^sti_d.csv', header = 0, index_col = 'Date')
sing_sti.index = pd.to_datetime(sing_sti.index, format = '%Y-%m-%d')
sing_sti = sing_sti.reindex(index = sp500.index, method = 'bfill')
sing_sti.fillna(method = 'bfill', inplace=True)
print(len(sing_sti))

uk_100 = pd.read_csv('/content/drive/My Drive/collab_files/^ukx_d.csv', header = 0, index_col = 'Date')
uk_100.index = pd.to_datetime(uk_100.index, format = '%Y-%m-%d')
uk_100 = uk_100.reindex(index = sp500.index, method = 'bfill')
uk_100.fillna(method = 'bfill', inplace=True)
print(len(uk_100))

hangseng = pd.read_csv('/content/drive/My Drive/collab_files/^hsi_d.csv', header = 0, index_col = 'Date')
hangseng.index = pd.to_datetime(hangseng.index, format = '%Y-%m-%d')
hangseng = hangseng.reindex(index = sp500.index, method = 'bfill')
hangseng.fillna(method = 'bfill', inplace=True)
print(len(hangseng))

nikkei = pd.read_csv('/content/drive/My Drive/collab_files/^nkx_d.csv', header = 0, index_col = 'Date')
nikkei.index = pd.to_datetime(nikkei.index, format = '%Y-%m-%d')
nikkei = nikkei.reindex(index = sp500.index, method = 'bfill')
nikkei.fillna(method = 'bfill', inplace=True)
print(len(nikkei))

shanghai_comp = pd.read_csv('/content/drive/My Drive/collab_files/^shc_d.csv', header = 0, index_col = 'Date')
shanghai_comp.index = pd.to_datetime(shanghai_comp.index, format = '%Y-%m-%d')
shanghai_comp = shanghai_comp.reindex(index = sp500.index, method = 'bfill')
shanghai_comp.fillna(method = 'bfill', inplace=True)
print(len(shanghai_comp))

inr = pd.read_csv('/content/drive/My Drive/collab_files/DEXINUS.csv', header = 0, index_col = 'DATE')
inr.index = pd.to_datetime(inr.index, format = '%Y-%m-%d')
inr = inr.reindex(index = sp500.index, method = 'bfill')
inr.fillna(method = 'bfill', inplace=True)
print(len(inr))

cny = pd.read_csv('/content/drive/My Drive/collab_files/DEXCHUS.csv', header = 0, index_col = 'DATE')
cny.index = pd.to_datetime(cny.index, format = '%Y-%m-%d')
cny = cny.reindex(index = sp500.index, method = 'bfill')
cny.fillna(method = 'bfill', inplace=True)
print(len(cny))

jpy = pd.read_csv('/content/drive/My Drive/collab_files/DEXJPUS.csv', header = 0, index_col = 'DATE')
jpy.index = pd.to_datetime(jpy.index, format = '%Y-%m-%d')
jpy = jpy.reindex(index = sp500.index, method = 'bfill')
jpy.fillna(method = 'bfill', inplace=True)
print(len(jpy))

sgd = pd.read_csv('/content/drive/My Drive/collab_files/DEXSIUS.csv', header = 0, index_col = 'DATE')
sgd.index = pd.to_datetime(sgd.index, format = '%Y-%m-%d')
sgd = sgd.reindex(index = sp500.index, method = 'bfill')
sgd.fillna(method = 'bfill', inplace=True)
print(len(sgd))

hkd = pd.read_csv('/content/drive/My Drive/collab_files/DEXHKUS.csv', header = 0, index_col = 'DATE')
hkd.index = pd.to_datetime(hkd.index, format = '%Y-%m-%d')
hkd = hkd.reindex(index = sp500.index, method = 'bfill')
hkd.fillna(method = 'bfill', inplace=True)
print(len(hkd))

gbp = pd.read_csv('/content/drive/My Drive/collab_files/DEXUSUK.csv', header = 0, index_col = 'DATE')
gbp.index = pd.to_datetime(gbp.index, format = '%Y-%m-%d')
gbp = gbp.reindex(index = sp500.index, method = 'bfill')
gbp.fillna(method = 'bfill', inplace=True)
print(len(gbp))

inr.iloc[:, 0] = pd.to_numeric(inr.iloc[:, 0].replace({'.':'0'}))
cny.iloc[:, 0] = pd.to_numeric(cny.iloc[:, 0].replace({'.':'0'}))
jpy.iloc[:, 0] = pd.to_numeric(jpy.iloc[:, 0].replace({'.':'0'}))
sgd.iloc[:, 0] = pd.to_numeric(sgd.iloc[:, 0].replace({'.':'0'}))
hkd.iloc[:, 0] = pd.to_numeric(hkd.iloc[:, 0].replace({'.':'0'}))
gbp.iloc[:, 0] = pd.to_numeric(gbp.iloc[:, 0].replace({'.':'0'}))

gbp = 1/gbp

df = pd.DataFrame(index = sp500.index)
df['nifty'] = nifty['Close']
df['sing_sti'] = sing_sti['Close']
df['hangseng'] = hangseng['Close']
df['nikkei'] = nikkei['Close']
df['shanghai_comp'] = shanghai_comp['Close']
df['sp500'] = sp500['Close']
df['uk_100'] = uk_100['Close']
df = df.transpose()

df_1 = pd.DataFrame(index = sp500.index)
df_1['inr'] = inr
df_1['sgd'] = sgd
df_1['hkd'] = hkd
df_1['jpy'] = jpy
df_1['cny'] = cny
df_1['gbp'] = gbp
df_1['usd'] = 1
df_1 = df_1.transpose()

df_1['base'] = 'usd'
df_1 = df_1.reset_index()
df_exp = df_1.set_index(['index', 'base'])
df_exp = df_exp.reset_index()
df_exp.set_index(['index', 'base'], inplace = True)

"""
Warning: for loops run for 6 hours. 
Import index_final.csv directly
"""
for currency_fix, base_fix in df_exp.index:
    for curr, base in df_exp.index[0:7]:
        df_exp.loc[(curr, currency_fix), :] = (df_exp.loc[(curr, base), :]/df_exp.loc[(currency_fix, base_fix), :])
        print(curr,base)

df['base'] = ['inr', 'sgd', 'hkd', 'jpy', 'cny', 'usd', 'gbp']
df = df.reset_index()
df_index = df.set_index(['index', 'base'])
for index, base in df_index.index[0:7]:
    for curr, base_curr in df_exp.loc[(slice(None), base), :].index:
        df_index.loc[(index, curr), :] = df_index.loc[(index, base), :]*df_exp.loc[(curr, base_curr), :]
        
''''''

df_index = pd.read_csv('/content/drive/My Drive/collab_files/index_final.csv', index_col = ['index', 'base'])
G_cache = {}

for i in range(0, len(df_index.columns) - 2, 2):
    corr_df = abs(df_index.iloc[:, i:i+20].T.corr())
    G_cache[df_index.columns[i]] = nx.from_pandas_adjacency(corr_df)
    
comm_cache_p1, comm_cache_mod_p1 = louvain_community({k:G_cache[k] for k in list(G_cache)[0:500]}, resolution = 1.032)
comm_cache_p2, comm_cache_mod_p2 = louvain_community({k:G_cache[k] for k in list(G_cache)[500:1000]}, resolution = 1.031)
comm_cache_p3, comm_cache_mod_p3 = louvain_community({k:G_cache[k] for k in list(G_cache)[1000:]}, resolution = 1.037)
comm_cache_cov, comm_cache_mod_cov = louvain_community({k:G_cache[k] for k in list(G_cache)[1468:1542]}, resolution = 1.037)

df_v = pd.DataFrame(index = list(G_cache), columns = ['var_info', 'modularity'])
for i in range(0, 499):
  df_v.loc[list(G_cache)[i+1], 'var_info'] = variation_of_information(comm_cache_mod_p1[list(G_cache)[i]], comm_cache_mod_p1[list(G_cache)[i+1]])

for timestamp in list(G_cache)[0:500]:
  G = G_cache[timestamp]
  df_v.loc[timestamp, 'modularity'] = louvain.modularity(comm_cache_p1[timestamp], G, weight = 'weight')

for i in range(500, 999):
  df_v.loc[list(G_cache)[i+1], 'var_info'] = variation_of_information(comm_cache_mod_p2[list(G_cache)[i]], comm_cache_mod_p2[list(G_cache)[i+1]])

for timestamp in list(G_cache)[500:1000]:
  G = G_cache[timestamp]
  df_v.loc[timestamp, 'modularity'] = louvain.modularity(comm_cache_p2[timestamp], G, weight = 'weight')
  
for i in range(1000, 1549):
  df_v.loc[list(G_cache)[i+1], 'var_info'] = variation_of_information(comm_cache_mod_p3[list(G_cache)[i]], comm_cache_mod_p3[list(G_cache)[i+1]])

for timestamp in list(G_cache)[1000:]:
  G = G_cache[timestamp]
  df_v.loc[timestamp, 'modularity'] = louvain.modularity(comm_cache_p3[timestamp], G, weight = 'weight')
  
df_v['timestamp'] = df_v.index
df_v.modularity = df_v.modularity.astype(float)
df_v.var_info = df_v.var_info.astype(float)

#df_v['modularity'].plot()
fig, ax = plt.subplots(figsize=(12,7))
plt.xticks(rotation=45)
#plt.ylim(0.75, plot_df['resolution'].max()+0.05)
ax.margins(x = 0)
g = sns.lineplot(data = df_v.iloc[:1501], x = 'timestamp', y = 'modularity', ax=ax, color='black')

g.xaxis.set_major_locator(ticker.LinearLocator(10))
ax1 = g.axes
#ax1.hlines(1.032, ls='--', color='red', linewidth=4, xmin = plot_df.loc[0, 'timestamp'], xmax = plot_df.loc[81300, 'timestamp'])
#ax1.hlines(1.031, ls='--', color='blue', linewidth=4, xmin = plot_df.loc[81301, 'timestamp'], xmax = plot_df.loc[162600, 'timestamp'])
#ax1.hlines(1.037, ls='--', color='green', linewidth=4, xmin = plot_df.loc[162601, 'timestamp'], xmax = plot_df.loc[243999, 'timestamp'])
ax1.vlines(x = plot_df.loc[81300, 'timestamp'], colors='purple', ymin = 0, ymax = df_v['modularity'].max()+0.05, linewidths = 4)
#ax1.vlines(x = df_v.iloc[1510, 2], colors='purple', ymin = 0, ymax = df_v['modularity'].max()+0.05, linewidths = 4)
ax1.vlines(x = plot_df.loc[162600, 'timestamp'], colors='purple', ymin = 0, ymax = df_v['modularity'].max()+0.05, linewidths = 4)

#df_v['var_info'].plot()
fig, ax = plt.subplots(figsize=(12,7))
plt.xticks(rotation=45)
plt.ylim(0, df_v['var_info'].max()+0.05)
ax.margins(x = 0)
g = plt.bar(df_v.iloc[1468:1542, 2], df_v.iloc[1468:1542, 0], color = 'black')

ax.xaxis.set_major_locator(ticker.LinearLocator(10))
#ax1 = g.axes
#ax1.hlines(1.032, ls='--', color='red', linewidth=4, xmin = plot_df.loc[0, 'timestamp'], xmax = plot_df.loc[81300, 'timestamp'])
#ax1.hlines(1.031, ls='--', color='blue', linewidth=4, xmin = plot_df.loc[81301, 'timestamp'], xmax = plot_df.loc[162600, 'timestamp'])
#ax1.hlines(1.037, ls='--', color='green', linewidth=4, xmin = plot_df.loc[162601, 'timestamp'], xmax = plot_df.loc[243999, 'timestamp'])
#ax1.vlines(x = plot_df.loc[81300, 'timestamp'], colors='purple', ymin = 0, ymax = df_v['var_info'].max()+0.05, linewidths = 4)
#ax1.vlines(x = plot_df.loc[162600, 'timestamp'], colors='purple', ymin = 0, ymax = df_v['var_info'].max()+0.05, linewidths = 4)

#community counter
merged = []
for timestamp in list(G_cache)[1500:1542]:
  merged.extend(comm_cache_mod_cov[timestamp])
merged_tuple = [tuple(elem) for elem in merged]
merged_dict = dict(Counter(merged_tuple))
sorted(merged_dict.items(), key=lambda item: item[1], reverse = True)

#measuring centrality
G_cache_centrality = {}

for i in range(0, len(df_index.columns) - 2, 2):
    corr_df = 1/abs(df_index.iloc[:, i:i+20].T.corr())
    #corr_df.fillna(0)
    corr_df = pd_fill_diagonal(corr_df, 0)
    G_cache_centrality[df_index.columns[i]] = nx.from_pandas_adjacency(corr_df)
    
betweenness_dict = {}
for timestamp in list(G_cache_centrality)[1468:1500]:
  G = G_cache_centrality[timestamp]
  betwenness = nx.current_flow_betweenness_centrality(G, weight = 'weight', solver = 'lu')
  betweenness_dict = {key: betweenness_dict.get(key, 0) + betwenness.get(key, 0) for key in set(betweenness_dict) | set(betwenness)}
  #print(timestamp)
  
sorted(betweenness_dict.items(), key=lambda item: item[1], reverse = True)

#plotting network
G = G_cache[list(G_cache)[1470]]
partition = comm_cache_cov[list(G_cache)[1470]]
plt.figure(figsize=(12,8))
# draw the graph
pos = nx.spring_layout(G, seed = 1337)
# color the nodes according to their partition
shapes = 'so^>v<dph8'
cmap = cm.get_cmap('viridis', max(partition.values()) + 1)
nx.draw_networkx_edges(G, pos, alpha=0.5)
for node, color in partition.items():
    nx.draw_networkx_nodes(G, pos, [node], node_size=300,
                           node_color=[cmap.colors[color]],
                           node_shape=shapes[color])
nx.draw_networkx_labels(G, pos, font_color='black', font_size = 9, verticalalignment='top', horizontalalignment='left')
nx.draw_networkx_edges(G, pos, edge_color='darkblue')

"""
WARNING: Runtime is 6 hours. Please use 'comm_count_final.csv' to import data
"""
df_res = pd.DataFrame(columns = G_cache.keys())
for timestamp in list(G_cache):
  G = G_cache[timestamp]
  print(timestamp)
  for i in range(500, 1200):
    df_res.loc[i-500, timestamp] = max((louvain.best_partition(G, random_state=1337, resolution=i/1000)).values())+1
    
G = G_cache[list(G_cache)[100]]
#print(timestamp)
for i in range(500, 1200):
  mod = (louvain.best_partition(G, random_state=1337, resolution=i/1000))
  mod1 = (louvain.best_partition(G, random_state=1337, resolution=(i/1000)+0.001))
  df_res.loc[i-500, 'comm_count'] = max(mod.values())+1
  df_res.loc[i-500, 'vi'] = max(mod.values())+1
  
''''''

#choosing resolution
df_res = pd.read_csv('/content/drive/My Drive/collab_files/comm_count_final.csv', index_col = ['index'])
G = G_cache[list(G_cache)[100]]
comm_cache_mod = {}
for res in range(700):
  comm_iter = louvain.best_partition(G, weight = 'weight', random_state = 1337, resolution = (res/1000)+0.5)
  comm_count = max(comm_iter.values())+1
  comm_cache_mod_list = [[] for i in range(comm_count)]
  for node in list(comm_iter):
    i = comm_iter[node]
    comm_cache_mod_list[i].append(node)
  comm_cache_mod[(res)+500] = comm_cache_mod_list
  
df_vi = pd.DataFrame(index = list(comm_cache_mod))
for res in df_vi.index[:-1]:
  df_vi.loc[res, list(G_cache)[100]] = variation_of_information(comm_cache_mod[res], comm_cache_mod[res+1])
for i in df_vi.index:
  df_vi.loc[i, 'comm_count'] = df_res.loc[i-500, list(G_cache)[100]]

#community counter plots
plt.bar(df_vi.index, df_vi[list(G_cache)[100]])
plt.plot(df_vi['comm_count']/100)
plt.xlabel('resolution*100')
plt.grid(b = True)

"""
Warning: High runtime
import "plot_df.csv"
"""
plot_df = pd.DataFrame(columns = ['timestamp', 'resolution'])
for timestamp in list(G_cache):
  print(timestamp)
  df_mod = df_res[(df_res[timestamp] == 4)].append(df_res[df_res[timestamp] == 3])
  for res in df_mod.index:
    plot_df = plot_df.append({'timestamp': timestamp, 'resolution': (res/1000)+0.5}, ignore_index=True)
plot_df.to_csv('/content/drive/My Drive/collab_files/plot_df.csv', index_label = 'index')

''''''

plot_df = pd.read_csv('/content/drive/My Drive/collab_files/plot_df.csv', index_col = ['index'])
#plateau plot
fig, ax = plt.subplots(figsize=(12,7))
plt.xticks(rotation=45)
plt.ylim(0.75, plot_df['resolution'].max()+0.05)
ax.margins(x = 0)
g = sns.lineplot(data = plot_df, x = 'timestamp', y = 'resolution', ax=ax, color='black')

g.xaxis.set_major_locator(ticker.LinearLocator(10))
ax1 = g.axes
ax1.hlines(1.032, ls='--', color='red', linewidth=4, xmin = plot_df.loc[0, 'timestamp'], xmax = plot_df.loc[81300, 'timestamp'])
ax1.hlines(1.031, ls='--', color='blue', linewidth=4, xmin = plot_df.loc[81301, 'timestamp'], xmax = plot_df.loc[162600, 'timestamp'])
ax1.hlines(1.037, ls='--', color='green', linewidth=4, xmin = plot_df.loc[162601, 'timestamp'], xmax = plot_df.loc[243999, 'timestamp'])
ax1.vlines(x = plot_df.loc[81300, 'timestamp'], colors='purple', ymin = 0.75, ymax = plot_df['resolution'].max()+0.05, linewidths = 4)
ax1.vlines(x = plot_df.loc[162600, 'timestamp'], colors='purple', ymin = 0.75, ymax = plot_df['resolution'].max()+0.05, linewidths = 4)

