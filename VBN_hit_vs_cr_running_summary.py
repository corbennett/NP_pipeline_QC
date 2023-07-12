# -*- coding: utf-8 -*-
"""
Created on Thu Nov  3 16:03:06 2022

@author: svc_ccg
"""
import numpy as np
import pandas as pd
import os
from matplotlib import pyplot as plt
from analysis import formatFigure

save_dir = r"\\allen\programs\mindscope\workgroups\np-behavior\vbn_data_release\hit_cr_running"

df_files = os.listdir(save_dir)

cr_running = []
hit_running = []
for dff in df_files:
    
    df = pd.read_csv(os.path.join(save_dir, dff))
    
    cr = df['cr_running'].values
    hit = df['hit_running'].values
    
    cr_running.append(cr)
    hit_running.append(hit)
    
time = df['time'].values

fig, ax = plt.subplots()
fig_inset, ax_inset = plt.subplots()
for r, color in zip([cr_running, hit_running], ['g', 'k']):
    
    r = np.array([t for t in r if not np.any(np.isnan(t))])
    base_inds = np.where((time>=-0.5)&(time<-0.25))[0]
    r_sub = r - np.mean(r[:, base_inds], axis=1)[:, None]
    r_sub = r
    mean = np.mean(r_sub, axis=0)
    sem = np.std(r_sub, axis=0)/len(r)**0.5
    ax.plot(time, mean, color=color)
    ax.fill_between(time, mean-sem, mean+sem, color=color, alpha=0.5)
    ax.set_xlabel('time from change or sham change, s')
    
    ax_inset.plot(time, mean, color=color)
    ax_inset.fill_between(time, mean-sem, mean+sem, color=color, alpha=0.5)
    ax_inset.set_xlim([0, 0.25])

formatFigure(fig, ax, xLabel='time from change or sham change (s)', yLabel='\u0394 run speed (cm/s)')
formatFigure(fig_inset, ax_inset, xLabel='time from change or sham change (s)', yLabel='\u0394 run speed (cm/s)')

