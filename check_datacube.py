# -*- coding: utf-8 -*-
"""
Created on Wed Jan 13 09:46:23 2021

@author: svc_ccg
"""

from run_qc_class import run_qc
from matplotlib import pyplot as plt
import pandas as pd
import get_sessions as gs
import json, os, glob

#TODO: LOGGING!!! 

sources = [r"\\10.128.50.43\sd6.3", r"\\10.128.50.20\sd7", r"\\10.128.50.20\sd7.2", 
           r"\\10.128.54.20\sd8", r"\\10.128.54.20\sd8.2"]
sessions_to_run = gs.get_sessions(sources, mouseID='!366122!544480', start_date='20200930')#, end_date='20200930')
destination = r"\\allen\programs\braintv\workgroups\nc-ophys\corbettb\NP_behavior_pipeline\mochi"

qc_dirs = [os.path.join(destination, os.path.basename(s)) for s in sessions_to_run]

[os.path.exists(q) for q in qc_dirs]

mid_dict = {}
session_genotypes = {}
mid_genotype = {}
for qd in qc_dirs:
    
    if os.path.exists(qd):
        
        #get specimen meta json
        specimen_meta = glob.glob(os.path.join(qd, 'specimen_meta.json'))
        if len(specimen_meta)>0:
            
            with open(specimen_meta[0], 'r') as file:
                specimen_info = json.load(file)
                mid = specimen_info['mid']
                
                session_genotypes[os.path.basename(qd)] = {'mid': specimen_info['mid'],
                                                          'genotype': specimen_info['genotype']} 
                mid_genotype[specimen_info['mid']] = specimen_info['genotype']
                
                if mid in mid_dict:
                    mid_dict[mid]['sessions'].append(os.path.basename(qd))
                else:
                    mid_dict[mid] = {'genotype': specimen_info['genotype'], 'sessions':[os.path.basename(qd)]}



for geno in ['C57BL6J', 'Sst-IRES-Cre;Ai32', 'Vip-IRES-Cre;Ai32']:
    print(geno)
    for mid in mid_dict:
        
        if mid_dict[mid]['genotype'] == geno:
            print(mid)#, mid_dict[mid]['sessions'])


for geno in ['C57BL6J', 'Sst-IRES-Cre;Ai32', 'Vip-IRES-Cre;Ai32']:
    print(geno)
    for mid in mid_dict:
        
        if mid_dict[mid]['genotype'] == geno:
            print(mid_dict[mid]['sessions'][0])


for geno in ['C57BL6J', 'Sst-IRES-Cre;Ai32', 'Vip-IRES-Cre;Ai32']:
    print(geno)
    for mid in mid_dict:
        
        if mid_dict[mid]['genotype'] == geno:
            print(mid_dict[mid]['sessions'][-1])