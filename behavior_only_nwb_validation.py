# -*- coding: utf-8 -*-
"""
Created on Mon Nov 14 17:44:48 2022

@author: svc_ccg
"""
import json, os
import numpy as np
import pandas as pd

beh_session_table = pd.read_csv(os.path.join(r'\\allen\programs\mindscope\workgroups\np-behavior\vbn_data_release\behavior_validation', 'behavior_sessions_table.csv'))
beh_session_table = beh_session_table.set_index('behavior_session_id')

save_dir = r"\\allen\programs\mindscope\workgroups\np-behavior\vbn_data_release\behavior_validation"

validation_dir = os.path.join(save_dir, 'validation_jsons')
validation_files = os.listdir(validation_dir)

def read_json(path):
    
    with open(path, 'r') as f:
        j = json.load(f)
    
    return j


failures = []
for vf in validation_files:
    
    vfpath = os.path.join(validation_dir, vf)
    results = read_json(vfpath)
    
    if results['summary']==False:
        failures.append(vf)        

failure_types = [f.split('_')[1] for f in failures]
failure_sessions = [f.split('_')[0] for f in failures]

failed_df = pd.DataFrame({'session_id': failure_sessions,
                          'problem': failure_types,
                          'filename': failures})

failed_df['session_id'] = failed_df['session_id'].astype(int)  
failed_df = failed_df[failed_df['problem']=='trialtypes']
failed_df = failed_df.merge(beh_session_table, left_on='session_id', right_index=True)

problem_trial_type = []
for f in failed_df.filename.values:
    fpath = os.path.join(validation_dir, f)
    results = read_json(fpath)
    
    badtypes = [k for k,v in results.items() if (v==False)&(k!='summary')]
    problem_trial_type.append(badtypes)
    

failed_df.to_csv(os.path.join(r"\\allen\programs\mindscope\workgroups\np-behavior\vbn_data_release\behavior_validation", "failed_sessions.csv"))


failed_session_dir = r"\\allen\programs\mindscope\workgroups\np-behavior\vbn_data_release\behavior_validation\failed_session_analysis"
failed_sessions = os.listdir(r"\\allen\programs\mindscope\workgroups\np-behavior\vbn_data_release\behavior_validation\failed_session_analysis")

df = pd.DataFrame()
for f in failed_sessions:
    filepath = os.path.join(failed_session_dir, f)
    sdf = pd.read_csv(filepath)
    
    df = pd.concat([df, sdf])


rw75 = df[df['response_window_end']==0.75]



behavior_metrics_dir = r"\\allen\programs\mindscope\workgroups\np-behavior\vbn_data_release\behavior_validation\behavior_metrics"
behavior_metrics_files = os.listdir(behavior_metrics_dir)

for bmf in behavior_metrics_files:
    
    m = read_json(os.path.join(behavior_metrics_dir, bmf))
    session_id = bmf.split('.')[0]
    beh_session_table.loc[int(session_id), 'total_licks'] = m['total_licks']
    beh_session_table.loc[int(session_id), 'mean_speed'] = m['mean_speed']







