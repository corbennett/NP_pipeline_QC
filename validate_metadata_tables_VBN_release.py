# -*- coding: utf-8 -*-
"""
Created on Fri Apr 29 15:22:24 2022

@author: svc_ccg
"""
import pandas as pd
import numpy as np

df = pd.read_excel(r"C:\Users\svc_ccg\ccb_onedrive\OneDrive - Allen Institute\all_np_behavior_mice.xlsx")

b = pd.read_csv(r"\\allen\programs\mindscope\workgroups\np-behavior\vbn_data_release\metadata_220429\behavior_sessions.csv")

mice =  b['mouse_id'].unique()


def two_ephys(mdf):
    
    ephys = mdf.loc[(mdf['session_type'].str.contains('EPHYS'))&
                    (~mdf['session_type'].str.contains('pretest'))]
    
    return len(ephys)==2, len(ephys)

def pre_ephys_same_set(mdf):
    
    pre_ephys_df = mdf.loc[~mdf['session_type'].str.contains('EPHYS')]
    pre_ephys_image_set = pre_ephys_df.image_set.dropna()
    
    return len(np.unique(pre_ephys_image_set)) == 1


validation_funcs = [two_ephys, pre_ephys_same_set]
validation_dict = {func.__name__:[] for func in validation_funcs}
validation_dict.update({'mouse_id':[]})
for mouse in mice:
    
    mdf = b.loc[b['mouse_id']==mouse]
    mdf = mdf.sort_values('date_of_acquisition')
    validation_dict['mouse_id'].append(mouse)

    for func in validation_funcs:
        validation_dict[func.__name__].append(func(mdf))
    