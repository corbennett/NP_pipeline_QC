# -*- coding: utf-8 -*-
"""
Created on Wed Feb 15 17:32:11 2023

@author: svc_ccg
"""

import numpy as np
import json
import pandas as pd
import os


metrics_dir = r"\\allen\programs\mindscope\workgroups\np-behavior\vbn_data_release\behavior_validation\behavior_metrics"

metrics_files = os.listdir(metrics_dir)

metrics_files = [os.path.join(metrics_dir, m) for m in metrics_files]

def read_json(path):
    
    with open(path, 'r') as f:
        j = json.load(f)
    
    return j

mean_speeds = {'speed':[], 'session':[]}
for m in metrics_files:
    
    data = read_json(m)
    mean_speeds['speed'].append(data['mean_speed'])
    mean_speeds['session'].append(os.path.basename(m)[:-5])
    
mean_speeds_df = pd.DataFrame(mean_speeds)

suspect = mean_speeds_df[mean_speeds_df['speed']<0]['session']
source_dir = r"\\allen\programs\mindscope\workgroups\np-behavior\vbn_data_release\behavior_validation\running"
save_dir = r"\\allen\programs\mindscope\workgroups\np-behavior\vbn_data_release\behavior_validation\encoder_problems"
import shutil
import glob
for s in suspect:
    
    file = glob.glob(os.path.join(source_dir, '*'+s+'*'))[0]
    #file = [m for m in metrics_files if s in m][0]
    shutil.copy(file, os.path.join(save_dir, os.path.basename(file)))
    
    
manually_ided = [
    '1029420108_change_triggered_running',
    '1069660976_change_triggered_running',
    '1070957221_change_triggered_running',
    '1071280305_change_triggered_running',
    '1078579330_change_triggered_running',
    '1078783957_change_triggered_running',
    '1079013681_change_triggered_running',
    '1079223135_change_triggered_running',
    '1088209099_change_triggered_running',
    '1091800400_change_triggered_running',
    '1092231641_change_triggered_running',
    '1096262499_change_triggered_running',
    '1096465710_change_triggered_running',
    '1096658058_change_triggered_running',
    '1096701870_change_triggered_running',
    '1096951446_change_triggered_running',
    '1096962903_change_triggered_running',
    '1097741489_change_triggered_running',
    '1097960997_change_triggered_running',
    '1098141739_change_triggered_running',
    '1098155124_change_triggered_running',
    '1098371885_change_triggered_running',
    '1098385687_change_triggered_running',
    '1098523049_change_triggered_running',
    '1099164948_change_triggered_running',
    '1099395913_change_triggered_running',
    '1099579114_change_triggered_running',
    '1099623071_change_triggered_running',
    '1099832458_change_triggered_running',
    '1099898239_change_triggered_running',
    '1100087539_change_triggered_running',
    '1100176592_change_triggered_running',
    '1100194778_change_triggered_running',
    '1100230109_change_triggered_running',
    '1100688180_change_triggered_running',
    '1100793165_change_triggered_running',
    '1101080541_change_triggered_running',
    '1101322871_change_triggered_running',
    '1101486794_change_triggered_running',
    '1101698040_change_triggered_running',
    '1102183214_change_triggered_running',
    '1102211933_change_triggered_running',
    '1106044953_change_triggered_running'
]

manually_ided = [m.split('_')[0] for m in manually_ided]



