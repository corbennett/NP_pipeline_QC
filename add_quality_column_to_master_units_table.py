# -*- coding: utf-8 -*-
"""
Created on Thu Mar  2 17:52:57 2023

@author: svc_ccg
"""
import os
import pandas as pd

unit_quality_dir = r"\\allen\programs\mindscope\workgroups\np-behavior\vbn_data_release\unit_quality"
unit_quality_files = os.listdir(unit_quality_dir)

df = pd.DataFrame()
for u in unit_quality_files:
    udf = pd.read_csv(os.path.join(unit_quality_dir, u))
    df = pd.concat([df, udf])
    

master_unit_table = r"\\allen\programs\mindscope\workgroups\np-behavior\vbn_data_release\supplemental_tables\units_with_cortical_layers.csv"

master_table = pd.read_csv(master_unit_table)

new_master = master_table.merge(df, left_on='unit_id', right_on='id')

cols_to_drop = [c for c in new_master.columns if 'Unnamed:' in c] + ['id']

new_master = new_master.drop(columns=cols_to_drop)

new_master.to_csv(r"\\allen\programs\mindscope\workgroups\np-behavior\vbn_data_release\supplemental_tables\master_unit_table.csv")