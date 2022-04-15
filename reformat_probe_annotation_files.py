# -*- coding: utf-8 -*-
"""
Created on Thu Mar 10 11:47:34 2022

@author: svc_ccg
"""

import pandas as pd
import numpy as np
import os, glob

structure_tree = pd.read_csv(r"\\allen\programs\mindscope\workgroups\np-behavior\ccf_structure_tree_2017.csv")
labels = np.load(r"C:\Users\svc_ccg\Desktop\Data\Atlas\annotation_volume_10um_by_index.npy")
ctx_annotations = pd.read_excel(r"C:\Users\svc_ccg\ccb_onedrive\OneDrive - Allen Institute\ctx_annotation.xlsx")
df = pd.read_excel(r"C:\Users\svc_ccg\ccb_onedrive\OneDrive - Allen Institute\all_np_behavior_mice.xlsx")
opt_dir = r'\\allen\programs\mindscope\workgroups\np-behavior\processed_ALL'
save_dir = r'\\allen\programs\braintv\workgroups\neuralcoding\corbettb\VBN_production'
#save_dir = opt_dir

def get_depth5_grandparent(struct_id):
    
    depth = 100
    s_id = struct_id
    out = 'None'
    while depth > 5:
        parent_id = structure_tree.loc[structure_tree['id']==s_id]['parent_structure_id']
        if parent_id.isnull().bool():
            break
        parent = structure_tree.loc[structure_tree['id']==int(parent_id)]
        s_id = int(parent['id'])
        depth = int(parent['depth'])
        out = parent['acronym'].squeeze()
    
    
    return out

#GET DATA FOR THIS MOUSE
failed = []
for im, mouseID in enumerate(ctx_annotations['animalID'].unique()):
    mouseID = str(int(mouseID))
    print('running {}, {} of {}'.format(mouseID, im, len(ctx_annotations['animalID'].unique())))
    try:
        mouseID = '548717'
        mouse_opt_dir = os.path.join(opt_dir, mouseID)
        final_ccf_coords_file = glob.glob(os.path.join(mouse_opt_dir, 'final_ccf_coordinates.csv'))[0]
        final_ccf_coords = pd.read_csv(final_ccf_coords_file)
        
        ctx_labels = ctx_annotations.loc[ctx_annotations['animalID'] == float(mouseID)]
        site_vertical_positions = np.arange(20, 3860, 20).repeat(2)
        site_horizontal_positions = [43, 11, 59, 27]*int(384/4)
        final_columns = ['structure_acronym', 'structure_id', 'A/P', 'D/V', 'M/L',
                         'horizontal_position', 'vertical_position', 'is_valid', 'cortical_depth']
        
        exp_ids = df.loc[df['mouse_id']==float(mouseID)].sort_values(by='session_date')['full_id'].to_list()
        
        for probe in final_ccf_coords['probe'].unique():
            
            #grab data for this probe, we'll have to save files for each probe individually
            probe_df = final_ccf_coords.loc[final_ccf_coords['probe']==probe]
            
            
            #remove white matter labels; just subsume them into deeper grey area
            ids = probe_df.structure_id.values
            names = [structure_tree.loc[sid].acronym for sid in probe_df.structure_id.values]
            is_white_matter = np.array([s.islower() for s in names])
            
            #first remove weird labels that sometimes happen above brain...
            if (not is_white_matter[0]) and (not 'VIS' in names[0]):
                ind = 0
                while ids[ind]>0:
                    ids[ind] = 0
                    ind += 1
                    #print('correcting ind {}'.format(ind))
            
            names = [structure_tree.loc[sid].acronym for sid in ids]
            is_white_matter = np.array([s.islower() for s in names])
            
            first_grey = np.where(~is_white_matter)[0][0]
            new_ids = []
            for ind, (iswm, sid) in enumerate(zip(is_white_matter, ids)):
                if ind<first_grey:
                    new_ids.append(sid)
                    continue
                else:
                    if iswm:
                        counter = ind
                        while counter<len(is_white_matter) and is_white_matter[counter]:
                            counter+=1
                        new_id = ids[counter] if counter<len(is_white_matter) else 1
                        new_ids.append(new_id)
                    else:
                        new_ids.append(sid)
            
            probe_df['structure_id'] = new_ids
            #condense dataframe to one row per channel
            #given the uncertainty in all of this, just take the closest row
            channels = probe_df['channels']
            indices_to_keep = []
            for chan in range(384):
                idx = np.argmin(abs(channels - chan))
                indices_to_keep.append(idx)
            
            probe_df_condensed = probe_df.loc[indices_to_keep]
            probe_df_condensed['channels'] = np.arange(384)
            probe_df_condensed = probe_df_condensed.set_index('channels')
            probe_df_condensed['horizontal_position'] = site_horizontal_positions
            probe_df_condensed['vertical_position'] = site_vertical_positions
            
            #REPLACE CORTICAL LABELS WITH ANNOTATIONS FROM THE ISI IMAGE
            probe_ctx_label = ctx_labels[probe[-2:]].squeeze()
            probe_ctx_label = probe_ctx_label.rstrip()
            probe_ctx_label = probe_ctx_label.lower()
            if probe_ctx_label == 'vislm':
                probe_ctx_label = 'visl'
            probe_ctx_label = probe_ctx_label.lower().replace('vis', 'VIS')
            
            for ir, row in probe_df_condensed.iterrows():
                
                s_id = row['structure_id']
                struct_row = structure_tree.loc[s_id]
                structure_label = struct_row['acronym']
                struct_name = struct_row['name']
                
                d5_parent = get_depth5_grandparent(int(struct_row['id']))
                
                if d5_parent == 'Isocortex' or 'Entorhinal area lateral' in struct_name:
                    if struct_name.find('layer ')>0:
                        layer = struct_name.split('layer ')[-1]
                    else:
                        layer = ''
                    
                    if (probe_ctx_label == 'nonVIS') or ('RSPv' in structure_label):
                        new_label = structure_label
                    else:
                        new_label = probe_ctx_label + layer
                    
                    if layer[-1]=='1' or ('RSP' in new_label):
                        ctx = -1
                    else:
                        ctx = 1
                    
                else:
                    new_label = structure_label
                    ctx = -1
                
                probe_df_condensed.at[ir, 'structure_acronym'] = new_label
                probe_df_condensed.at[ir, 'cortical_depth'] = ctx
                #original_labels.append((probe, structure_label))
                #new_labels.append((probe, new_label))
            
        
            #now adjust cortical depth column
            ctx_depth = np.zeros(384) - 1
            ctx_channels = np.where(probe_df_condensed['cortical_depth']==1)
            ctx_depth[ctx_channels] = np.linspace(0, 1, len(ctx_channels[0]))
            probe_df_condensed['cortical_depth'] = ctx_depth
            probe_df_condensed['is_valid'] = True
            probe_df_condensed.loc[191, 'is_valid'] = False
            
            #also clean up white matter area assignments
            
            
            probe_df_condensed = probe_df_condensed[final_columns]
            
            p_save_dir = os.path.join(save_dir, exp_ids[int(probe[-1])-1] + '_probe' + probe[6] + '_sorted')
            probe_df_condensed.to_csv(os.path.join(p_save_dir, 'ccf_regions.csv'))
                
    except Exception as e:
        print('failed to run mouse {}'.format(mouseID))
        failed.append((mouseID, e))
        
        
        
        
        
        