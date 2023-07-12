# -*- coding: utf-8 -*-
"""
Created on Wed Jul 12 15:14:05 2023

@author: svc_ccg
"""
import numpy as np
import pandas as pd

structure_tree = pd.read_csv(r"\\allen\programs\mindscope\workgroups\dynamicrouting\dynamic_gating_insertions\ccf_structure_tree_2017.csv")
def get_parent(area, structure_tree):
    
    parent_id = structure_tree[structure_tree['acronym']==area]['parent_structure_id'].values[0]
    return structure_tree[structure_tree['id'] == parent_id]
    
def get_area(area, structure_tree):
    return structure_tree[structure_tree['acronym'] == area]
    
def list_parents(area, structure_tree):
    if area == 'root':
        return []
    
    area_info = get_area(area, structure_tree)
    if len(area_info['name'].values)>0:
        starting_entry = area_info['name'].values[0]
    else:
        starting_entry = ''
    
    parent_list = [starting_entry]
    parent_structure_id = 0.0
    
    try:
    
        while not np.isnan(parent_structure_id):
            parent = get_parent(area, structure_tree)
            parent_list.append(parent['name'].values[0])
            parent_structure_id = parent['parent_structure_id'].values[0]
            area = parent['acronym'].values[0]
    except:
        print(f'failed to get parents for {area}')
    
    return parent_list

def get_brain_division_for_area(area, structure_tree, 
                                divisions = ['Isocortex', 'Hippocampal formation', 
                                             'Thalamus', 'Midbrain', 'Hypothalamus',
                                             'Striatum', 'Olfactory areas'],
                               cached_dict = None):
    
    if cached_dict:
        if area in cached_dict.keys():
            return cached_dict[area]
        
    parents = list_parents(area, structure_tree)
    intersection = np.intersect1d(parents, divisions)
    if len(intersection)>0:
        return intersection[0]
    
    else:
        print(area)
        return 'not in list'
    
def get_area_color(area, structure_tree):
    
    area = get_area(area, structure_tree)
    if area.size>0:
        color = area['color_hex_triplet'].values[0]
    else:
        color = '808080'
    return '#' + color