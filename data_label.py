#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 13 20:07:15 2020

@author: mahade
"""

"""Encryption of labels to make unnique cell values. Dropped ''veil-type for having only one value regardless of 
mashroom being edible or poisonous. Dropped the 'stalk-root' column for having too many missing values"""
def data_labelling(df):
    df = df.drop(['stalk-root'], axis=1).iloc[:, :]
    df = df.drop(['veil-type'], axis=1).iloc[:, :]
    
    df['cap-shape'] = df['cap-shape'].replace(['b', 'c', 'x', 'f', 'k', 's'], ['bell', 'conical', 'convex', 'flat', 'knobbed', 'sunken'])
    df['cap-surface'] = df['cap-surface'].replace(['f', 'g', 'y', 's'], ['fibrous', 'grooves', 'scaly', 'smooth'])
    df['cap-color'] = df['cap-color'].replace(['n', 'b', 'c', 'g', 'r', 'p', 'u', 'e', 'w', 'y'], ['brown', 'buff', 'cinnamon', 'gray', 'green', 'pink', 'purple', 'red', 'white', 'yellow'])
    df['bruises'] = df['bruises'].replace(['t', 'f'], ['bruises', 'no'])
    df['odor'] = df['odor'].replace(['a', 'l', 'c', 'y', 'f', 'm', 'n', 'p', 's'], ['almond', 'anise', 'cresote', 'fishy', 'foul', 'musty', 'none', 'pungent', 'spicy'])
    df['gill-attachment'] = df['gill-attachment'].replace(['a', 'd', 'f', 'n'], ['attached', 'descending', 'free', 'notched'])
    df['gill-spacing'] = df['gill-spacing'].replace(['c', 'w', 'd'], ['close', 'crowded', 'distant'])
    df['gill-size'] = df['gill-size'].replace(['b', 'n'], ['broad', 'narrow'])
    df['gill-color'] = df['gill-color'].replace(['k', 'n', 'b', 'h', 'g', 'r', 'o', 'p', 'u', 'e', 'w', 'y'], ['black_', 'brown_', 'buff_', 'chocolate_', 'gray_', 'green_', 'orange_', 'pink_', 'purple_', 'red_', 'white_', 'yellow_'])
    df['stalk-shape'] = df['stalk-shape'].replace(['e', 't'], ['eng', 'tang'])
    df['stalk-surface-above-ring'] = df['stalk-surface-above-ring'].replace(['f', 'k', 's', 'y'], ['full', 'kile', 'small', 'yield'])
    df['stalk-surface-below-ring'] = df['stalk-surface-below-ring'].replace(['f', 'k', 's', 'y'], ['full_', 'kile_', 'small_', 'yield_'])
    df['stalk-color-above-ring'] = df['stalk-color-above-ring'].replace(['b', 'c', 'e', 'g', 'n', 'o', 'p', 'w', 'y'], ['_b', '_c', '_e', '_g', '_n', '_o', '_p', '_w', '_y'])
    df['stalk-color-below-ring'] = df['stalk-color-below-ring'].replace(['b', 'c', 'e', 'g', 'n', 'o', 'p', 'w', 'y'], ['__b', '__c', '__e', '__g', '__n', '__o', '__p', '__w', '__y'])
    df['veil-color'] = df['veil-color'].replace(['n', 'o', 'w', 'y'], ['enn', 'ooo', 'dublew', 'wii'])
    df['ring-number'] = df['ring-number'].replace(['n', 'o', 't'], ['_n_', '_o_', '_t_'])
    df['ring-type'] = df['ring-type'].replace(['e', 'f', 'l', 'n', 'p'], ['type_e', 'type_f', 'type_l', 'type_n', 'type_p'])
    df['spore-print-color'] = df['spore-print-color'].replace(['b', 'h', 'k', 'n', 'o', 'r', 'u', 'w', 'y'], ['color_b', 'color_h', 'color_k', 'color_n', 'color_o', 'color_r', 'color_u', 'color_w', 'color_y'])
    df['population'] = df['population'].replace(['a', 'c', 'n', 's', 'v', 'y'], ['pop_a', 'pop_c', 'pop_n', 'pop_s', 'pop_v', 'pop_y'])
    df['habitat'] = df['habitat'].replace(['d', 'g', 'l', 'm', 'p', 'u', 'w'], ['habi_d', 'habi_g', 'habi_l', 'habi_m', 'habi_p', 'habi_u', 'habi_w'])

    return df