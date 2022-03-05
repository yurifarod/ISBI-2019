#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Chagging code for extend paper on 14/01/2022;

Created on Thu May 13 11:13:38 2021

@author: yurifarod, Elwyslan
"""

import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import GaussianNB
from sklearn.feature_selection import SequentialFeatureSelector

def clean_Dirt_Data(x):
    ret = []
    for i in x:
        i = str(i)
        ret.append(float(i.replace('[', '').replace('.]', '').replace(']', '') ))
    
    return pd.DataFrame(ret)

def prepareData(data_df):
    #Prepare Validation data
    y = data_df['cellType(ALL=1, HEM=-1)'].values
    for i in range(len(y)):
        if y[i]==-1:
            y[i] = 0
        elif y[i]==1:
            y[i] = 1
    y = np.array(y)
    x = data_df.drop(['cellType(ALL=1, HEM=-1)'], axis=1)
    for col in x.columns:
        x[col] = (x[col] - data_df[col].mean()) / data_df[col].std() #mean=0, std=1
    x = x.values
    return x, y

print('Reading Train Dataframe...')
train_df = pd.read_csv(Path('feature-dataframes/AugmPatLvDiv_TRAIN-AllFeats_1612-Features_6405-images.csv'), index_col=0)
print('Done Read Train Dataframe!')

print('Reading Validation Dataframe...')
valid_df = pd.read_csv(Path('feature-dataframes/PatLvDiv_TEST-AllFeats_1612-Features_607-images.csv'), index_col=0)
print('Done Read Validation Dataframe!')

print('Preparing Data...')

for i in range(1603):
    valid_df[valid_df.columns[i]] = clean_Dirt_Data(valid_df[valid_df.columns[i]])
    train_df[train_df.columns[i]] = clean_Dirt_Data(train_df[train_df.columns[i]])

x_train, y_train = prepareData(train_df)
x_valid, y_valid = prepareData(valid_df)

print('Done Read Train and Validation data!')

#Avaliacao do SVC
classificador_svm = LinearSVC()
# Sequential Forward Selection(sfs)
sfs = SequentialFeatureSelector(classificador_svm, n_features_to_select=1)
sfs.fit(x_train, y_train)
result_svc = sfs.get_support()

#Avaliacao do NB
classificador_nb = GaussianNB(priors=None, var_smoothing=1e-9)
# Sequential Forward Selection(sfs)
sfs = SequentialFeatureSelector(classificador_nb, n_features_to_select=1)
sfs.fit(x_train, y_train)
result_nb = sfs.get_support()