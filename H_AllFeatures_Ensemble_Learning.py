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
from sklearn.metrics import f1_score, accuracy_score, recall_score, roc_auc_score, cohen_kappa_score

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

valid_df = pd.read_csv(Path('feature-dataframes/PatLvDiv_TEST-AllFeats_1612-Features_1503-images.csv'), index_col=0)
print('Done Read Validation Dataframe!')

print('Preparing Data...')

for i in range(1613):
    valid_df[valid_df.columns[i]] = clean_Dirt_Data(valid_df[valid_df.columns[i]])

x_valid, y_valid = prepareData(valid_df)

nb_previsoes = pd.read_csv(Path('previsoes/naive_bayes_previsoes.csv'), index_col = 0)
knn_previsoes = pd.read_csv(Path('previsoes/knn_previsoes.csv'), index_col = 0)
svm_previsoes = pd.read_csv(Path('previsoes/svm_previsoes.csv'), index_col = 0)
rna_previsoes = pd.read_csv(Path('previsoes/rna_previsoes.csv'), index_col = 0)

ensemble_scenary_1 = rna_previsoes + knn_previsoes + svm_previsoes
ensemble_scenary_2 = rna_previsoes + nb_previsoes + svm_previsoes
ensemble_scenary_3 = rna_previsoes + knn_previsoes + nb_previsoes
ensemble_scenary_4 = svm_previsoes + knn_previsoes + nb_previsoes

precisao = f1_score(y_valid, nb_previsoes)
print('Naive Bayes F1-Score:' , precisao)

precisao = f1_score(y_valid, knn_previsoes)
print('KNN 1 F1-Score:' , precisao)

precisao = f1_score(y_valid, svm_previsoes)
print('SVC F1-Score:' , precisao)

precisao = f1_score(y_valid, ensemble_scenary_1)
print('Scenary 1 F1-Score:' , precisao)

precisao = f1_score(y_valid, ensemble_scenary_2)
print('Scenary 2 F1-Score:' , precisao)

precisao = f1_score(y_valid, ensemble_scenary_3)
print('Scenary 3 F1-Score:' , precisao)

precisao = f1_score(y_valid, ensemble_scenary_4)
print('Scenary 4 F1-Score:' , precisao)

precisao = f1_score(y_valid, rna_previsoes)
print('All Features RNA F1-Score:' , precisao)

'''
Mais metricas do melhor cenario
'''

acc = accuracy_score(y_valid, ensemble_scenary_1)
print('Scenary 1 Acuraccy:' , acc)
rec = recall_score(y_valid, ensemble_scenary_1)
print('Scenary 1 Recall:' , rec)
roc = roc_auc_score(y_valid, ensemble_scenary_1)
print('Scenary 1 ROC AUC:' , roc)
kappa = cohen_kappa_score(y_valid, ensemble_scenary_1)
print('Reduced Ensemble Kappa:' , roc)