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
from sklearn.metrics import f1_score, accuracy_score, recall_score, roc_auc_score, cohen_kappa_score, precision_score

def metric_calc(pred_labels, true_labels):
    TP = np.sum(np.logical_and(pred_labels == 1, true_labels == 1))
    FP = np.sum(np.logical_and(pred_labels == 1, true_labels == 0))
    TN = np.sum(np.logical_and(pred_labels == 0, true_labels == 0))
    FN = np.sum(np.logical_and(pred_labels == 0, true_labels == 1))
    return(TP, FP, TN, FN)

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

nrange = valid_df.shape[1]
for i in range(nrange):
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
Converte em numero
'''

prev_rna = []
for i in rna_previsoes['0']:
    if i:
        prev_rna.append(1)
    else:
        prev_rna.append(0)

prev_nb = []
for i in nb_previsoes['0']:
    if i:
        prev_nb.append(1)
    else:
        prev_nb.append(0)

prev_svm = []
for i in svm_previsoes['0']:
    if i:
        prev_svm.append(1)
    else:
        prev_svm.append(0)
        
'''
Mais metricas do melhor cenario
'''

ensemble_best = []
tam = len(prev_rna)
for i in range(tam):
    if prev_rna[i] + prev_nb[i] + prev_svm[i] > 1:
        ensemble_best.append(1)
    else:
        ensemble_best.append(0)
ensemble_best = np.array(ensemble_best)

tp, fp, tn, fn = metric_calc(ensemble_best, y_valid)

print('====== ENSEMBLE METRIC RESULTS ========')
f1 = f1_score(y_valid, ensemble_best)
print('Scenary 1 F1-SCORE:' , f1)
acc = accuracy_score(y_valid, ensemble_best)
print('Scenary 1 Acuraccy:' , acc)
rec = recall_score(y_valid, ensemble_best)
print('Scenary 1 Recall:' , rec)
roc = roc_auc_score(y_valid, ensemble_best)
print('Scenary 1 ROC AUC:' , roc)
kappa = cohen_kappa_score(y_valid, ensemble_best)
print('Scenary 1 Kappa:' , roc)
prec = precision_score(y_valid, ensemble_best)
print('Scenary 1 Precision:' , prec)
specificity = tn / (tn+fp)
print('Scenary 1 Specificity:' , specificity)

prev_rna = np.array(prev_rna)
tp, fp, tn, fn = metric_calc(prev_rna, y_valid)
print('====== ANN METRIC RESULTS ========')
f1 = f1_score(y_valid, prev_rna)
print('ANN F1-SCORE:' , f1)
acc = accuracy_score(y_valid, prev_rna)
print('ANN Acuraccy:' , acc)
rec = recall_score(y_valid, prev_rna)
print('ANN Recall:' , rec)
roc = roc_auc_score(y_valid, prev_rna)
print('ANN ROC AUC:' , roc)
kappa = cohen_kappa_score(y_valid, prev_rna)
print('ANN Kappa:' , roc)
prec = precision_score(y_valid, prev_rna)
print('ANN Precision:' , prec)
specificity = tn / (tn+fp)
print('ANN Specificity:' , specificity)