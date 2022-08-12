#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Chagging code for extend paper on 14/01/2022;

Created on Thu May 13 11:13:38 2021

@author: yurifarod, Elwyslan
"""

import csv
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import GaussianNB
from keras.wrappers.scikit_learn import KerasClassifier
from keras.layers import Dropout, Dense
from keras.models import Sequential
from tensorflow import keras
from sklearn.metrics import f1_score, accuracy_score, recall_score, roc_auc_score, cohen_kappa_score, precision_score

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

def metric_calc(pred_labels, true_labels):
    TP = np.sum(np.logical_and(pred_labels == 1, true_labels == 1))
    FP = np.sum(np.logical_and(pred_labels == 1, true_labels == 0))
    TN = np.sum(np.logical_and(pred_labels == 0, true_labels == 0))
    FN = np.sum(np.logical_and(pred_labels == 0, true_labels == 1))
    return(TP, FP, TN, FN)

print('Reading Train Dataframe...')
train_df = pd.read_csv(Path('feature-dataframes/AugmPatLvDiv_TRAIN-AllFeats_1612-Features_40000-images.csv'), index_col=0)
print('Done Read Train Dataframe!')

print('Reading Validation Dataframe...')
valid_df = pd.read_csv(Path('feature-dataframes/PatLvDiv_TEST-AllFeats_1612-Features_1503-images.csv'), index_col=0)

print('Done Read Validation Dataframe!')

print('Reducing Data...')

nb_features = pd.read_csv(Path('nb_interpretavel.csv'), index_col=0).values
svc_features = pd.read_csv(Path('svc_interpretavel.csv'), index_col=0).values
rna_features = pd.read_csv(Path('rna_interpretavel_manual.csv'), index_col=0).values

features = valid_df.columns

f = open('z_interpretable_ensemble_analysis.txt', 'w')

nrange = len(valid_df.columns)
for i in range(nrange):
    if i != 0:
        if nb_features[i-1] or svc_features[i-1] or rna_features[i-1]:
            f.write(features[i] + '\n')
        else:
            valid_df.drop(features[i], inplace=True, axis=1)
            train_df.drop(features[i], inplace=True, axis=1)
f.close()

print('Preparing Data...')

new_size = valid_df.shape[1]

for i in range(new_size):
    valid_df[valid_df.columns[i]] = clean_Dirt_Data(valid_df[valid_df.columns[i]])
    train_df[train_df.columns[i]] = clean_Dirt_Data(train_df[train_df.columns[i]])

x_train, y_train = prepareData(train_df)
x_valid, y_valid = prepareData(valid_df)

print('Done Read Train and Validation data!')

print('Training NB')
classificador = GaussianNB(priors=None, var_smoothing=1e-9)
classificador.fit(x_train, y_train)

previsoes_nb = classificador.predict(x_valid)

print('Training SVM')
classificador = LinearSVC()
classificador.fit(x_train, y_train)

previsoes_svc = classificador.predict(x_valid)

print('Training RNA')
kernel_initializer = 'normal'
activation = 'relu'
loss = 'binary_crossentropy'
batch_size = 1500
neurons = 1536
dropout = 0.1
learning_rate = 0.001
beta_1 = 0.97
beta_2 = 0.97
decay  = 0.05
epochs = 150
classificador = Sequential()
classificador.add(Dense(units = neurons, activation = activation, 
                    kernel_initializer = kernel_initializer, input_shape = (x_train.shape[1],)))
classificador.add(Dropout(dropout))

classificador.add(Dense(units = neurons, activation = activation, 
                    kernel_initializer = kernel_initializer))
classificador.add(Dropout(dropout))

classificador.add(Dense(units = 1, activation = 'sigmoid'))

opt = keras.optimizers.Adam(learning_rate=learning_rate, decay=decay, beta_1 = beta_1 , beta_2 = beta_2)

classificador.compile(optimizer = opt, loss = loss,
                  metrics = ['binary_accuracy'])

classificador.fit(x_train, y_train, batch_size = batch_size, epochs = epochs)

qtd_param = classificador.count_params()

print('Number of Parameters: ', qtd_param)

print('Calculating the Reduced Ensemble F1-Score...')

previsoes_rna = classificador.predict(x_valid)
previsoes_rna = (previsoes_rna > 0.5)
previsoes_num_rna = []
for i in previsoes_rna:
    if i:
        previsoes_num_rna.append(1)
    else:
        previsoes_num_rna.append(0)
previsoes_rna = np.array(previsoes_num_rna)

ensemble_reduced = []
tam = len(previsoes_rna)
for i in range(tam):
    if previsoes_rna[i] + previsoes_nb[i] + previsoes_svc[i] > 1:
        ensemble_reduced.append(1)
    else:
        ensemble_reduced.append(0)
ensemble_reduced = np.array(ensemble_reduced)

tp, fp, tn, fn = metric_calc(ensemble_reduced, y_valid)

precisao = f1_score(y_valid, ensemble_reduced)
print('Reduced Ensemble F1-Score: ' , precisao)
acc = accuracy_score(y_valid, ensemble_reduced)
print('Reduced Ensemble  Acuraccy:' , acc)
rec = recall_score(y_valid, ensemble_reduced)
print('Reduced Ensemble  Recall:' , rec)
roc = roc_auc_score(y_valid, ensemble_reduced)
print('Reduced Ensemble  ROC AUC:' , roc)
kappa = cohen_kappa_score(y_valid, ensemble_reduced)
print('Reduced Ensemble Kappa:' , roc)
prec = precision_score(y_valid, ensemble_reduced)
print('Reduced Ensemble Precision:' , prec)
specificity = tn / (tn+fp)
print('Reduced Ensemble Specificity:' , specificity)