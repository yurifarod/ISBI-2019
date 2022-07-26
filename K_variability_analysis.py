#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Chagging code for extend paper on 23/07/2022;

@author: yurifarod, Elwyslan
"""
import random
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import GaussianNB
from keras.layers import Dropout, Dense
from keras.models import Sequential
from tensorflow import keras
from sklearn.metrics import f1_score, accuracy_score, recall_score, roc_auc_score, cohen_kappa_score

reduce_factor = 1
f1s = []
acc = []
roc = []
rec = []
kappa = []

ann_f1s = []
ann_acc = []
ann_roc = []
ann_rec = []
ann_kappa = []

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
            
'''
100-FOLD VALIDATION
'''

for it in range(100):
    print('Reading Train Dataframe...')
    train_df = pd.read_csv(Path('feature-dataframes/AugmPatLvDiv_TRAIN-AllFeats_1612-Features_6405-images.csv'), index_col=0)
    print('Done Read Train Dataframe!')
    
    print('Reading Validation Dataframe...')
    valid_df = pd.read_csv(Path('feature-dataframes/PatLvDiv_TEST-AllFeats_1612-Features_607-images.csv'), index_col=0)
    print('Done Read Validation Dataframe!')
    
    print('Preparing Data...')
    
    new_size = valid_df.shape[1]
    
    for i in range(new_size):
        valid_df[valid_df.columns[i]] = clean_Dirt_Data(valid_df[valid_df.columns[i]])
        train_df[train_df.columns[i]] = clean_Dirt_Data(train_df[train_df.columns[i]])
    
    
    print('Reducing Valid Data...')
    v_size = len(valid_df.index)
    label = []
    for i in range(v_size):
        if(reduce_factor == random.randint(0, 100) ):
            label.append(i)
    
    valid_df = valid_df.drop(labels = label, inplace=False, axis=0)
    
    print('Reducing Training Data...')
    t_size = len(train_df.index)
    label = []
    for i in range(t_size):
        if(reduce_factor == random.randint(0, 100) ):
            label.append(i)
    
    train_df = train_df.drop(labels = label, inplace=False, axis=0)
    
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
    
    print('Calculating the Ensemble F1-Score...')
    
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
    new_size_x = valid_df.shape[0]
    for i in range(new_size_x):
        if previsoes_rna[i] + previsoes_nb[i] + previsoes_svc[i] > 1:
            ensemble_reduced.append(1)
        else:
            ensemble_reduced.append(0)
    ensemble_reduced = np.array(ensemble_reduced)
    
    ann_f1s.append(f1_score(y_valid, previsoes_rna))
    ann_acc.append(accuracy_score(y_valid, previsoes_rna))
    ann_rec.append(recall_score(y_valid, previsoes_rna))
    ann_roc.append(roc_auc_score(y_valid, previsoes_rna))
    ann_kappa.append(cohen_kappa_score(y_valid, previsoes_rna))
    
    f1s.append(f1_score(y_valid, ensemble_reduced))
    acc.append(accuracy_score(y_valid, ensemble_reduced))
    rec.append(recall_score(y_valid, ensemble_reduced))
    roc.append(roc_auc_score(y_valid, ensemble_reduced))
    kappa.append(cohen_kappa_score(y_valid, ensemble_reduced))

print('=======================================================')
print('==============RESULTADO FINAL==========================')
print('=======================================================')

#ANN SCORE
print('ANN DATA')
print('f1-score mean: ', np.mean(ann_f1s ))
print('acc mean: ', np.mean(ann_acc ))

print('ENSEMBLE DATA')
print('F1-SCORE:')
print('mean: ', np.mean(f1s ))
print('median: ', np.median(f1s ))
print('max: ', np.max(f1s ))
print('min: ', np.min(f1s ))
print('variance: ', np.var(f1s ))
print('std: ', np.std(f1s ))

print('ACURACCY:')
print('mean: ', np.mean(acc ))
print('median: ', np.median(acc ))
print('max: ', np.max(acc ))
print('min: ', np.min(acc ))
print('variance: ', np.var(acc ))
print('std: ', np.std(acc ))

print('REC:')
print('mean: ', np.mean(rec ))
print('median: ', np.median(rec ))
print('max: ', np.max(rec ))
print('min: ', np.min(rec ))
print('variance: ', np.var(rec ))
print('std: ', np.std(rec ))

print('ROC:')
print('mean: ', np.mean(roc ))
print('median: ', np.median(roc ))
print('max: ', np.max(roc ))
print('min: ', np.min(roc ))
print('variance: ', np.var(roc ))
print('std: ', np.std(roc ))

print('KAPPA:')
print('mean: ', np.mean(kappa ))
print('median: ', np.median(kappa ))
print('max: ', np.max(kappa ))
print('min: ', np.min(kappa ))
print('variance: ', np.var(kappa ))
print('std: ', np.std(kappa ))

#Escrevendo o arquivo com os dados das execucoes
import pandas
df = pandas.DataFrame(data={"ANN F1": ann_f1s, "ANN ACC": ann_acc, "ENS F1": acc, 'ENS ACC': f1s})
df.to_csv("./100-fold_cross_validation.csv", sep=',',index=False)
