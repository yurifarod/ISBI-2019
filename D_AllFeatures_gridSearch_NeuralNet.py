#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 13 11:13:38 2021

@author: yurifarod, Elwyslan
"""

from keras.layers import Dropout, Dense
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import GridSearchCV
from keras.models import Sequential
from pathlib import Path
import numpy as np
import pandas as pd

ALL_LABEL = [1.0, 0.0]
HEM_LABEL = [0.0, 1.0]

def prepareData(train_df, valid_df):
    #Prepare Validation data
    y_valid = list(valid_df['cellType(ALL=1, HEM=-1)'].values)
    for i in range(len(y_valid)):
        if y_valid[i]==-1:
            y_valid[i] = HEM_LABEL
        elif y_valid[i]==1:
            y_valid[i] = ALL_LABEL
    y_valid = np.array(y_valid)
    x_valid = valid_df.drop(['cellType(ALL=1, HEM=-1)'], axis=1)
    for col in x_valid.columns:
        x_valid[col] = (x_valid[col] - train_df[col].mean()) / train_df[col].std() #mean=0, std=1
    x_valid = x_valid.values

    #Prepare Train data
    y_train = list(train_df['cellType(ALL=1, HEM=-1)'].values)
    for i in range(len(y_train)):
        if y_train[i]==-1:
            y_train[i] = HEM_LABEL
        elif y_train[i]==1:
            y_train[i] = ALL_LABEL
    y_train = np.array(y_train)
    x_train = train_df.drop(['cellType(ALL=1, HEM=-1)'], axis=1)
    for col in x_train.columns:
        x_train[col] = (x_train[col] - train_df[col].mean()) / train_df[col].std() #mean=0, std=1
    x_train = x_train.values
    return x_train, y_train, x_valid, y_valid

print('Reading Train Dataframe...')
train_df = pd.read_csv(Path('feature-dataframes/AugmPatLvDiv_TRAIN-AllFeats_1387-Features_7081-images.csv'), index_col=0)
print('Done Read Train Dataframe!')

print('Reading Validation Dataframe...')
valid_df = pd.read_csv(Path('feature-dataframes/AugmPatLvDiv_VALIDATION-AllFeats_1387-Features_2680-images.csv'), index_col=0)
print('Done Read Validation Dataframe!')

print('Preparing Data...')

x_train, y_train, x_valid, y_valid = prepareData(train_df=train_df, valid_df=valid_df)

y_train = np.argmax(y_train, axis=1)
y_valid = np.argmax(y_valid, axis=1)   
print('Done Read Train and Validation data!')
    
def criarRede(optimizer, loos, kernel_initializer, activation,
              neurons, hidden, dropout):
    classificador = Sequential()
    classificador.add(Dense(units = neurons, activation = activation, 
                        kernel_initializer = kernel_initializer, input_shape = (x_train.shape[1],)))
    classificador.add(Dropout(dropout))
    
    for i in range(hidden):
        classificador.add(Dense(units = neurons, activation = activation, 
                            kernel_initializer = kernel_initializer))
        classificador.add(Dropout(dropout))
    
    classificador.add(Dense(units = 1, activation = 'sigmoid'))
    classificador.compile(optimizer = optimizer, loss = loos,
                      metrics = ['binary_accuracy'])
    return classificador

classificador = KerasClassifier(build_fn = criarRede)
parametros = {'batch_size': [250, 750, 1000, 1500],
              'hidden' : [1, 2, 3, 4],
              'dropout' : [0.1, 0.25, 0.3, 0.5],
              'epochs': [10],
              #depois pica o resto
              'optimizer': ['adamax', 'adam', 'sgd'],
              'loos': ['binary_crossentropy'],
              'kernel_initializer': ['random_uniform', 'normal'],
              'activation': ['relu', 'prelu', 'sigmoid', 'softmax'],
              'neurons': [1024, 1536, 2048, 2560]}
grid_search = GridSearchCV(estimator = classificador,
                            param_grid = parametros,
                            scoring = 'accuracy')
grid_search = grid_search.fit(x_train, y_train)
melhores_parametros = grid_search.best_params_
melhor_precisao = grid_search.best_score_