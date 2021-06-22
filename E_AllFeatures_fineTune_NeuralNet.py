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
from tensorflow import keras

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
    
def criarRede(loos, kernel_initializer, activation,
              neurons, dropout, learning_rate,
              beta_1, beta_2):
    classificador = Sequential()
    classificador.add(Dense(units = neurons, activation = activation, 
                        kernel_initializer = kernel_initializer, input_shape = (x_train.shape[1],)))
    classificador.add(Dropout(dropout))
    
    classificador.add(Dense(units = neurons, activation = activation, 
                        kernel_initializer = kernel_initializer))
    classificador.add(Dropout(dropout))
    
    classificador.add(Dense(units = 1, activation = 'sigmoid'))
    
    opt = keras.optimizers.Adam(learning_rate=learning_rate, beta_1 = beta_1 , beta_2 = beta_2)
    
    classificador.compile(optimizer = opt, loss = loos,
                      metrics = ['binary_accuracy'])
    return classificador

classificador = KerasClassifier(build_fn = criarRede)
parametros = {'batch_size': [250],
              'dropout' : [0.1],
              'epochs': [150],
              #50, 100, 150, 200, 250
              'loos': ['binary_crossentropy'],
              'kernel_initializer': ['normal'],
              'activation': ['relu'],
              'neurons': [2560],
              #'learning_rate': [0.001]
              'learning_rate': [0.01, 0.001, 0.005, 0.0001, 0.0005],
              'beta_1': [0.99, 0.98, 0.97],
              'beta_2': [0.99, 0.98, 0.97]
              }
grid_search = GridSearchCV(estimator = classificador,
                            param_grid = parametros,
                            scoring = 'accuracy',
                            cv = 2)
grid_search = grid_search.fit(x_train, y_train)
melhores_parametros = grid_search.best_params_
melhor_precisao = grid_search.best_score_