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
train_df = pd.read_csv(Path('feature-dataframes/AugmPatLvDiv_TRAIN-AllFeats_1612-Features_40000-images.csv'), index_col=0)
print('Done Read Train Dataframe!')

print('Reading Validation Dataframe...')
valid_df = pd.read_csv(Path('feature-dataframes/AugmPatLvDiv_VALIDATION-AllFeats_1612-Features_10000-images.csv'), index_col=0)
print('Done Read Validation Dataframe!')

print('Preparing Data...')

nrange = valid_df.shape[1]
for i in range(nrange):
    valid_df[valid_df.columns[i]] = clean_Dirt_Data(valid_df[valid_df.columns[i]])
    train_df[train_df.columns[i]] = clean_Dirt_Data(train_df[train_df.columns[i]])

x_train, y_train = prepareData(train_df)
x_valid, y_valid = prepareData(valid_df)

print('Done Read Train and Validation data!')
    
def criarRede(loos, kernel_initializer, activation,
              neurons, dropout, learning_rate,
              beta_1, beta_2, decay):
    classificador = Sequential()
    classificador.add(Dense(units = neurons, activation = activation, 
                        kernel_initializer = kernel_initializer, input_shape = (x_train.shape[1],)))
    classificador.add(Dropout(dropout))
    
    classificador.add(Dense(units = neurons, activation = activation, 
                        kernel_initializer = kernel_initializer))
    classificador.add(Dropout(dropout))
    
    classificador.add(Dense(units = 1, activation = 'sigmoid'))
    
    opt = keras.optimizers.Adam(learning_rate=learning_rate, beta_1 = beta_1 , beta_2 = beta_2, decay=decay)
    
    classificador.compile(optimizer = opt, loss = loos,
                      metrics = ['binary_accuracy'])
    return classificador

classificador = KerasClassifier(build_fn = criarRede)
parametros = {'batch_size': [1500],
              'dropout' : [0.1],
              'epochs': [150],
              #50, 100, 150, 200, 250
              'loos': ['binary_crossentropy'],
              'kernel_initializer': ['normal'],
              'activation': ['relu'],
              'neurons': [1536],
              'learning_rate': [0.01, 0.001, 0.005, 0.0001, 0.0005],
              'beta_1': [0.99, 0.98, 0.97],
              'beta_2': [0.99, 0.98, 0.97],
              'decay': [0.01, 0.05, 0.001, 0.005, 0.0001, 0.00005, 0.00001, 0.00005]
              }
grid_search = GridSearchCV(estimator = classificador,
                            param_grid = parametros,
                            scoring = 'accuracy',
                            cv = 2)
grid_search = grid_search.fit(x_train, y_train)
melhores_parametros = grid_search.best_params_
melhor_precisao = grid_search.best_score_