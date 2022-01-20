#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Chagging code for extend paper on 14/01/2022;

Created on Thu May 13 11:13:38 2021

@author: yurifarod, Elwyslan
"""


from keras.layers import Dropout, Dense
from keras.models import Sequential
from sklearn.metrics import f1_score
from pathlib import Path
import numpy as np
import pandas as pd
from tensorflow import keras
import timeit

start = timeit.default_timer()

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

'''
dropout = 0.1
epochs = 150
kernel_initializer = 'normal'
activation = 'relu'
loss = 'binary_crossentropy'
neurons = 2560
learning_rate = 0.001
batch_size = 250
'''
dropout = 0.1
epochs = 500
kernel_initializer = 'normal'
activation = 'relu'
loss = 'binary_crossentropy'
neurons = 2048
learning_rate = 0.0001
batch_size = 250


classificador = Sequential()
classificador.add(Dense(units = neurons, activation = activation, 
                    kernel_initializer = kernel_initializer, input_shape = (x_train.shape[1],)))
classificador.add(Dropout(dropout))

classificador.add(Dense(units = neurons, activation = activation, 
                    kernel_initializer = kernel_initializer))
classificador.add(Dropout(dropout))

classificador.add(Dense(units = 1, activation = 'sigmoid'))

opt = keras.optimizers.Adam(learning_rate=learning_rate, decay=0.0001)

classificador.compile(optimizer = opt, loss = loss,
                  metrics = ['binary_accuracy'])

classificador.fit(x_train, y_train, batch_size = batch_size, epochs = epochs)

qtd_param = classificador.count_params()

print(qtd_param)

#Aqui fazemos a previsão
previsoes = classificador.predict(x_valid)
previsoes = (previsoes > 0.5)

#Agora vamos medir a acurácia da rede
precisao = f1_score(y_valid, previsoes)

previsoes_saida = pd.DataFrame(previsoes)
previsoes_saida.to_csv('rna_previsoes_%d.csv' % epochs)

print('F1-Score: ', precisao)
stop = timeit.default_timer()
print('Time: ', stop - start)  
