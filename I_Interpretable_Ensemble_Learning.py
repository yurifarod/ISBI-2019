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
from keras.wrappers.scikit_learn import KerasClassifier
from keras.layers import Dropout, Dense
from keras.models import Sequential
from tensorflow import keras
from sklearn.feature_selection import SequentialFeatureSelector
import mlxtend
from mlxtend.feature_selection import SequentialFeatureSelector as SFS

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

for i in range(1613):
    valid_df[valid_df.columns[i]] = clean_Dirt_Data(valid_df[valid_df.columns[i]])
    train_df[train_df.columns[i]] = clean_Dirt_Data(train_df[train_df.columns[i]])

x_train, y_train = prepareData(train_df)
x_valid, y_valid = prepareData(valid_df)

print('Done Read Train and Validation data!')

#Avaliacao do NB
classificador_nb = GaussianNB(priors=None, var_smoothing=1e-9)
# Sequential Forward Selection(sfs)
sfs = SequentialFeatureSelector(classificador_nb, n_features_to_select = x_train.shape[1] * 0.15)
sfs.fit(x_train, y_train)
result_nb = sfs.get_support()

result_nb_out = pd.DataFrame(result_nb)
result_nb_out.to_csv('nb_interpretavel.csv')

#Avaliacao do SVC
classificador_svm = LinearSVC()
# Sequential Forward Selection(sfs)
sfs = SequentialFeatureSelector(classificador_svm, n_features_to_select = x_train.shape[1] * 0.15)
sfs.fit(x_train, y_train)
result_svc = sfs.get_support()

result_svc_out = pd.DataFrame(result_svc)
result_svc_out.to_csv('svc_interpretavel.csv')


# #Avaliacao da RNA
def criarRede(train_input):
    classificador = Sequential()
    classificador.add(Dense(units = 1536,
                            activation = 'relu', 
                            kernel_initializer = 'normal',
                            input_shape = (train_input.shape[1],)))
    classificador.add(Dropout(0.1))
    
    classificador.add(Dense(units = 1536, activation = 'relu', 
                            kernel_initializer = 'normal'))
    classificador.add(Dropout(0.1))
    
    classificador.add(Dense(units = 1, activation = 'sigmoid'))
    
    opt = keras.optimizers.Adam(learning_rate=0.001,
                                beta_1 = 0.97,
                                beta_2 = 0.97,
                                decay=0.05)
    
    classificador.compile(optimizer = opt, loss = 'binary_crossentropy',
                      metrics = ['binary_accuracy'])
    return classificador

keras.backend.clear_session()

# Wrap Keras nn and generating SFS object
class MakeModel(object):

    def __init__(self, X=None, y=None):
        pass

    def predict(self, X):
        y_pred = self.model.predict(X)
        y_pred = (y_pred > 0.5)
        return y_pred
    
    def fit(self, X, y):
        skwrapped_model = KerasClassifier(build_fn=criarRede,
                                          train_input=X,
                                          epochs=250,
                                          batch_size=1500,
                                          validation_split=2,
                                          verbose=0)
        self.model = skwrapped_model
        self.model.fit(X, y)
        return self.model

sffs = SFS(MakeModel(),
            k_features=(1, x_train.shape[1]),
            floating=True,
            clone_estimator=False,
            cv=0,
            n_jobs=1,
            scoring='accuracy')

# Apply SFS to identify best feature subset
sffs = sffs.fit(x_train, y_train)
result_rna_sub = sffs.subsets_
result_rna_id = sffs.k_feature_idx_