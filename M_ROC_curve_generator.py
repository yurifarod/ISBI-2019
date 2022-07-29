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
from sklearn.calibration import CalibratedClassifierCV
from sklearn.naive_bayes import GaussianNB
from keras.layers import Dropout, Dense
from keras.models import Sequential
from tensorflow import keras
from sklearn.metrics import f1_score, accuracy_score, recall_score, roc_auc_score, cohen_kappa_score
from sklearn import metrics
import matplotlib.pyplot as plt

reduce_factor = 15

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
prior_train_df = pd.read_csv(Path('feature-dataframes/AugmPatLvDiv_TRAIN-AllFeats_1612-Features_40000-images.csv'), index_col=0)

extra_train_df = pd.read_csv(Path('feature-dataframes/AugmPatLvDiv_VALIDATION-AllFeats_1612-Features_10000-images.csv'), index_col=0)

extra_train_df.index += len(prior_train_df.index)

frames = [prior_train_df, extra_train_df]

train_df = pd.concat(frames)
print('Done Read Train Dataframe!')
   
print('Reading Validation Dataframe...')
valid_df = pd.read_csv(Path('feature-dataframes/PatLvDiv_TEST-AllFeats_1612-Features_1503-images.csv'), index_col=0)
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
prob_nb = classificador.predict_proba(x_valid)

print('Training SVM')
classificador = LinearSVC()
classificador.fit(x_train, y_train)

previsoes_svc = classificador.predict(x_valid)

clf = CalibratedClassifierCV(classificador) 
clf.fit(x_train, y_train)
prob_svc = clf.predict_proba(x_valid)

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

print('Calculating the ROC curve...')

previsoes_rna = classificador.predict(x_valid)
prob_rna = previsoes_rna
previsoes_rna = (previsoes_rna > 0.5)
previsoes_num_rna = []
for i in previsoes_rna:
    if i:
        previsoes_num_rna.append(1)
    else:
        previsoes_num_rna.append(0)
previsoes_rna = np.array(previsoes_num_rna)

prev_ensemble = []
prob_ensemble = []
new_size_x = valid_df.shape[0]
for i in range(new_size_x):
    
    #pega as probabilidades
    lista = [prob_nb[i][0], prob_svc[i][0], prob_rna[i][0]]
    prob_ensemble.append(np.median(lista))
    
    if previsoes_rna[i] + previsoes_nb[i] + previsoes_svc[i] > 1:
        prev_ensemble.append(1)
    else:
        prev_ensemble.append(0)
prev_ensemble = np.array(prev_ensemble)

fpr, tpr, _ = metrics.roc_curve(y_valid,  prob_ensemble)
auc = metrics.roc_auc_score(y_valid, prob_ensemble)

#create ROC curve
plt.plot(fpr,tpr,label="AUC="+str(auc))
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.legend(loc=4)
plt.show()

