#https://towardsdatascience.com/step-by-step-vgg16-implementation-in-keras-for-beginners-a833c686ae6c

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 13 11:13:38 2021

@author: yurifarod
"""

import keras
import os
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPool2D, Flatten
import timeit
from PIL import Image
start = timeit.default_timer()

diretorio = './train/fold_0/fold_0/all/'
x_train = []
y_train = []
for diretorio, subpasta, arquivos in os.walk(diretorio):
    for arquivo in arquivos:
        dir_img = diretorio + arquivo
        img = Image.open(dir_img)
        img = img.resize((450,450))
        img =np.asarray(img)
        x_train.append(img)
        y_train.append(1)
        break

diretorio = './train/fold_1/fold_1/all/'
for diretorio, subpasta, arquivos in os.walk(diretorio):
    for arquivo in arquivos:
        dir_img = diretorio + arquivo
        img = Image.open(dir_img)
        img = img.resize((450,450))
        img =np.asarray(img)
        x_train.append(img)
        y_train.append(1)

diretorio = './train/fold_2/fold_2/all/'
for diretorio, subpasta, arquivos in os.walk(diretorio):
    for arquivo in arquivos:
        dir_img = diretorio + arquivo
        img = Image.open(dir_img)
        img = img.resize((450,450))
        img =np.asarray(img)
        x_train.append(img)
        y_train.append(1)

diretorio = './train/fold_2/fold_2/hem/'
for diretorio, subpasta, arquivos in os.walk(diretorio):
    for arquivo in arquivos:
        dir_img = diretorio + arquivo
        img = Image.open(dir_img)
        img = img.resize((450,450))
        img =np.asarray(img)
        x_train.append(img)
        y_train.append(0)

diretorio = './train/fold_1/fold_1/hem/'
for diretorio, subpasta, arquivos in os.walk(diretorio):
    for arquivo in arquivos:
        dir_img = diretorio + arquivo
        img = Image.open(dir_img)
        img = img.resize((450,450))
        img =np.asarray(img)
        x_train.append(img)
        y_train.append(0)

diretorio = './train/fold_0/fold_0/hem/'
for diretorio, subpasta, arquivos in os.walk(diretorio):
    for arquivo in arquivos:
        dir_img = diretorio + arquivo
        img = Image.open(dir_img)
        img = img.resize((450,450))
        img =np.asarray(img)
        x_train.append(img)
        y_train.append(0)
        
x_train = np.array(x_train)
x_train = x_train.astype('float32') 
x_train /= 255

y_train = np.array(y_train)

model = Sequential()

model.add(Conv2D(input_shape=(450, 450, 3), filters=64, kernel_size=(3,3),
                  padding='same', activation='relu'))

model.add(Conv2D(filters=64,kernel_size=(3,3),padding="same", activation="relu"))

model.add(MaxPool2D(pool_size=(2,2),strides=(2,2)))

model.add(Conv2D(filters=128, kernel_size=(3,3), padding="same", activation="relu"))

model.add(Conv2D(filters=128, kernel_size=(3,3), padding="same", activation="relu"))

model.add(MaxPool2D(pool_size=(2,2),strides=(2,2)))

model.add(Conv2D(filters=256, kernel_size=(3,3), padding="same", activation="relu"))

model.add(Conv2D(filters=256, kernel_size=(3,3), padding="same", activation="relu"))

model.add(Conv2D(filters=256, kernel_size=(3,3), padding="same", activation="relu"))

model.add(MaxPool2D(pool_size=(2,2),strides=(2,2)))

model.add(Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))

model.add(Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))

model.add(Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))

model.add(MaxPool2D(pool_size=(2,2),strides=(2,2)))

model.add(Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))

model.add(Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))

model.add(Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))

model.add(MaxPool2D(pool_size=(2,2),strides=(2,2)))

model.add(Flatten())

model.add(Dense(units=512,activation="relu"))

model.add(Dense(units=512,activation="relu"))

model.add(Dense(units=1, activation="softmax"))

opt = keras.optimizers.Adam(learning_rate=0.9)

model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])

model.fit(x_train,
          y_train,
          batch_size=20,
          epochs = 100)

stop = timeit.default_timer()
print('Time: ', stop - start)  

qtd_param = model.count_params()
print('Parametros da Rede: ', qtd_param)