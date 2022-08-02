#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 26 18:56:34 2022

@author: yurifarod
"""

import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
from scipy.stats import kruskal, wilcoxon

classifiers = pd.read_csv(Path('100-fold_cross_validation.csv'), index_col=0)
ann_acc = classifiers["ANN ACC"].values.tolist()
ann_f1 = classifiers["ANN F1"].values.tolist()
ens_acc = classifiers["ENS ACC"].values.tolist()
ens_f1 = classifiers["ENS F1"].values.tolist()

reduced_classifier = pd.read_csv(Path('100-fold_cross_validation_red.csv'), index_col=0)
red_ens_acc = reduced_classifier["ENS ACC"].values.tolist()
red_ens_f1 = reduced_classifier["ENS F1"].values.tolist()

#Resolve a bug on the write
for i in range(100):
    if ann_acc[i] > 1:
        ann_acc[i] = ann_acc[i]/1000
    if ann_f1[i] > 1:
        ann_f1[i] = ann_f1[i]/1000
    if ens_acc[i] > 1:
        ens_acc[i] = ens_acc[i]/1000
    if ens_f1[i] > 1:
        ens_f1[i] = ens_f1[i]/1000

print('ACCURACY')
print(kruskal(red_ens_acc, ann_acc))
print(wilcoxon(red_ens_acc, ann_acc))

print('F1-SCORE')
print(kruskal(red_ens_f1, ann_f1))
print(wilcoxon(red_ens_f1, ann_f1))

  
# Creating plot 
# dt_acc = ([ann_acc, red_ens_acc])
# plt.boxplot(dt_acc)

dt_f1 = ([ann_f1, red_ens_f1])
plt.boxplot(dt_f1) 
  
# show plot 
plt.show()
