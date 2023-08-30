# -*- coding: utf-8 -*-
"""
Created on Tue Jul 18 22:18:34 2023

@author: lenovo
"""

import pandas as pd
import numpy as np
import os
import seaborn as sns # to visiualize the data
from sklearn.model_selection import train_test_split # to partition the data 
from sklearn.linear_model import LogisticRegression# import logistic regression from 
# sk learn library
from sklearn.metrics import accuracy_score,confusion_matrix
# to look for performance matrix we're going to import accuracy and confusion matrix
# from sklearn matrix
from sklearn.neighbors import KNeighborsClassifier # importing the library of
# kNeighbours
#import matplot lib for plotting
import matplotlib.pyplot as plt
# to visiualize some output of the kNearest Neighbour classifier
'''storing the knearest neighbour classifier'''
KNN= KNeighborsClassifier(n_neighbors=5)
''' here n_neighbours =5 means it takes k =5'''
data=pd.read_csv('Income(1).csv',na_values=[' ?'])
data1=data.copy(deep=True)
data1=data1.dropna(axis=0)
data1['SalStat']=data1['SalStat'].map({' less than or equal to 50,000':0,' greater than 50,000':1})
new_data=pd.get_dummies(data1,drop_first=True)
column_list= list(new_data.columns)
features=list(set(column_list)-set(['SalStat']))
x=new_data[features].values
y=new_data['SalStat'].values
train_x,test_x,train_y,test_y=train_test_split(x,y,test_size=0.3,random_state=0)
KNN.fit(train_x,train_y)
# prediction
prediction = KNN.predict(test_x)
confusion_matrix= confusion_matrix(test_y,prediction)
print(confusion_matrix)
accuracy= accuracy_score(test_y,prediction)
print(accuracy)
print('misclassified sample=%d'%(test_y !=prediction).sum())

Misclassified_samples=[]
acc_uracy=list()
''' under the neighbours argument while we have building the model we have 
randomly fixed the value of k as 5. so here we are trying to calculate the error
of k between 1 and 20 by itertrating through for loop.'''
for i in range(1,20):
    knn= KNeighborsClassifier(n_neighbors=i)
    knn.fit(train_x,train_y)
    pre_dict=knn.predict(test_x)
    acc_uracy=accuracy_score(test_y,pre_dict)
    Misclassified_samples.append((test_y !=pre_dict).sum())
print(acc_uracy)
print(Misclassified_samples)
    