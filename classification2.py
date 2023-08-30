# -*- coding: utf-8 -*-
"""
Created on Mon Jul 17 00:05:52 2023

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
data=pd.read_csv('income(1).csv')
data1= data.copy(deep=True)
data1.columns
data1.isnull()
data1.isnull().sum()
data1.describe()
data1.describe(include='O')
data1['JobType'].value_counts()
print(np.unique(data1['JobType']))
data1['occupation'].value_counts()
print(np.unique(data1['occupation']))
data2=pd.read_csv('income(1).csv',na_values=' ?')
data2.isnull().sum()
data2=data2.dropna(axis=0)
 
''' CLassification 2'''


###### Logistic Regression##############
# logistic regression is a machine learnig algorith that is used to predict the
# probablity of a categorical dependent variable. so using logistic regression 
# we made a classifier model based on the given data


# step1 : RE indexing the salary status as 0 and 1 because the ml algorithm does
# work with the categorical data
print(np.unique(data2['SalStat']))
data2['SalStat']=data2['SalStat'].map({' less than or equal to 50,000':0,' greater than 50,000':1})
print(data2['SalStat']) 
'''here we used map function and giving value using dictionay'''
# so the method we used here is integer encoding, so that we can invert the encoding
# the data later and get labels back to integer values such as in the case of making
# prediction and using the pandas function to make dummies.using this we can convert
# the categorical variables into dummy variables which is called as one hot encoding.
# it refers to splitting the columns which has the categorical data to many column 
# depend on the no of categories present in the column. 
new_data=pd.get_dummies(data2,drop_first=True) 
#drop_first=True is important to use, as it helps in reducing the extra column
# created during dummy variable creation. Hence it reduces the correlations 
#created among dummy variables.
#  Example :Let’s say we have 3 types of values in Categorical column and 
#  we want to create dummy variable for that column. If one variable is not 
#  furnished and semi_furnished, then It is obvious unfurnished. So we do not
# need 3rd variable to identify the unfurnished. 
''' so now we mapped all the string values to integer values ,so that they can 
work with any machine learning algorithm'''
# next we select the features where we need to divide the given column into two 
# types one is the independent variable and the second is the dependent variable
# dependent variable denoted by 'y' , independent as 'x'
# for that we are going to store column as list, by acessing the column names
column_list=list(new_data.columns)
print(column_list)

''' so now we need to seperate input variables from the data'''
features=list(set(column_list)-set(['SalStat']))
print(features)

'''storing the output values, so the .values is used to extract the data from the
data frame or a vector'''
y=new_data['SalStat'].values
print(y)
 '''simlarly for features '''
x=new_data[features].values
''' so x and y contain the integer values
next we are goint to split the data into traing and test data, so that we create
a model in the traing set and test the model on the test data, this can be done 
train test split command
the input parameter for the train and test parameter would be x and y ,where x 
represents the input values and y represent the output values here test_size=.3
represt the proportion of data taken as the test data, the next one is
random_state i have given an integer 0 , so here random state is the seed used 
by the random number generator so each and every time you run this line while
sampling same set of sample will be chosen , if you have not set the random seed
then the different set of sample will be chosen for the analysis '''
train_x,test_x,train_y,test_y=train_test_split(x,y,test_size=0.3,random_state=0)



''' next step we are going to classify logistic regression classifier'''
logistic=LogisticRegression()


''' now we can fit the model in the train set using the fit function'''
logistic.fit(train_x,train_y)
'''so , we used logistic regrssion to fit the model
so now we fit the model , so you can now extract some of the attribute from the 
logisticregression classify model '''
logistic.coef_ # coefficient 
logistic.intercept_
''' so now we have to predict the data on the test data so that we can know how
the model is performing'''
### prediction
prediction = logistic.predict(test_x)
print(prediction)
'''by putting the value of x we can find the value of y'''
''' so now we have evaluate the model using the confusion matrix , it is the 
table that is used to evaluate the performance of the classification model,
the confusion matrix output gives you the output of no. of correct prediction 
and no. of wrong prediction and it will sum up values class wise'''
confusion_matrix= confusion_matrix(test_y,prediction)
print(confusion_matrix)
'''so you have the four values the diagonal value give total no of correct 
value and off diagonal values give no of wrongly classified sample
column wise show the prediction and row wise shows the actual classes
'''

''' so the model does not classify all the observation correctly using the accuracy
we can get the idea how accurate is the model'''
accuracy = accuracy_score(test_y,prediction)
print(accuracy)

''' we can also get the misclassified values from the prediction'''
print('misclassified sample=%d'%(test_y !=prediction).sum())
''' here you giving the condition'''
########################################################################

'''so now we built a logistic regression model but you can improve the accuracy
of the model by reducing the no. of misclassified sample so of the method is to
remove the no of insignificant variables  when we explored more on the data we find
there are some insignificant variables'''
### REMOVING INSIGNIFICANT VARIABLES
data3= pd.read_csv('income(1).csv',na_values=[' ?'])
data3=data3.dropna(axis=0)
data3['SalStat']=data3['SalStat'].map({' less than or equal to 50,000':0,' greater than 50,000':1})
print(data3['SalStat']) 

cols=['gender','nativecountry','race','JobType']
''' these are the variable that are insignificant so we drop them'''
new_data1=data3.drop(cols,axis=1)
new_data1=pd.get_dummies(new_data1,drop_first=True)
column_list1=list(new_data1.columns)
features1=list(set(column_list1)-set(['SalStat']))
y1=new_data1['SalStat'].values
x1=new_data1[features1].values
print(x1)
train_x1,test_x1,train_y1,test_y1=train_test_split(x1,y1,test_size=0.3,random_state=0)
logistic.fit(train_x1,train_y1)
prediction1=logistic.predict(test_x1)
print(prediction1)
logistic.intercept_
confusion_matrix1= confusion_matrix(test_y1,prediction1)
print(confusion_matrix1)
accuracy1=accuracy_score(test_y1,prediction1)
print(accuracy1)
print(accuracy)
print('misclassified sample=%d'%(test_y1 !=prediction1).sum())