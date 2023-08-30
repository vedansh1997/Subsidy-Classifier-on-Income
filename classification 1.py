# -*- coding: utf-8 -*-
"""
Created on Wed Jul 12 23:51:51 2023

@author: lenovo
"""

import os
import pandas as pd # to work with the dataframes
import numpy as np # to perform numerical operations
import seaborn as sns # to visiualize the data
from sklearn.model_selection import train_test_split # to partition the data 
# from package sklearn , model_selection is sub package under sklearn,now we going
# to train_test_split from sk learn
#
from sklearn.linear_model import LinearRegression # import logistic regression from 
# sk learn library
from sklearn.metrics import accuracy_score,confusion_matrix
# to look for performance matrix we're going to import accuracy and confusion matrix
# from sklearn matrix

# importing data
data_income= pd.read_csv('income(1).csv')
# creating copy of orignal data
data=data_income.copy(deep=True)

# Exploratory data analytics 
 # 1. getting to know the data
 # here we basicall see what type of variables we are going to deal with 
 # 2. data processing (Missing values)
 # deal with the missing values like how to identify missing values and how we deal it.
 # 3. cross tables and visiualization
 # we understand the data with viziualization and cross tables
 
 
## getting to know data type

# to check variable data type
print(data.info())
# now we see the data types of each variables
# now we check whether there is any missing values in the data frame
data.isnull()
# so it is difficult to read whole data so we used ,so we take a sum of missing 
# values in each column so that we can identify the missing values
data.isnull().sum()
# now its time to understand the data by getting descriptive statistics out of it.
# in our data frame both categorical and numerical data are there the decriptive 
# statistics give great insight to the shape of each variable 
description = data.describe()
print(description)
# capital gain : it the sale of property or invesment , when an sale price exceeds
# purchase price similarly for capital loss
#**** summary of categorical variables
# we would be intrested in the frequency of each variables by seeting a command
summary_cate=data.describe(include='O')
# here capital o is related to object type
print(summary_cate)
# unique tell how many unique categories avilable
# from this we came to know, how many unique values are there
# but we are intrested in no of unique categories under a variable
# so that it gives an idea what are the categories are there in a particular variables
# so in that case value_counts() gives the better representation.
data['JobType'].value_counts()
# this gives us better representation by listing the avilable categories and their
# frequencies, so here ? is an problem because python read nan values as missing values
data['occupation'].value_counts()
# to see how these special characters are we used the unique function
print(np.unique(data['JobType']))
print(np.unique(data['occupation']))
# so basically unique is from numpy librabry and on that i specify the data from 
# data frame data
# so you can observed that there is space before the question mark that's why we
# unique function
# now we remove '?' with nan values
data=pd.read_csv('income(1).csv',na_values=' ?')
### now the next step is to identify the missing data and then we identifying the
# missing value pattern
data.isnull().sum()
# to deal with the missing data , let see in a particular row either one of the 
# column is missing or both column value are missing so we subset the row at 
# least one column is missing in a row
missing = data[data.isnull().any(axis=1)]
# it consider any 1 missing in a column in a row
# axis=1 means to consider at least one column where value is missing
# after observing the data of missing it became clear tha where job type is missing
# the occupation is also missing
# so there is special category never worked  in job type and nan in job type
# if the job type is never worked so the occupation is never filled so thats that
# portion give you nan values

''' points to remember :
    1 missing values in the JobType    = 1809
    2 missing values in the occupation = 1816
    3 there are 1809 row where two specific column are nan
      i.e JobType and occupation
    4 there are still 7 occupation unfilled because the never worked
    '''
# so inthis case we can delete the missing data or put the alternative value
# if the data are not missing at random then we have to model the mechanism 
# that produce missing values as well as the relationship context which is very
# comlex in this context so here we here we remove all the rows where missing
# values is consider
data1= data.dropna(axis=0)
# relationship with independent variables using correlation measure
correlation=data1.corr()
# so you can observe none of the values are nearer to one that means none of the
# variables are correlated with each other
# now we look for the categorical variables to look at the relationship


###### cross tables and visiualization ######
# extracting the column names
data1.columns
'' 'Gender Proporttion Table using the crosstable function'''
gender = pd.crosstab(index=data1['gender'], columns='count',normalize=True)
print(gender)
# setting normalize = True you will get the proportion table,give the variable of
# intrest in the index 

''' gender vs salary stat table  using the crosstab function'''
gender_salstat=pd.crosstab(index=data1['gender'], columns=data1['SalStat'],\
                           margins=True,normalize='index')
    print(gender_salstat)
### by setting normalize= 'index' , i am going to get the row proportional equal
## to index , row proportional is equal to 1
## frequency stats of salary status
Salstat=sns.countplot(data=data1,x='SalStat')
Salw=sns.countplot(data1['SalStat']) # shows error

data1['SalStat'].value_counts()
''' so as you can see 25% of people earned less than 50000
    and 75% of people earn more 50000$ '''
### Histogram Of Age
sns.displot(data=data1,x='age',bins=10,kde=False)
sns.displot(data1['age'],bins=10,kde=False)
'''from histogram it is clear that the people age 20-45 are high in frequency'''
### similarly we genrate cross table and plots to understand other variables
## so if you want to do a bivariate analysis how age is affecting the salary status
## then we can do a box plot to how salary varying acc  to age
sns.boxplot('Salstat','age',data=data1)
sns.boxplot(data=data1,x='SalStat',y='age')
data1.groupby('SalStat')['age'].median()
''' people between 20 to 35 earn < 50000
    people between 35 to 50 earn >=50000 '''
# similary you can create crosstable and plots to create a relation ship between 
## other variables


''' now  i using exploratory data analysis showing you the same relation between 
    the variables using the same command '''
### JOB TYPE VS SALARY STATUS
pd.crosstab(index=data1['JobType'], columns=data1['SalStat'])
sns.countplot(data=data1,x='JobType',hue='SalStat')
#hence from the crosstable we get to know 56% of self employed people earn grater
# than 50000$ ,hence is an important variable in avoiding misuse of the subsadies
## Education and SalStatus
sns.countplot(data=data1,x='EdType',hue='SalStat')
pd.crosstab(index=data1['EdType'], columns=data1['SalStat'])
## From the table we can see that the people who have done their docterate ,masters
## and prof school are more likely to earn more than 50000$ , hence an influencing
# variable in avoiding the misuse of the subsidies
## Occupation and SalStatus
sns.countplot(data=data1,y='occupation',hue='SalStat')
pd.crosstab(index=data1['occupation'], columns=data1['SalStat'],normalize='index')
# hence manager and profession are more likely to earn more than 50000$ ,
# hence an influencing variable in avoiding the misuse of the subsidies
sns.displot(data=data1,x='capitalgain')
sns.boxplot(data=data1,x='SalStat',y='hoursperweek',hue='SalStat')
data1.groupby('SalStat')['hoursperweek'].median()
 

''' CLassification 2'''


###### Logistic Regression##############
# logistic regression is a machine learnig algorith that is used to predict the
# probablity of a categorical dependent variable. so using logistic regression 
# we made a classifier model based on the given data

print(data1)
# step1 : RE indexing the salary status as 0 and 1 because the ml algorithm does
# work with the categorical data
data1['Salstat']=data1['SalStat'].map({'sala'})