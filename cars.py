# -*- coding: utf-8 -*-
"""
Created on Sat Aug 15 17:06:55 2020

@author: 91880
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
cars=pd.read_csv("cars_sampled.csv")
cars.columns
cars.shape
cars.head()
cars.info()
description=cars.describe()
cars.head()
cars.columns[cars.isnull().any()]

#The missing values are in columns 'vehicleType', 'gearbox', 'model', 'fuelType', 'notRepairedDamage'

cars.groupby(by=['brand']).price.max().sort_values()
#Max price is of ford i.e 12345678
#Min price is of daewoo i.e 2799
cars['price'].value_counts()
cars1=cars.copy(deep=True)

#Making two separate data on the basis of price

car_low_price=cars1.loc[cars1['price']<100]
car_data_new=cars1.loc[cars1['price']>=100]
car_data_new.shape

#Saving it to separate csv file

car_low_price.to_csv('car_price_less_than_100.csv',index=False)
car_data_new.to_csv('car_price_greater_than_100.csv',index=False)

#Splitting the data

from sklearn.model_selection import train_test_split
train,test=train_test_split(car_data_new,test_size=0.2073957,random_state=0)
train_new=pd.concat([train,car_low_price])
train.to_csv('train_without_price_missing.csv',index=False)
train_new.to_csv('train_with_price_missing.csv',index=False)
test.to_csv('test_data.csv',index=False)

#Reading the train data that includes vague price values to impute X's

train_data=pd.read_csv('train_with_price_missing.csv')
train_data.head()

#Cheching for randomness of data

import missingno as msno
msno.bar(train_data)
msno.heatmap(train_data)
msno.matrix(train_data)
corr_matrix=train_data.corr()
sns.heatmap(corr_matrix,annot=True)
plt.show

#Imputing missing values with term 'missing'

train_data.isnull().sum()
train_data['model'].fillna('missing_model',inplace=True)
train_data['vehicleType'].fillna('missing_vehicle',inplace=True)
train_data['gearbox'].fillna('missing_gear',inplace=True)
train_data['fuelType'].fillna('missing_fuel',inplace=True)
train_data['notRepairedDamage'].fillna('missing_damage',inplace=True)
train_data.isnull().sum()

#saving the imputed data

train_data.to_csv('train_data_missing1.csv',index=False)

data2=pd.read_csv('train_data_missing1.csv')

#Finding the Outliers

for i in data2.columns:
    if (data2[i].dtype=='O'):
       print( data2[i].value_counts(sort=False,normalize=True))

#The variables 'Offertype' and 'seller' can be dropped from the data as it have only one significant category

data2.drop("seller",inplace=True,axis=1)
data2.drop("offerType",inplace=True,axis=1)
data2.info()

#box plot

import seaborn as sns
sns.boxplot(x=data2['abtest'],y=data2['price'])

#percentile

np.percentile(data2['price'],99.4)
#4.5 %of the data is below 100
