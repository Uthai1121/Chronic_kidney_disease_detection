# -*- coding: utf-8 -*-
"""Kidney.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1oEti91eLvrwd0v4wc9DJW_xSIrtZj4Pn
"""

import pandas as pd
import numpy as np
from collections import Counter as c
import matplotlib.pyplot as plt
import seaborn as sns
import missingno as msno
from sklearn.metrics import accuracy_score,confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LinearRegression
import pickle

data=pd.read_csv("kidney_disease.csv")
data

data.drop(["id"],axis=1,inplace=True)

data

data.columns

data.columns=['age','blood_pressure','specific_gravity','albumin',
              'sugar','red_blood_cells','pus_cell','pus_cell_clumps','bacteria',
              'blood glucose random','blood_urea','serum_creatinine','sodium','potassium',
              'hemoglobin','packed_cell_volume','white_blood_cell_count','red_blood_cell_count',
              'hypertension','diabetesmellitus','coronary_artery_disease','appetite',
              'pedal_edema','anemia','class'] # manually giving the name  of the columns
data.columns

data['class'].unique()

data['class']=data['class'].replace("ckd\t","ckd")
data['class']

catcols=set(data.dtypes[data.dtypes=='O'].index.values)
print(catcols)

for i in catcols:
  print("Columns :",i)
  print(c(data[i]))
  print('*'*120+'\n')

catcols.remove('red_blood_cell_count')
catcols.remove('packed_cell_volume')
catcols.remove('white_blood_cell_count')
print(catcols)

contcols=set(data.dtypes[data.dtypes!='O'].index.values)
print(contcols)

for i in contcols:
  print("Continuos columns :",i)
  print(c(data[i]))
  print('*'*120+'\n')

contcols.remove('specific_gravity')
contcols.remove('albumin')
contcols.remove('sugar')

print(contcols)

contcols.add('red_blood_cell_count')
contcols.add('packed_cell_volume')
contcols.add('white_blood_cell_count')
print(contcols)

catcols.add('specific_gravity')
catcols.add('albumin')
catcols.add('sugar')
print(catcols)

data['coronary_artery_disease']=data.coronary_artery_disease.replace('\tno','no')
c(data['coronary_artery_disease'])

data['diabetesmellitus']=data.diabetesmellitus.replace(to_replace={'\tno':'no','\tyes':'yes',' yes':'yes'})
c(data['diabetesmellitus'])

data.packed_cell_volume=pd.to_numeric(data.packed_cell_volume, errors='coerce')
data.white_blood_cell_count=pd.to_numeric(data.white_blood_cell_count, errors='coerce')
data.red_blood_cell_count=pd.to_numeric(data.red_blood_cell_count, errors='coerce')

data['blood glucose random'].fillna(data['blood glucose random'].mean(),inplace=True)
data['blood_pressure'].fillna(data['blood_pressure'].mean(),inplace=True)
data['blood_urea'].fillna(data['blood_urea'].mean(),inplace=True)
data['hemoglobin'].fillna(data['hemoglobin'].mean(),inplace=True)
data['packed_cell_volume'].fillna(data['packed_cell_volume'].mean(),inplace=True)
data['potassium'].fillna(data['potassium'].mean(),inplace=True)
data['red_blood_cell_count'].fillna(data['red_blood_cell_count'].mean(),inplace=True)
data['serum_creatinine'].fillna(data['serum_creatinine'].mean(),inplace=True)
data['sodium'].fillna(data['sodium'].mean(),inplace=True)
data['white_blood_cell_count'].fillna(data['white_blood_cell_count'].mean(),inplace=True)

data['age'].fillna(data['age'].mode()[0],inplace=True)
data['hypertension'].fillna(data['hypertension'].mode()[0],inplace=True)
data['pus_cell_clumps'].fillna(data['pus_cell_clumps'].mode()[0],inplace=True)
data['appetite'].fillna(data['appetite'].mode()[0],inplace=True)
data['albumin'].fillna(data['albumin'].mode()[0],inplace=True)
data['pus_cell'].fillna(data['pus_cell'].mode()[0],inplace=True)
data['red_blood_cells'].fillna(data['red_blood_cells'].mode()[0],inplace=True)
data['coronary_artery_disease'].fillna(data['coronary_artery_disease'].mode()[0],inplace=True)
data['bacteria'].fillna(data['bacteria'].mode()[0],inplace=True)
data['anemia'].fillna(data['anemia'].mode()[0],inplace=True)
data['sugar'].fillna(data['sugar'].mode()[0],inplace=True)
data['diabetesmellitus'].fillna(data['diabetesmellitus'].mode()[0],inplace=True)
data['pedal_edema'].fillna(data['pedal_edema'].mode()[0],inplace=True)
data['pedal_edema'].fillna(data['pedal_edema'].mode()[0],inplace=True)

for i in catcols:
  print("LABEL ENCODING OF:",i)
  LEi = LabelEncoder() # creating an object of LabelEncoder
  print(c(data[i])) #getting the classes values before transformation
  data[i] = LEi.fit_transform(data[i])# trannsforming our text classes to numerical values
  print(c(data[i])) #getting the classes values after transformation
  print("*"*100)

data

x=data.iloc[:,0:25]
y=data.iloc[:,24]



x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=2)#train test split
print(x_train.shape)
print(y_train.shape)
print(x_test.shape)
print(y_test.shape)

from sklearn.ensemble import RandomForestRegressor
rf = RandomForestRegressor()

from sklearn.linear_model import LogisticRegression
lgr = LogisticRegression()
lgr.fit(x_train,y_train)

y_pred = lgr.predict(x_test)

print(y_pred)

from sklearn.metrics import r2_score
acc= r2_score(y_pred,y_test)
acc

import pickle
pickle.dump(lgr,open('lgr.pkl','wb'))

conf_mat = confusion_matrix(y_test,y_pred)
conf_mat

x_train.shape

data.columns