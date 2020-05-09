# -*- coding: utf-8 -*-
"""
Created on Sun Jan 27 17:45:35 2019

@author: Sahil Nathani
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn import cross_validation
from sklearn.svm import SVC

le = LabelEncoder()#Label encoder

df = pd.read_csv('Flags.csv')

Continent = {1: 'N. America', 2: 'S. America', 3 : 'Europe', 4 : 'Africa', 5 : 'Asia', 
             6 : 'Oceania'}

df['Continent'] = df['Continent'].map(Continent)
df['Continent'] = le.fit_transform(df['Continent'])

Hemisphere = {1: 'NE', 2: 'SE', 3 : 'SW', 4 : 'NW'}
df['Hemisphere'] = df['Hemisphere'].map(Hemisphere)
df['Hemisphere'] = le.fit_transform(df['Hemisphere'])

Languag = {1:'English', 2:'Spanish', 3:'French', 4:'German', 5:'Slavic', 6:'Indo-European', 
           7:'Chinese', 8:'Arabic', 9:'Jap/Turk/Finn', 10:'Others'}

df['Language'] = df['Language'].map(Languag)
df['Language'] = le.fit_transform(df['Language'])

Religion = {0:'Catholic', 1:'Other Christian', 2:'Muslim', 3:'Buddhist', 4:'Hindu', 5:'Ethnic',
            6:'Marxist', 7:'Others'}

df['Religion'] = df['Religion'].map(Religion)
df['Religion'] = le.fit_transform(df['Religion'])

cols = ['Hemisphere', 'Continent', 'Language', 'Religion']

df['Topleft'] = le.fit_transform(df['Topleft'])
df['Botright'] = le.fit_transform(df['Botright'])
df['Hue'] = le.fit_transform(df['Hue'])

y = np.array(df['Language'])

x = np.array(df.drop(['Language', 'Country', 'Id', 'Hemisphere', 'Area', 'Population'], 1))

x_train, x_test, y_train, y_test = cross_validation.train_test_split(x, y, test_size=0.2)

clf = SVC(kernel='rbf', C=1000, gamma='auto')
clf.fit(x_train, y_train)
accuracy = clf.score(x_test, y_test)
print(accuracy*100)

'''
#Plot to determine distribution of countries hemisphere wise
plt.figure(figsize=(9, 9))
sns.countplot(x='Continent', hue='Hemisphere', data=df)
plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)

#Plot to determine distribution of countries language wise
plt.figure(figsize=(10, 16))
plt.subplot(2, 1, 1)
sns.countplot(x='Language', hue='Continent', data=df)
plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)

#Plot to determine distribution of countries Religion wise
plt.subplot(2, 1, 2)
sns.countplot(x='Religion', hue='Continent', data=df)
plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
'''

#Bars in Flags
'''
plt.figure(figsize=(24, 12))
for _, z in zip(cols, range(1, 5)):
    plt.subplot(2, 2, z)
    sns.countplot(x='Bars', hue=_, data=df)
    plt.xlabel((_+' '+'Wise'))
    plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
'''

#Stripes in Flags     
'''     
plt.figure(figsize=(25, 48))
for _, z in zip(cols, range(1, 5)):
    plt.subplot(4, 1, z)
    sns.countplot(x='Stripes', hue=_, data=df)
    plt.xlabel((_+' '+'Wise'))
    plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
  

print(df[(df['Language']=='German') & (df['Bars']==0)])
'''
'''
#Language vs Religion
plt.figure(figsize=(18, 18))
sns.countplot(x='Religion', hue='Language', data=df)
plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
'''

#Colors in Flags
'''
plt.figure(figsize=(24, 24))
for _, z in zip(cols, range(1, 5)):
    plt.subplot(4, 1, z)
    sns.countplot(x='Colors', hue=_, data=df)
    plt.xlabel((_+' '+'Wise'))
    plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
'''

colors = ['Red', 'Green', 'Blue', 'Gold', 'White', 'Black', 'Orange']

#Plot for TopLeft and Bottomright Color
'''
plt.figure(figsize=(25, 48))
for _, z in zip(cols, range(1, 5)):
   plt.subplot(4, 1, z)
   sns.countplot(x='Hue', hue=_, data=df)
   plt.xlabel((_+' '+'Wise'))
   plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
   
print(df[(df['Religion']=='Buddhist') & (df['Triangle']==1)])
'''   