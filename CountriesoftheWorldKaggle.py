# -*- coding: utf-8 -*-
"""
Created on Sun Jan 13 11:11:40 2019

@author: Sahil Nathani
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

plt.style.use('fivethirtyeight')

df = pd.read_csv('Countries of the World.csv')


cols = ['Pop Density', 'Coastline', 'Net migration', 'Infant mortality', 'Literacy', 'Phones', 'Arable', 'Crops', 'Other', 'Birthrate', 'Deathrate', 'Agriculture', 'Industry', 'Service']

for _ in cols:
  df[_] = df[_].str.replace(',', '.').astype(float)
  
df.fillna(df.mean(), inplace=True)

#Reg = df['Region'].value_counts()
plt.figure(figsize=(14, 18))
for _, z in zip(cols, range(14)):
    plt.subplot(7, 2, z+1)
    plt.plot(df[_], color='r', linewidth=1)
    plt.xlabel('Country Index')
    plt.ylabel(_)

pop_factors = ['Net migration', 'Infant mortality', 'Birthrate', 'Deathrate']
plt.figure(figsize=(18, 18))
for _, z in zip(pop_factors, range(4)):
    plt.subplot(2, 2, z+1)
    plt.scatter(df['Population'], df[_], color='r', linewidth=1)
    plt.xlabel('Population')
    plt.ylabel(_)
    plt.xlim(0, 150000000)
    
plt.figure(figsize=(18, 18))    
sns.heatmap(df.corr(), annot=True)    

plt.figure(figsize=(9, 9))
plt.xlim(0, 2000000)
plt.ylim(0, 150)
sns.scatterplot(x='Area', y='Pop Density', data=df)
plt.show()

sectors = ['Agriculture', 'Phones', 'Service', 'Industry', 'Literacy', 'Infant mortality']
plt.figure(figsize=(18, 18))
for _, z in zip(sectors, range(6)):
    plt.subplot(3, 2, z+1)
    plt.scatter(df['GDP'], df[_], color='r', linewidth=1)
    plt.xlabel('GDP')
    plt.ylabel(_)
    plt.xlim(0, 40000)
   
plt.figure(figsize=(9, 9))    
plt.scatter(df['Literacy'], df['Infant mortality'],color='b') 
plt.xlabel('Literacy Level')
plt.ylabel('Infant Mortality Rate')

plt.figure(figsize=(9, 9))    
plt.scatter(df['Literacy'], df['Phones'],color='b') 
plt.xlabel('Literacy Level')
plt.ylabel('Phones per 1000')
