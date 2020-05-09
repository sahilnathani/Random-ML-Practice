# -*- coding: utf-8 -*-
"""
Created on Wed Jan  9 20:29:43 2019

@author: Sahil Nathani
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn import cross_validation, svm, preprocessing
from sklearn.cluster import KMeans
import seaborn as sns

plt.style.use('ggplot')

df = pd.read_csv('StudentsPerformance.csv')

df.fillna(-99999, inplace=True)

m1 = {'m' : 1, 'f' : 0}
df['gender'] = df['gender'].str[0].str.lower().map(m1)

m2 = {'A' : 0, 'B' : 1, 'C' : 2, 'D' : 3, 'E' : 4}
df['race/ethnicity'] = df['race/ethnicity'].str[6].map(m2)
 
m3 = {'n' : 0, 'c' : 1}
df['test preparation course'] = df['test preparation course'].str[0].map(m3)

m4 = {'f' : 0, 's' : 1} 
df['lunch'] = df['lunch'].str[0].map(m4)

m5 = {'b' : 0, 's' : 1, 'm' : 2, 'a' : 3, 'h' : 4}
df['parental level of education'] = df['parental level of education'].str[0].map(m5)

x = np.array(df)

df['average'] = (((df['math score'] + df['reading score'] + df['writing score'])/3)/50).astype('int64')

y = df['average']

sns.catplot(x='test preparation course', y='writing score', kind='violin', data=df)

x_train, x_test, y_train, y_test = cross_validation.train_test_split(x, y, test_size=0.2)

'''
b, s, m, a, h = 0, 0, 0, 0, 0
for _, z in zip(df['race/ethnicity'], df['average']):
    if z==1:
      if _ == 0:
         b+=1
      if _ == 1:
         s+=1 
      if _ == 2:
         m+=1 
      if _ == 3:
         a+=1 
      if _ == 4:
         h+=1    

print(b, s, m, a, h) 
'''       