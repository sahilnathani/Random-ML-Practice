# -*- coding: utf-8 -*-
"""
Created on Thu Jan 17 16:30:28 2019

@author: Sahil Nathani
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn import cross_validation, svm

plt.style.use('ggplot')

df = pd.read_csv('Admission_graduate_school.csv')
#print(df.columns)
df.drop(['Serial No.'], 1, inplace=True)
'''
cols = ['GRE Score', 'TOEFL Score']

plt.figure(figsize=(8, 8))
for _, z in zip(cols, range(2)):
    plt.subplot(2, 1, z+1)
    plt.scatter(x=df['Chance of Admit '], y=df[_], color='r')
    plt.xlabel('Chance of Admit')
    plt.ylabel(_)  
    
plt.figure(figsize=(8, 8))
plt.subplot(2, 2, 1)    
sns.distplot(df['Chance of Admit '], kde=False, bins=5)
plt.subplot(2, 2, 2)    
sns.distplot(df['LOR '], kde=False, bins=15)
plt.subplot(2, 2, 3)    
sns.distplot(df['SOP'], kde=False, bins=10)
plt.subplot(2, 2, 4)    
sns.distplot(df['University Rating'], kde=False, bins=10) 

plt.figure(figsize=(4, 4))
sns.distplot(df['Research'],kde=False, bins=2) 
plt.figure(figsize=(4, 4))
plt.scatter(x=df['Chance of Admit '], y=df['Research'], color='b')
plt.xlim(0.8, 1.0)

plt.figure(figsize=(10, 10))
sns.heatmap(df.corr(), annot=True)
'''
x = np.array(df.drop(['Chance of Admit '], 1))
y = np.array(df['Chance of Admit '])

x_train, x_test, y_train, y_test = cross_validation.train_test_split(x, y, test_size=0.2)

clf = svm.SVR(kernel='linear', gamma=0.02)
clf.fit(x_train, y_train)
accuracy = clf.score(x_test, y_test)

print(accuracy*100) 
z = [315, 105, 4, 4, 4, 9.5, 1]
z = np.array(z).reshape(1, -1)
print(clf.predict(z)*100)
