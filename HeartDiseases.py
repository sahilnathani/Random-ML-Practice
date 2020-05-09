# -*- coding: utf-8 -*-
"""
Created on Tue Feb 26 17:55:37 2019

@author: Sahil Nathani
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import style
import seaborn as sns
from sklearn import cross_validation
from sklearn.neighbors import KNeighborsClassifier
from sklearn import svm

style.use('ggplot')

df = pd.read_csv("C:\\Users\\Sahil Nathani\\Desktop\\Python and ML Material\\Databases\\heart.csv")

plt.title('No of People in bothe the Categories')
sns.countplot(x='target', data=df)
'''the people are equally distributed among the two classes.'''

'''distribution of sex vs target'''
plt.title('Distribution of Gender vs Target. Although Male are twice the female')
gender = {0 : 'female', 1 : 'male'}
df['sex'] = df['sex'].map(gender)

z=1
plt.figure(figsize=(20, 20))
for x in df.columns:
    plt.subplot(6, 2, z)
    sns.countplot(x=x, data=df, hue='target')
    z+=1

for x in df['age']:
    if x>=50:
        x=1
    else:
        x=0
    
for x in df['chol']:
    if x>=225:
        x=1
    else:
        x=0    
    
X = np.array(df.drop(['target'], 1))
Y = np.array(df['target'])  

X_train, X_test, Y_train, Y_test = cross_validation.train_test_split(X, Y, test_size=0.2)

knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train, Y_train)
accuracy = 100*knn.score(X_test, Y_test)


clf = svm.SVC(kernel='linear')
clf.fit(X_train, Y_train)
accuracy = clf.score(X_test, Y_test)
print(accuracy*100)  