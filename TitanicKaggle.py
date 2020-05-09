# -*- coding: utf-8 -*-
"""
Created on Mon Dec 24 12:23:53 2018

@author: Sahil Nathani
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import style
from sklearn import cross_validation, preprocessing, svm
from sklearn.ensemble import RandomForestClassifier
style.use('ggplot')

data = pd.read_csv('titanictrain.csv')

data.fillna(data.mean(), inplace=True)
data = data.drop(['PassengerId', 'Name', 'Ticket', 'Cabin', 'Embarked', 'Fare'], axis=1)

data['total'] = (data['SibSp']+data['Parch']+1)

data.Sex[data.Sex=='male']=1
data.Sex[data.Sex=='female']=0
'''
data.Age[data.Age<=35]=0
data.Age[data.Age>35]=1
'''
'''
data.SibSp[data.SibSp<=2]=0
data.SibSp[data.SibSp>2]=1
'''
data.Parch[data.Parch<=2]=0
data.Parch[data.Parch>2]=1

'''
plt.xlabel('Survival')
plt.ylabel('Age')
plt.scatter(data['Survived'], data['Age'])
'''

x = np.array(data.drop(['Survived'], 1))
y = np.array(data['Survived'])

x_train, x_test, y_train, y_test = cross_validation.train_test_split(x, y, test_size=0.2)

clf = svm.SVC(kernel = 'rbf', C=1100, gamma=0.002)
clf.fit(x_train, y_train)
accuracy = clf.score(x_test, y_test)

print(accuracy*100)

df = pd.read_csv('titanictest.csv')
df.fillna(-99999, inplace=True)
df.Sex[df.Sex=='male']=1
df.Sex[df.Sex=='female']=0
'''
df.Age[df.Age<=29]=0
df.Age[df.Age>29]=1
'''
'''
df.SibSp[df.SibSp<=2]=0
df.SibSp[df.SibSp>2]=1
'''
df.Parch[df.Parch<=2]=0
df.Parch[df.Parch>2]=1

df = df.drop(['PassengerId', 'Name', 'Ticket', 'Cabin', 'Embarked', 'Fare'], axis=1)
df['total'] = (df['SibSp']+df['Parch']+1)
result = clf.predict(df[0:])

print(result)

out_data = pd.DataFrame(result)
out_data.index.name = 'PassengerID'
out_data.index+=1
out_data.columns=['Survived']
out_data.to_csv('TitanicSurvivalKaggle.csv', header=True)