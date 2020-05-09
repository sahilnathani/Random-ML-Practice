# -*- coding: utf-8 -*-
"""
Created on Wed Oct  3 13:18:15 2018

@author: Sahil Nathani
"""

import pandas as pd
import numpy as np
from sklearn import cross_validation, neighbors

data = pd.read_csv('BreastCancer (2).txt')
data.replace('?', -9999999, inplace=True)#we use a bizarre value such that it is treated as outlier
data.drop(['id'], 1, inplace=True)#id has no impact on tumor being benign or malignant, hence dropped

X = np.array(data.drop(['class'], 1))
Y = np.array(data['class'])

x_train, x_test, y_train, y_test = cross_validation.train_test_split(X, Y, test_size=0.2)

clf = neighbors.KNeighborsClassifier()
clf.fit(x_train, y_train)
accuracy = clf.score(x_test, y_test)

print(accuracy*100)

example_measure = np.array([6, 2, 8, 1, 1, 2, 3, 2, 1])
example_measure = example_measure.reshape(1, -1)
print(example_measure)

prediction = clf.predict(example_measure)
print(prediction)