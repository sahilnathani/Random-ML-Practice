# -*- coding: utf-8 -*-
"""
Created on Fri Dec 21 10:54:37 2018

@author: Sahil Nathani
"""

import pandas as pd
import matplotlib.pyplot as plt
from sklearn import cross_validation, svm

data = pd.read_csv('C:\\Users\\Sahil Nathani\\Desktop\\Python and ML Material\\Databases\\DigitRecognizer\\traindigits.csv')

x = data.iloc[0:, 1:]
y = data.iloc[0:, :1]

x_train, x_test, y_train, y_test = cross_validation.train_test_split(x, y, test_size=0.2)

i=786
img=x_train.iloc[i].as_matrix()
img=img.reshape((28,28))
plt.imshow(img,cmap='gray')
plt.title(x_train.iloc[i])

x_train[x_train>0]=1
x_test[x_test>0]=1

clf = svm.SVC(kernel='rbf', C=7, gamma=0.01)
clf.fit(x_train, y_train.values.ravel())
accuracy = clf.score(x_test, y_test)

print(accuracy*100)

data_for_test = pd.read_csv('testdigits.csv')
data_for_test[data_for_test>0]=1
result = clf.predict(data_for_test[0:])

print(result)

out_data = pd.DataFrame(result)
out_data.index.name = 'ImageID'
out_data.index+=1
out_data.columns=['Label']
out_data.to_csv('digitresult1.csv', header=True)