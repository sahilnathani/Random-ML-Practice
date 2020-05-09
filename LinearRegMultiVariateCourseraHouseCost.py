# -*- coding: utf-8 -*-
"""
Created on Mon Nov  5 10:37:56 2018

@author: Sahil Nathani
"""

import pandas as pd
#import matplotlib.pyplot as plt
#from mpl_toolkits.mplot3d import Axes3D
#fig=plt.figure()
#ax = fig.add_subplot(111, projection='3d')
from matplotlib import style
import numpy as np

style.use('fivethirtyeight')
data = pd.read_csv("C:\\Users\\Sahil Nathani\\Desktop\\Python and ML Material\\Databases\\Coursera\\ex1data2.csv")

n = (len(data['Size']))

X = []
for i in range(47):
    X.append([])
   
for i, j, k in zip(range(47), data['Size'], data['Room']):
    X[i].append(1)
    X[i].append(j)
    X[i].append(k)
    
X = np.array(X)    
Y = np.array(data['Cost'])
Y = Y.reshape((47, 1))
parameters = []

parameters = np.dot(np.dot(np.linalg.inv(np.dot(X.transpose(), X)), X.transpose()), Y)

print(parameters)