# -*- coding: utf-8 -*-
"""
Created on Tue Nov  6 12:51:05 2018

@author: Sahil Nathani
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import style

data = pd.read_csv('courseraLogistic2.csv')
style.use('fivethirtyeight')

def sigmoid(z):
    return 1/(1 + np.exp(-z))

def poly_Logistic_Regression(x1, x2, y, m1_ini, m2_ini, m3_ini, m4_ini, m5_ini, b_ini, iters=400, learn_rate=0.001, lam=1):
   n = float(len(y))
   for i in range(iters):
       z = (b_ini+m1_ini*x1*x1+m2_ini*x2*x2+m3_ini*x1+m4_ini*x2+m5_ini*x1*x2)#tis is a list
       y_ini = sigmoid(z)#this is also a list
       '''for each_val in y_ini:
           if each_val>=0.5:
               each_val=1
           else:
               each_val=0
       '''        
       cost = -sum([(i*np.log(j) + (1-i)*(np.log(1-j))) for i, j in zip(y, y_ini)])/n
       b_grad = -sum((-1*y*np.exp(-z)/(1+np.exp(-z)))+((1-y)/(1+np.exp(-z))))/n 
       m1_grad = -sum(((x1*x1*y*np.exp(-z))/(1+np.exp(-z)))-((x1*x1*(1-y))/(1+np.exp(-z))))/n + (lam/n)*m1_ini
       m2_grad = -sum(((x2*x2*y*np.exp(-z))/(1+np.exp(-z)))-((x2*x2*(1-y))/(1+np.exp(-z))))/n + (lam/n)*m2_ini
       m3_grad = -sum(((x1*y*np.exp(-z))/(1+np.exp(-z)))-((x1*(1-y))/(1+np.exp(-z))))/n + (lam/n)*m3_ini
       m4_grad = -sum(((x2*y*np.exp(-z))/(1+np.exp(-z)))-((x2*(1-y))/(1+np.exp(-z))))/n + (lam/n)*m4_ini
       m5_grad = -sum(((x2*x1*y*np.exp(-z))/(1+np.exp(-z)))-((x1*x2*(1-y))/(1+np.exp(-z))))/n + (lam/n)*m5_ini
       b_ini-=(learn_rate*b_grad)
       m1_ini-=(learn_rate*m1_grad)
       m2_ini-=(learn_rate*m2_grad)
       m3_ini-=(learn_rate*m3_grad)
       m4_ini-=(learn_rate*m4_grad)
       m5_ini-=(learn_rate*m5_grad)
   return m1_ini, m2_ini, m3_ini, m4_ini, m5_ini, b_ini, cost  

for i, j, k in zip(data['Test1'], data['Test2'], data['Decision']):
    if k==1:
        plt.scatter(i, j, marker='o', s=50, color='r')
    else:
        plt.scatter(i, j, marker='x', s=50, color='g')
        
plt.xlabel('Test1 Scores')        
plt.ylabel('Test2 Scores')
        
m1_ini, m2_ini, m3_ini, m4_ini, m5_ini, b_ini, cost = poly_Logistic_Regression(data['Test1'], data['Test2'], data['Decision'], 0, 0, 0, 0, 0, 0)        
        
print(m1_ini, m2_ini, m3_ini, m4_ini, m5_ini, b_ini, cost)

t1 = float(input('Performance in First Test'))
t2 = float(input('Performance in Second Test'))
if sigmoid((b_ini+m1_ini*t1*t1+m2_ini*t2*t2+m3_ini*t1+m4_ini*t2+m5_ini*t1*t2))>=0.5:
    print('1: Accepted', sigmoid((b_ini+m1_ini*t1*t1+m2_ini*t2*t2+m3_ini*t1+m4_ini*t2+m5_ini*t1*t2))*100)
else:
    print('0: Rejected', sigmoid((b_ini+m1_ini*t1*t1+m2_ini*t2*t2+m3_ini*t1+m4_ini*t2+m5_ini*t1*t2))*100)         