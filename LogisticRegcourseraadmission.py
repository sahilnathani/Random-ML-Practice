# -*- coding: utf-8 -*-
"""
Created on Mon Nov  5 21:54:29 2018

@author: Sahil Nathani
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import style

data = pd.read_csv('courseraLogistic1.csv')
style.use('fivethirtyeight')

def sigmoid(z):
    return 1/(1 + np.exp(-z))

def Logistic_Regression(x1, x2, y, m1_ini, m2_ini, b_ini, iters=400, learn_rate=0.001):
   n = float(len(y))
   for i in range(iters):
       z = (b_ini+m1_ini*x1+m2_ini*x2)#tis is a list
       y_ini = sigmoid(z)#this is also a list
       '''for each_val in y_ini:
           if each_val>=0.5:
               each_val=1
           else:
               each_val=0
       '''        
       cost = -sum([(i*np.log(j) + (1-i)*(np.log(1-j))) for i, j in zip(y, y_ini)])/n
       b_grad = -sum((-1*y*np.exp(-z)/(1+np.exp(-z)))+((1-y)/(1+np.exp(-z))))/n
       m1_grad = -sum(((x1*y*np.exp(-z))/(1+np.exp(-z)))-((x1*(1-y))/(1+np.exp(-z))))/n
       m2_grad = -sum(((x2*y*np.exp(-z))/(1+np.exp(-z)))-((x2*(1-y))/(1+np.exp(-z))))/n
       b_ini-=(learn_rate*b_grad)
       m1_ini-=(learn_rate*m1_grad)
       m2_ini-=(learn_rate*m2_grad)
   return m1_ini, m2_ini, b_ini, cost

m1, m2, b, cost = Logistic_Regression(data['Exam1'], data['Exam2'], data['Decision'], 0, 0, 0) 
print(m1, m2, b, cost)  

for i, j, k in zip(data['Exam1'], data['Exam2'], data['Decision']):
    if k==1:
        plt.scatter(i, j, marker='+', s=50, color='g')
    else:
        plt.scatter(i, j, marker='o', s=50, color='r')
       
marks1 = int(input('Enter your marks in First Exam'))
marks2 = int(input('Enter your marks in Second Exam'))
if sigmoid((marks1*m1+marks2*m2+b))>=0.5:
    print('1: Through', sigmoid((marks1*m1+marks2*m2+b))*100)
else:
    print('0: Not Through', sigmoid((marks1*m1+marks2*m2+b))*100) 
    
      