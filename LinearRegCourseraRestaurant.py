# -*- coding: utf-8 -*-
"""
Created on Sun Nov  4 19:31:45 2018

@author: Sahil Nathani
"""

import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import style

style.use('fivethirtyeight')
data = pd.read_csv("C:\\Users\\Sahil Nathani\\Desktop\\Python and ML Material\\Databases\\Coursera\\ex1data1.csv")

def Linear_Regression(x, y, m_ini, b_ini, iters=1500, learn_rate=0.01):
    n = float(len(y))
    for i in range(iters):
        y_ini = (m_ini)*x + b_ini
        cost = sum([each_val**2 for each_val in (y-y_ini)])/n
        m_grad = -(2/n)*sum(x*(y-y_ini))
        b_grad = -(2/n)*sum((y-y_ini))
        m_ini-=(learn_rate*m_grad)
        b_ini-=(learn_rate*b_grad)
    return m_ini, b_ini, cost

m, c, cost = Linear_Regression(data['Population'], data['Profit'], 0, 0)

print(m, c, cost)
reg_vals = [(m*each_val+c) for each_val in data['Population']]

plt.scatter(data['Population'], data['Profit'], marker='x', s=20)
plt.plot(data['Population'], reg_vals, color='r')
plt.xlabel('Population(in 1000)')
plt.ylabel('Profit(in $1000)')
plt.show()
#cost is minimum for learn_rate=0.01