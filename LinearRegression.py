# -*- coding: utf-8 -*-
"""
Created on Thu Sep 13 16:18:22 2018

@author: Sahil Nathani
"""

from statistics import mean
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import style
import random

style.use('fivethirtyeight')

#xs = np.array([1, 2, 3, 4, 5, 6], dtype = np.float64)
#ys = np.array([5, 4, 6, 5, 6, 7], dtype = np.float64)

def best_fit_slope(xs, ys):
    m = ((mean(xs)*mean(ys)-mean(xs*ys)) / (mean(xs)**2-mean(xs**2)))
    return m

m = best_fit_slope(xs, ys)

def intercept_line(xs, ys):
    c = mean(ys) - m*mean(xs)
    return c

def create_dataset(hm, variance, step=2, correlation=False):
    val = 1
    ys = []
    for i in range(hm):
        y = val + random.randrange(-variance, variance)
        ys.append(y)
        if correlation and correlation == 'pos':
            val+=step
        if correlation and correlation == 'neg':
            val-=step
    xs = [i for i in range(len(ys))]
        
    return np.array(xs, dtype=np.float64), np.array(ys, dtype=np.float64)

xs, ys = create_dataset(100, 1000, 2, correlation='pos')

c = intercept_line(xs, ys)

regression_line = [(m*x)+c for x in xs]
#for x in xs:
#   regression_line.append((m*x)+c)

plt.scatter(xs, ys)
plt.plot(xs, regression_line)
plt.show()