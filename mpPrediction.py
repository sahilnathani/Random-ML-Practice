# -*- coding: utf-8 -*-
"""
Created on Sun Sep 30 18:25:49 2018

@author: Sahil Nathani
"""

from statistics import mean
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import style

style.use('fivethirtyeight')

data = pd.read_csv("MPDistrictData.csv", nrows=8, header=0)

data["Year"] = [1999, 2000, 2001, 2002, 2003, 2004, 2005, 2006]    
xs = np.array(data["Year"], dtype= np.float64)
ys = np.array(data["Seoni"], dtype = np.float64)

def get_slope(xs, ys):
     m = ((mean(xs)*mean(ys)-mean(xs*ys)) / (mean(xs)**2-mean(xs**2)))
     b = (mean(ys) - m*mean(xs))
     return m,b

m,b = get_slope(xs, ys) 

regression_line_vals = [(m*x)+b for x in xs]

def predict_gdp(year, m, b):
    return m*year + b

year = float(input("Enter the year of GDP"))

gdp = predict_gdp(year, m, b)

print("The predicted value of GDP in year ", year, "is", gdp)

plt.scatter(xs, ys)
plt.plot(xs, regression_line_vals, color = 'b')
plt.xlabel("Year")
plt.ylabel("Revenue of Bhopal")
plt.show()