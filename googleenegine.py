# -*- coding: utf-8 -*-
"""
Created on Thu Sep 13 16:18:22 2018

@author: Sahil Nathani
"""

import quandl
import math
import datetime
import numpy as np
from sklearn import preprocessing, cross_validation, svm
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from matplotlib import style

style.use('fivethirtyeight')

quandl.ApiConfig.api_key = 'FhPJZ2kuyzc5hbsz6HdY'

data = quandl.get("CHRIS/MGEX_IH1", authtoken="FhPJZ2kuyzc5hbsz6HdY")

#previously we take all the coloumns irrespective of relations
data = data[['Open', 'High', 'Low', 'Last', 'Volume']]
#Difference b/w High and Low tells us of the volatility of the stock
#here we create coloumns which are meaningful to us
data['HL_PCT'] = ((data['High'] - data['Low'])/data['Low'])*100#Volatility
data['PCT_change'] = ((data['Last'] - data['Open'])/data['Open'])*100

data['forecast_col'] = data['Last']
#this will replace nan data with -99999. Treated as outlier.
data.fillna(-99999, inplace=True)

#this rounds the data to the nearest upper range integer
forecast_out = int(math.ceil(0.01*len(data)))#Total length of the data is 3467
print(forecast_out)#=34

data['label'] = data['forecast_col'].shift(-forecast_out)
#this will ignore all the nan values

#x is a 2-D array that is without label while y is 2-D array that has only labels
x = np.array(data.drop(['label'], 1))#1 corresponds to the axis

#the line below has been commented while doing prediction and is needed while making accuracy calc.
#y = np.array(data['label'])

x = preprocessing.scale(x)
#this is the label which actually predicts the data(stock price)
x_lately = x[-forecast_out:]

x = x[:-forecast_out]

data.dropna(inplace=True)

y = np.array(data['label'])

#test_size parameter: 0.2 means 20% data will be used for testing while 80% is taken for training.
#train_test_split returns 4 values which are as follows
x_train, x_test, y_train, y_test = cross_validation.train_test_split(x, y, test_size = 0.2)

#This calculates using Linear Regression
#clf is a classifier
clf = LinearRegression()#LinearRegression is a class whilefit and predict are the methods
clf.fit(x_train, y_train)#fit is synonymous with train
accuracy_reg = clf.score(x_test, y_test)#score is synonymous with testing

#This calculates using support vector machine
clf = svm.SVR(kernel='rbf')#by default kernel='linear'#accuracy = 70.89
clf.fit(x_train, y_train)#fit is synonymous with train
accuracy_svm = clf.score(x_test, y_test)#score is synonymous with testing

print("Accuracy using Regression:", accuracy_reg*100)
print("Accuracy using Support Vector Machine:", accuracy_svm*100)

forecast_set = clf.predict(x_lately)
print(forecast_set, accuracy_reg, forecast_out)

data['Forecast'] = np.nan

last_day = data.iloc[-1].name
last_unix = last_day.timestamp()
one_day = 86400 #no.of seconds in a day
next_unix = last_unix + one_day

for i in forecast_set:
    next_date = datetime.datetime.fromtimestamp(next_unix)
    next_unix += one_day
    data.loc[next_date] = [np.nan for _ in range(len(data.columns)-1)] + [i]#this sets the value of date to nan wherever the forecast is not present.
    
data['Last'].plot()
data['Forecast'].plot()
plt.legend(loc=0)
plt.xlabel('Date')
plt.ylabel('Price')
plt.show()    