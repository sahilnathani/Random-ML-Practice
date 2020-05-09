# -*- coding: utf-8 -*-
"""
Created on Wed Oct 17 10:44:46 2018

@author: Sahil Nathani
"""

import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import style
import numpy as np
style.use('fivethirtyeight')
from sklearn.cluster import KMeans
from sklearn import preprocessing, cross_validation

data = pd.read_excel('titanic.xls') 
data.drop(['body', 'name', 'ticket', 'home.dest'], 1, inplace=True)
data.convert_objects(convert_numeric=True)
data.fillna(0, inplace=True)

def non_numeric_data_handling(data):
    columns = data.columns.values
    
    for column in columns:
        text_digit_vals = {}
        def convert_to_int(val):
            return text_digit_vals[val]
        
        if data[column].dtype != np.int64 and data[column].dtype != np.float64:
            column_contents = data[column].values.tolist()
            unique_elements = set(column_contents)
            x=0
            for unique in unique_elements:
                if unique not in text_digit_vals:
                    text_digit_vals[unique] = x
                    x+=1
                    
            data[column] = list(map(convert_to_int, data[column]))
    return data

data = non_numeric_data_handling(data)

X = np.array(data.drop(['survived'], 1)).astype(float)
X = preprocessing.scale(X)
Y = np.array(data['survived'])

clf = KMeans(n_clusters=2)
clf.fit(X)

correct=0
for i in range(len(X)):
    predict_me=np.array(X[i].astype(float))
    predict_me=predict_me.reshape(-1, len(predict_me))
    prediction = clf.predict(predict_me)
    if prediction[0] == Y[i]:
        correct+=1
        
print(correct/len(X))        