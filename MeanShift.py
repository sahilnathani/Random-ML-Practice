# -*- coding: utf-8 -*-
"""
Created on Sat Oct 20 14:37:16 2018

@author: Sahil Nathani
"""

import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import style
import numpy as np
style.use('fivethirtyeight')
from sklearn.cluster import MeanShift
from sklearn import preprocessing, cross_validation

data = pd.read_excel('titanic.xls') 
orig_data = pd.DataFrame.copy(data)
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

clf = MeanShift()
clf.fit(X)

labels = clf.labels_
clusters_centers = clf.cluster_centers_

orig_data['cluster_group'] = np.nan

for i in range(len(X)):
    orig_data['cluster_group'].iloc[i] = labels[i]#iloc references to index
    
n_clusters_ = len(np.unique(labels))
    
survival_rates = {}
for i in range(n_clusters_):
    temp_data = orig_data[(orig_data['cluster_group']==float(i))]
    survival_cluster = temp_data[(temp_data['survived']==1)]
    survival_rate = len(survival_cluster)/len(temp_data)
    survival_rates[i] = survival_rate
    
print(survival_rates)  
print(orig_data[(orig_data['cluster_group']==4)].describe())

    
    