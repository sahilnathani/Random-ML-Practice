# -*- coding: utf-8 -*-
"""
Created on Wed Oct  3 15:05:53 2018

@author: Sahil Nathani
"""

#Using euclidean distance is not effiient hence we use circles of given radii

import numpy as np
import pandas as pd
import warnings
import random
from collections import Counter

data = pd.read_csv('BreastCancer (2).txt')
data.replace('?', -9999999, inplace=True)#we use a bizarre value such that it is treated as outlier
data.drop(['id'], 1, inplace=True)#id has no impact on tumor being benign or malignant, hence dropped
full_data = data.astype(float).values.tolist()#converts every value to float for reusability

full_data = random.shuffle(full_data)

def k_nearest_neighbor(data, predict, k=3):
    if len(data)>=k:
        warnings.warn('K is set less than total voting groups!')
    distances = []
    for group in data:#this is the set of each value for single patient
        for features in data[group]:#this considers each value of the parameter
            euclidean_distances = np.linalg.norm(np.array(features)-np.array(predict))
            distances.append([euclidean_distances, group])
            
    votes = [i[1] for i in sorted(distances)[:k]]
    print(Counter(votes).most_common(1))
    vote_result = Counter(votes).most_common(1)[0][0]        
    return vote_result  

test_size = 0.2
train_set = {2:[], 4:[]}
test_set = {2:[], 4:[]}
train_data = data[:int(test_size*len(data))]
test_data = data[int(test_size*len(data)):]

for i in train_data:
    train_set[i[-1]].append(i[:-1])##this add all the data in the corresponding to respective class

for i in test_data:
    test_set[i[-1]].append(i[:-1])
    
correct = 0
total = 0

for group in test_set:#for data of each patient do:
    for data in test_set[group]:#for each instance of paramter of one patient do: 
        vote = k_nearest_neighbor(train_set, data, k=3)
        if group==vote:
            correct+=1
        total+=1
        
print('Accuracy:', correct/total)          