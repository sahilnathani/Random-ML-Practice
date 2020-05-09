# -*- coding: utf-8 -*-
"""
Created on Mon Apr  1 19:57:11 2019

@author: Sahil Nathani
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import style
style.use('fivethirtyeight')

df = pd.read_csv("C:\\Users\\Sahil Nathani\\Desktop\\Python and ML Material\\Databases\\Youtube Channels\\data.csv")

df['Subscribers'] = df['Subscribers'].replace('-- ', '10000000')

df['Video Uploads'] = df['Video Uploads'].replace('--', '1000')

df['Subscribers'] = [int(each) for each in df['Subscribers']]
df['Video Uploads'] = [int(each) for each in df['Video Uploads']]

from sklearn.preprocessing import LabelEncoder

lb = LabelEncoder()
lb.fit(df['Grade'])
df['Grade'] = lb.fit_transform(df['Grade'])

for each in df['Video views']:
    each = int(each)
    
plt.figure(figsize=(9, 9))
plt.ylim(0, 10000000000)
sns.boxplot(x='Grade', y='Video views', data=df)

plt.figure(figsize=(9, 9))
plt.xlim(4000, 5000)
#plt.yticks(np.arange(min(df['Video Uploads']), max(df['Subscribers'])+1, 10000000))
plt.ylim(0, 10000000)
plt.ylabel('Subscribers')
plt.xlabel('Rank')
plt.scatter(df['Rank'], df['Subscribers'], color='b')

z=1
plt.figure(figsize=(10, 39))
for each in ['Video Uploads', 'Subscribers', 'Video views']:
    plt.subplot(3, 1, z)
    plt.xlabel('Grade')
    plt.ylabel(each)
    plt.ylim(0, 2000000)
    sns.violinplot(x='Grade', y=each, data=df)
    z+=1
    