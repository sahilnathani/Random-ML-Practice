# -*- coding: utf-8 -*-
"""
Created on Mon May 13 10:41:39 2019

@author: Sahil Nathani
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv("C:\\Users\\Sahil Nathani\\Desktop\\Python and ML Material\\Databases\\Google Play Store\\googleplaystore_user_reviews.csv", encoding='ISO-8859-1')

print(df.info())
print(df.isnull().sum())

df = df.dropna()

plt.figure(figsize=(20, 20))
plt.xlabel('Sentiment_Polarity')
plt.xlabel('Sentiment_Subjectivity')
sns.scatterplot(df.Sentiment_Polarity, df.Sentiment_Subjectivity, color='red')