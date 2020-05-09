# -*- coding: utf-8 -*-
"""
Created on Wed Feb 27 16:47:31 2019

@author: Sahil Nathani
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import style
import seaborn as sns

style.use('ggplot')

df = pd.read_csv("C:\\Users\\Sahil Nathani\\Desktop\\Python and ML Material\\Databases\\SuicideRates.csv")
#plt.figure(figsize=(10, 10))
#df['country'].value_counts().hist()

#suicide rates in countries
plt.figure(figsize=(35, 10))
suicide_num = (df.groupby(['country'])['suicides_no']).agg(['size'])
plt.xticks(rotation='vertical')
plt.grid(True)
plt.scatter(df['country'].unique(), suicide_num)

sui = []

#suicide rates vs gdp_capita
gdp_capita = (df.groupby(['country'])['gdp_per_capita ($)'].mean())
plt.figure(figsize=(10, 10))
plt.xlabel('No of Suicides')
plt.ylabel('GDP')
plt.title('Suicides vs GDP_per_Capita')
plt.scatter(suicide_num, gdp_capita, color='blue')

#for each in df[' gdp_for_year ($) ']:
#    each = each.replace(",", " ")
#    
#print(df[' gdp_for_year ($) '])
#print(type(df[' gdp_for_year ($) '][0]))
#
##suicide rates vs gdp_year
#gdp_year = (df.groupby(['country'])[' gdp_for_year ($) '].mean())
#plt.figure(figsize=(10, 10))
#plt.xlabel('No of Suicides')
#plt.ylabel('GDP')
#plt.title('Suicides vs GDP_Year')
#plt.scatter(suicide_num, gdp_year, color='blue')

#suicide rates vs population
gdp_year = (df.groupby(['country'])['population'].mean())
plt.figure(figsize=(10, 10))
plt.xlabel('No of Suicides')
plt.ylabel('Population')
plt.title('Population vs Suicides')
#plt.ylim(0, 5000000)
plt.scatter(suicide_num, gdp_year, color='blue')

gender = (df.groupby(['country'])['sex']).agg(['size'])
print(gender)

#_Suicidial Rates
plt.figure(figsize=(35, 10))
suicide_rate = ((df.groupby(['country'])['suicides/100k pop']).mean())
plt.xticks(rotation='vertical')
plt.grid(True)
plt.ylim(20, 40)
plt.scatter(df['country'].unique(), suicide_rate)

age_df = ((df.groupby(['country'])['generation']).agg(['size']))
print(age_df)
plt.figure(figsize=(20, 10))
sns.countplot(x='generation', data=df)