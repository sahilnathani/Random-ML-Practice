# -*- coding: utf-8 -*-
"""
Created on Sat Dec 22 10:46:55 2018

@author: Sahil Nathani
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy.stats import norm
from sklearn.preprocessing import StandardScaler
from sklearn import svm, cross_validation
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

df = pd.read_csv('AmesHousePricePredTrain.csv')

df = df[['OverallQual', 'OverallCond', 'YearBuilt', 'YearRemodAdd', 'TotalBsmtSF', 'CentralAir', '1stFlrSF', '2ndFlrSF', 'GrLivArea', 'BedroomAbvGr', 'TotRmsAbvGrd', 'GarageCars', 'GarageArea', 'SaleType', 'SalePrice']]
     
m1 = {'Y' : 1, 'N' : 0}
df['CentralAir'] = df['CentralAir'].str[0].map(m1)

m2 = {'New' : 1, 'WD' : 0, 'COD' : 0, 'ConLD' : 0, 'CWD' : 0, 'ConLw' : 0, 'Con' : 0, 'Oth' : 0}
df['SaleType'] = df['SaleType'].map(m2)
df.dropna(inplace=True)   
'''
total = df.isnull().sum().sort_values(ascending=False)
percent = (df.isnull().sum()/df.isnull().count()).sort_values(ascending=False)
missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])
missing_data.head(20)

df = df.drop((missing_data[missing_data['Total'] > 1]).index,1)
df.isnull().sum().max()
'''
#heatmap
'''
corrmat = df.corr()
f, ax = plt.subplots(figsize=(12, 9))
sns.heatmap(corrmat, vmax=.8, square=True);

k = 10 #number of variables for heatmap
cols = corrmat.nlargest(k, 'SalePrice')['SalePrice'].index
cm = np.corrcoef(df[cols].values.T)
sns.set(font_scale=1.25)
hm = sns.heatmap(cm, cbar=True, annot=True, square=True, fmt='.2f', annot_kws={'size': 10}, yticklabels=cols.values, xticklabels=cols.values)
plt.show()
'''

#pairplot relations
'''
sns.set()
cols = ['SalePrice', 'OverallQual', 'GrLivArea', 'GarageCars', 'TotalBsmtSF', 'FullBath', 'YearBuilt']
sns.pairplot(df[cols], size = 2.5)
plt.show()
'''

saleprice_scaled = StandardScaler().fit_transform(df['SalePrice'][:,np.newaxis])
low_range = saleprice_scaled[saleprice_scaled[:,0].argsort()][:10]
high_range= saleprice_scaled[saleprice_scaled[:,0].argsort()][-10:]
print('outer range (low) of the distribution:')
print(low_range)
print('\nouter range (high) of the distribution:')
print(high_range)

#Other scatter relations
'''
sns.distplot(df['SalePrice'], fit=norm);
fig = plt.figure()
res = stats.probplot(df['SalePrice'], plot=plt)

df['SalePrice'] = np.log(df['SalePrice'])

sns.distplot(df['SalePrice'], fit=norm);
fig = plt.figure()
res = stats.probplot(df['SalePrice'], plot=plt)

sns.distplot(df['GrLivArea'], fit=norm);
fig = plt.figure()
res = stats.probplot(df['GrLivArea'], plot=plt)

df['GrLivArea'] = np.log(df['GrLivArea'])

sns.distplot(df['GrLivArea'], fit=norm);
fig = plt.figure()
res = stats.probplot(df['GrLivArea'], plot=plt)

sns.distplot(df['TotalBsmtSF'], fit=norm);
fig = plt.figure()
res = stats.probplot(df['TotalBsmtSF'], plot=plt)

df['HasBsmt'] = pd.Series(len(df['TotalBsmtSF']), index=df.index)
df['HasBsmt'] = 0 
df.loc[df['TotalBsmtSF']>0,'HasBsmt'] = 1

df.loc[df['HasBsmt']==1,'TotalBsmtSF'] = np.log(df['TotalBsmtSF'])

sns.distplot(df[df['TotalBsmtSF']>0]['TotalBsmtSF'], fit=norm);
fig = plt.figure()
res = stats.probplot(df[df['TotalBsmtSF']>0]['TotalBsmtSF'], plot=plt)

plt.scatter(df['GrLivArea'], df['SalePrice'])

plt.scatter(df[df['TotalBsmtSF']>0]['TotalBsmtSF'], df[df['TotalBsmtSF']>0]['SalePrice'])
'''
df = pd.get_dummies(df)

#print(df.columns)
#df.columns = []

x = np.array(df.drop(['SalePrice'], 1))
y = np.array(df['SalePrice'])

x_train, x_test, y_train, y_test = cross_validation.train_test_split(x, y, test_size=0.2)

clf = svm.SVR(kernel='linear', C=40, gamma=0.007)
clf.fit(x_train, y_train)

df_in = pd.read_csv('AmesHousePricePredTest.csv')
df_in = df_in[['OverallQual', 'OverallCond', 'YearBuilt', 'YearRemodAdd', 'TotalBsmtSF', 'CentralAir', '1stFlrSF', '2ndFlrSF', 'GrLivArea', 'BedroomAbvGr', 'TotRmsAbvGrd', 'GarageCars', 'GarageArea', 'SaleType']]

m1 = {'Y' : 1, 'N' : 0}
df_in['CentralAir'] = df_in['CentralAir'].str[0].map(m1)

m2 = {'New' : 1, 'WD' : 0, 'COD' : 0, 'ConLD' : 0, 'CWD' : 0, 'ConLw' : 0, 'Con' : 0, 'Oth' : 0}
df_in['SaleType'] = df_in['SaleType'].map(m2)

df_in.fillna(df_in.mean(), inplace=True)

total = df_in.isnull().sum().sort_values(ascending=False)
percent = (df_in.isnull().sum()/df_in.isnull().count()).sort_values(ascending=False)
missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])
missing_data.head(20)

df_in = df_in.drop((missing_data[missing_data['Total'] > 1]).index,1)
df_in.isnull().sum().max()

x_in = np.array(df_in)
result = clf.predict(x_in)
result = [int(_) for _ in result]
print(result)

out_data = pd.DataFrame(result)
out_data.index.name = 'Id'
out_data.index+=1
out_data.columns=['SalePrice']
out_data.to_csv('AmesHousePredPrices.csv', header=True)
print(clf.score(x_test, y_test)*100)