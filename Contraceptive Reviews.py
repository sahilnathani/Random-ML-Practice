# -*- coding: utf-8 -*-
"""
Created on Mon May 13 07:05:51 2019

@author: Sahil Nathani
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv("C:\\Users\\Sahil Nathani\\Desktop\\Python and ML Material\\Databases\\Contraceptive Reviews.csv", encoding='ISO-8859-1')

print(df.info())
print(df.isnull().sum())

df = df.dropna()

ratings = df['ratings'].value_counts().to_dict()

plt.figure(figsize=(10, 9))
plt.title('Count of Reviews')
plt.xlabel('Number of Reviews')
plt.ylabel('Rating')
plt.barh(range(len(ratings)), list(ratings.values()), align='center')
plt.yticks(range(len(ratings)), list(ratings.keys()))

df_rate = df.groupby('ratings')
length = df_rate['length'].mean().to_dict()

plt.figure(figsize=(10, 9))
plt.title('Average Length of Review against Rating')
plt.xlabel('Average Length')
plt.ylabel('Rating')
plt.barh(range(len(length)), list(length.values()), align='center', color='green')
plt.yticks(range(len(length)), list(length.keys()))

import nltk as nlp
from sklearn import cross_validation
import re444

text_list=[]
for i in df['reviews']:
    text = re.sub("[^a-zA-Z]" ," ", i)
    text = text.lower()
    text = nlp.word_tokenize(text)
    lemma = nlp.WordNetLemmatizer()
    text = [lemma.lemmatize(word) for word in text]
    text = " ".join(text)
    text_list.append(text)

from sklearn.feature_extraction.text import CountVectorizer
max_features=200000
count_vectorizer= CountVectorizer(max_features=max_features, stop_words="english")
sparce_matrix = count_vectorizer.fit_transform(text_list).toarray()
all_words=count_vectorizer.get_feature_names()
print("Most used words: ", all_words[0:50])

from wordcloud import WordCloud
plt.subplots(figsize=(12,12))
wordcloud=WordCloud(background_color="white",width=1024,height=768).generate(" ".join(all_words[100:]))
plt.imshow(wordcloud)
plt.axis("off")
plt.show()

x = sparce_matrix
y = df['ratings']
x_train, x_test, y_train, y_test = cross_validation.train_test_split(x, y, test_size=0.33, random_state=42)

#counter = CountVectorizer(stop_words=stop_words)
#count_train = counter.fit_transform(x_train)
#y_train = np.asarray(y_train.values)
#
#from sklearn.feature_selection  import SelectKBest
#from sklearn.feature_selection import chi2
#                    
#chi = SelectKBest(chi2, k=200)
#x_new = chi.fit_transform(count_train, y_train)
#count_test = counter.transform(x_test) 
#x_test_new = chi.transform(X=count_test)
#
#from sklearn.feature_extraction.text import TfidfVectorizer
#
#tfidf = TfidfVectorizer(stop_words='english')
#tfidf_train = tfidf.fit_transform(x_train)
#tfidf_test = tfidf.transform(x_test)

from sklearn.naive_bayes import MultinomialNB
from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier

mnb = MultinomialNB()
mnb.fit(x_train, y_train)
predm = mnb.predict(x_test)
scorem = metrics.accuracy_score(y_test, predm)

rfc = RandomForestClassifier(n_estimators=50)
rfc.fit(x_train, y_train)
predr = rfc.predict(x_test)
scorer = metrics.accuracy_score(y_test, predr)

svc = SVC()
svc.fit(x_train, y_train)
preds = svc.predict(x_test)
scores = metrics.accuracy_score(y_test, preds)

knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(x_train, y_train)
predk = knn.predict(x_test)
scorek = metrics.accuracy_score(y_test, predk)

print("Score of MultinomialNB", scorem*100)
print("Score of RandomForestClassifier", scorer*100)
print("Score of SVC", scores*100)
print("Score of KNN", scorek*100)
#from wordcloud import WordCloud
#plt.figure(figsize=(20, 20))
#words