# -*- coding: utf-8 -*-
"""
Created on Mon Jun  3 11:01:51 2019

@author: Sahil Nathani
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style(style='darkgrid')

df = pd.read_csv("C:\\Users\\Sahil Nathani\\Desktop\\Python and ML Material\\Databases\\DonalTrumpTweets\\Donald-Tweets!.csv", 
                 encoding="ISO-8859-1")

sns.countplot(x='Type', data=df)
df['Type'].value_counts()

df = df.drop(['Media_Type', 'Tweet_Url', 'Unnamed: 10', 'Unnamed: 11', 'Tweet_Id'], axis=1)

plt.title('Number of tweets in each hour of the day')
df['Time'] = [x.split(':')[0] for x in df['Time']]
sns.countplot(x='Time', data=df)

df_group = df.groupby('Time').agg('sum')['Retweets']
plt.title('Total Retweets for tweets in each hour of the day')
plt.figure(figsize=(10, 10))
df_group.plot.bar()

df = df.drop(['Time', 'Date'], axis=1)

import nltk as nlp
import re

text_list=[]
for i in df['Tweet_Text']:
    text = re.sub("(@[A-Za-z0-9]+)|([^0-9A-Za-z \t])|(\w+:\/\/\S+)", " ", i)
    text = text.lower()
    text = nlp.word_tokenize(text)
    lemma = nlp.WordNetLemmatizer()
    text = [lemma.lemmatize(word) for word in text]
    text = " ".join(text)
    text_list.append(text)

from sklearn.feature_extraction.text import CountVectorizer
max_features=2000000
count_vectorizer= CountVectorizer(max_features=max_features, stop_words="english")
sparce_matrix = count_vectorizer.fit_transform(text_list).toarray()
all_words = count_vectorizer.get_feature_names()
print("Most used words: ", all_words[100:200])

#WordCloud
from wordcloud import WordCloud
plt.subplots(figsize=(12,12))
wordcloud=WordCloud(background_color="white", width=1024, height=768).generate(" ".join(all_words[100:]))
plt.imshow(wordcloud)
plt.axis("off")
plt.show()

from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk.sentiment.util import *
from nltk.corpus import stopwords

sid = SentimentIntensityAnalyzer()

df['sentiment_polarity'] = df.Tweet_Text.apply(lambda x: sid.polarity_scores(x)['compound'])

df.loc[df['sentiment_polarity']>0, 'sentiment'] = 1
df.loc[df['sentiment_polarity']<=0, 'sentiment'] = 0

plt.figure(figsize=(6, 6))
plt.title('Number of Tweets Sentiment-wise')
sns.countplot(x='sentiment', data=df)