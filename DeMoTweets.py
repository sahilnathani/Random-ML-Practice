# -*- coding: utf-8 -*-
"""
Created on Sat Feb  2 11:24:58 2019

@author: Sahil Nathani
"""

import pandas as pd
import re
import matplotlib.pyplot as plt
import seaborn as sns
from subprocess import check_output
import nltk
#nltk.download('vader_lexicon')
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk.sentiment.util import *
from nltk import tokenize
from bs4 import BeautifulSoup
from nltk.corpus import stopwords

tweets = pd.read_csv('C:\\Users\\Sahil Nathani\\Desktop\\Python and ML Material\\Databases\\DeMo Tweets\\demonetization-tweets.csv', encoding='ISO-8859-1')

sid = SentimentIntensityAnalyzer()

def clean(x):
    #Remove Html  
    x = BeautifulSoup(x).get_text()
    
    #Remove Non-Letters
    x = re.sub('[^a-zA-Z]',' ',x)
    
    #Convert to lower_case and split
    x = x.lower().split()
    
    #Remove stopwords
    stop = set(stopwords.words('english'))
    words = [w for w in x if not w in stop]
    
    #join the words back into one string
    return(' '.join(words))
    
tweets['sentiment_polarity'] = tweets.text.apply(lambda x: sid.polarity_scores(x)['compound'])
tweets['negative_sentiment'] = tweets.text.apply(lambda x: sid.polarity_scores(x)['neg'])
tweets['positive_sentiment'] = tweets.text.apply(lambda x: sid.polarity_scores(x)['pos'])
tweets['neutral_sentiment'] = tweets.text.apply(lambda x: sid.polarity_scores(x)['neu'])

tweets['sentiments'] = ''
tweets.loc[tweets['sentiment_polarity']>0, 'sentiments']=1
tweets.loc[tweets['sentiment_polarity']==0, 'sentiments']=0
tweets.loc[tweets['sentiment_polarity']<0, 'sentiments']=-1

tweets.sentiments.value_counts().plot(kind='bar')   

plt.figure(figsize=(6, 6))
from matplotlib.pyplot import style
style.use('fivethirtyeight')

tweets['hour'] = pd.DatetimeIndex(tweets['created']).hour
tweets['date'] = pd.DatetimeIndex(tweets['created']).date
tweets['minute'] = pd.DatetimeIndex(tweets['created']).minute
df=(tweets.groupby('hour',as_index=False).sentiment_polarity.mean())
sns.lineplot(x='hour', y='sentiment_polarity', data=df)
plt.title('Sentiment of a Tweet According to Time')