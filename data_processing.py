import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from bs4 import BeautifulSoup

plt.style.use('fivethirtyeight')

cols = ['sentiment', 'id', 'date', 'query_string', 'user', 'text']
df = pd.read_csv("training.1600000.processed.noemoticon.csv", header = None, names = cols, encoding = 'latin1')
#print(df.head()) #prints first 5 rows/entries
#print(df.info()) #prints data columns, their counts, and types
#print(df.sentiment.value_counts()) #prints value counts -> there are no nulls
#print(df.query_string.value_counts())

df.drop(['id', 'date', 'user'], axis = 1, inplace = True) #drops unecessary columns
#print(df.head())
#print(df[df.sentiment == 0].index) #first 800000
#print(df[df.sentiment == 4].index) #last 800000
df['sentiment'] = df['sentiment'].map({0: 0, 4: 1}) #maps 0 -> 0, 4 -> 1

df['pre_clean_len'] = [len(t) for t in df.text]
data_dict = {
    'sentiment': {
        'type': df.sentiment.dtype,
        'description': 'sentiment class - 0: negative, 1: positive'
    },
    'text': {
        'type': df.text.dtype,
        'description': 'tweet text'
    },
    'pre_clean_len': {
        'type': df.pre_clean_len.dtype,
        'description': 'length of tweet before cleaning'
    },
    'dataset_shape': df.shape
}

fig, ax = plt.subplots(figsize = (5, 5))
plt.boxplot(df.pre_clean_len)
plt.show() #reveals that lens are more than 140

print(df[df.pre_clean_len > 140].head(10))
