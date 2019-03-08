# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
"""reviews = filtered_data["Text"]


reviews = pd.DataFrame(data=reviews)

rvw = reviews[0:10]
lst = list(rvw['Text'][0:5])

lst1 =''.join(lst)


list1 = ['1', '2', '3']
str1 = ''.join(list1)
"""
"""
dp   = detailed page
ASIN = Amazon Standard Identification number
"""

import sqlite3
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import re
import bs4 as bs
import nltk
import string
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.stem.wordnet import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer

from sklearn import metrics
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve, auc


import heapq
import time
from collections import Counter


start_time = time.time()
amazon_review = pd.read_csv('./amazon-fine-food-reviews/Reviews.csv')
end_time = time.time()
print(end_time - start_time)


'--------------------------EDA----------------------------------'

amazon_review.info()

#Total Missing values
amazon_review.isnull().sum().sum()

#Missing values column wise
amazon_review.isnull().sum()

# Missing value Columns in Dataframe
Missing = pd.DataFrame(amazon_review.isnull().sum(), columns = ["count"] )
Missing = Missing[np.logical_not(Missing["count"] == 0)]

Total_rating_count = pd.DataFrame(
                       amazon_review.groupby(['Score']).size().reset_index(name='counts'))

Counter(amazon_review['Score'])

review = amazon_review[amazon_review.Score != 3]
Counter(review['Score'])

len(amazon_review) - len(review)
#42640
len(review)/len(amazon_review) * 100
(1 - len(review)/len(amazon_review))*100


start_time = time.time()  
review['Score'] = review['Score'].apply(lambda x: 'negative' if x < 3 else( 'positive' if x > 3 else 'neutral'))
end_time = time.time()
print(end_time - start_time)

Counter(review['Score'])

'--------------------finding duplicates--------------------'
#Sort
sorted_data=review.sort_values('ProductId', axis=0, ascending=True, 
                              inplace=False, kind='quicksort', na_position='last')
#count
dup_count = sorted_data.duplicated(subset=["UserId","ProfileName","Time","Text"]).sum()
len(sorted_data) - dup_count

# removing duplicates
final=sorted_data.drop_duplicates(subset={"UserId","ProfileName","Time","Text"}, 
                                  keep='first', inplace=False)
len(sorted_data) - len(final)

final.info()
#Missing values column wise
final.isnull().sum()


final.Score.head()
final.shape
final=final[final.HelpfulnessNumerator<=final.HelpfulnessDenominator]
final.shape
Counter(final['Score'])

len(final.loc[final['Score'] == "positive",'Text'])

product_reviewcount = pd.DataFrame(
                       review.groupby(['ProductId','Score']).Score.size().reset_index(name='counts'))

product_reviewcount = product_reviewcount.sort_values(by='counts', ascending=False)

# B007JFMH8M - 857
Counter(final['Score'])
product_rating = final.groupby(['ProductId','Score']).ProductId.count().reset_index(name='counts')
one_prod = final.loc[(final['ProductId'] == 'B007JFMH8M'),['Score','Text']]
Counter(one_prod['Score'] )
one_prod.isnull().sum().sum()

'--------------------------EDA----------------------------------'

negative =  one_prod.loc[(one_prod["Score"] == 'negative'),'Text'].tolist()
positive =  one_prod.loc[(one_prod["Score"] == 'positive'),'Text'].tolist()

negative = ''.join(negative)
positive = ''.join(positive)

'--------------------------Preprocessing---------------------------------'
# removing tags
negative = re.sub(r"<.*?>", '',negative)
positive = re.sub(r"<.*?>", "",positive)
positive = re.sub(r'\[[.]*\]', '.',positive)
          # space

positive = re.sub(r'\W+', ' ',positive)            # non word
positive = re.sub(r'\d', '',positive)



positive = re.sub(r'https?:\/\/.*[\r\n]*', '', positive, flags=re.MULTILINE)
positive = re.sub(r'\<a href', ' ', positive)
positive = re.sub(r'&amp;', '', positive) 
positive = re.sub(r'<.*?>', ' ', positive)
positive = re.sub(r'[_"\-;%()|+&=*%,!?:#$@/]', ' ', positive)
positive = re.sub(r'\[[.]*\]', '.',positive)

positive = re.sub(r'\'', ' ', positive)
positive = re.sub(r'\.\.+', ' ', positive)
positive = re.sub(r'\d+(\.\d*)?', ' ',positive)
positive = re.sub(r"\s\s+", ' ',positive)
positive = positive.lower()

negative = re.sub(r'\d+(\.\d*)?', ' ',negative)
'--------------------------Preprocessing---------------------------------'
clean_text = re.sub(r'\W+', ' ',positive)  

sentence = nltk.sent_tokenize(positive)

stops = set(stopwords.words("english"))
text = [w for w in text if not w in stops]


word2count = {}
for word in nltk.word_tokenize(clean_text):
    if word not in stops:
        if word not in word2count.keys():
            word2count[word] = 1
        else:
            word2count[word] += 1

for key in word2count.keys():
    word2count[key] = word2count[key]/len(word2count.values())


sent2score={}
for sentences in sentence:
    for word in nltk.word_tokenize(sentences.lower()):
        if word in word2count.keys():
            if len(sentences.split(' ')) < 30:
                if sentences not in sent2score.keys():
                    sent2score[sentences] = word2count[word]
                else:
                    sent2score[sentences] += word2count[word]



positive_summary = heapq.nlargest(5,sent2score, key=sent2score.get)
positive_summary = ''.join(positive_summary)



s = "~~~~sqeeze me 254 ...i. .... 10 1.4845 256.2 258888.200000111"

# Get Result
re.sub(r'\.\.+', ' ', s)
s.rstrip("..")
re.sub(r'\d+(\.\d*)?', 'c',s)










s = re.sub('[^A-Za-z]+', '', sen)
re.sub(r'\W', ' ', sen)
re.sub(r'\d', ' ', sen)
sen= "a13@ gh 12"


































