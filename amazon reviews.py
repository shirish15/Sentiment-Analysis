#!/usr/bin/env python
# coding: utf-8

# In[4]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns
import plotly.express as px

from tqdm import tqdm
from datetime import datetime
import nltk
import csv
import os
import re
import time
import random
from bs4 import BeautifulSoup
from gensim.parsing.preprocessing import remove_stopwords
from nltk import word_tokenize
from nltk.stem import PorterStemmer,WordNetLemmatizer,SnowballStemmer
import string
from gensim.models import KeyedVectors,Word2Vec

from sklearn.model_selection import KFold,train_test_split,GridSearchCV,RandomizedSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.feature_extraction.text import CountVectorizer,TfidfVectorizer
from sklearn.metrics import confusion_matrix,log_loss,f1_score,accuracy_score,classification_report
from sklearn.linear_model import LogisticRegression,SGDClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier,GradientBoostingClassifier,VotingClassifier

import xgboost as xgb
import lightgbm as lgb

import tensorflow as tf
from tensorflow import keras 
from keras import backend as K
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.preprocessing.text import Tokenizer

import warnings
warnings.filterwarnings('ignore')

sns.set_style('dark')
nltk.download('wordnet')

seed = 42


# In[ ]:


Data = pd.read_csv(r'C:\Users\Lenovo\Desktop\ML\Amazon-Fine-Food-Reviews\Reviews.csv')
Data.head(3)


# In[ ]:


print('Number of datapoints in the dataset : ',Data.shape[0])
print('Number of attributes in the dataset : ',Data.shape[1])
print('Attribute names : \n',Data.columns.values)


# In[ ]:


print('Number of null values in dataset : ',Data.isnull().sum())


# We can see that the data has 16 NULL values in the Profile name attribute and 27 NULL values in the Summary

# In[ ]:


Data.info()


# In[ ]:


Data.describe()


# In[ ]:


print('Number of duplicate values in the dataset : ',Data.duplicated().sum())
Data = Data.drop_duplicates(keep='first')


# In[ ]:


Data = pd.DataFrame.dropna(Data)
Data.info()


# In[ ]:


Data = Data[Data['HelpfulnessNumerator'] <= Data['HelpfulnessDenominator']]
Data.head()


# In[ ]:


dt_object = [None] * len(Data['Time'])
k=0
for i in tqdm(Data['Time']):
    dt_object[k] = datetime.fromtimestamp(i)
    k += 1
Data['time'] = dt_object
Data['time'][:10]


# In[ ]:


Data = Data.sort_values(by='time').reset_index(drop=True)
Data.head(3)


# Analysis of data  :

# In[ ]:


plt.figure(figsize=(15,6))
ax = sns.countplot(Data['Score'],
                   saturation=0.82,
                   palette=['red','cyan','black','silver','gold'])
plt.title('Count of number of individual Scores in the dataset',fontsize=17)
plt.ylabel('Count of the Scores',fontsize=16)
plt.xlabel('Scores',fontsize=16)
plt.xticks(fontsize=15)
for p in ax.patches:
    height = p.get_height()
    ax.text(p.get_x()+p.get_width()/2.,
            height + 3,
            '{:1.2f}%'.format(height/len(Data) * 100),
            ha="center") 
plt.grid()
plt.show()


# In[ ]:


Data.loc[Data['Score'] == 1,'Score'] = 1
Data.loc[Data['Score'] == 2,'Score'] = 1
Data.loc[Data['Score'] == 3,'Score'] = 2
Data.loc[Data['Score'] == 4,'Score'] = 3
Data.loc[Data['Score'] == 5,'Score'] = 3


# In[ ]:


Data.Score.value_counts()


# In[ ]:


plt.figure(figsize=(15,6))
ax = sns.countplot(Data['Score'],
                   saturation=0.82,
                   palette=['greenyellow','pink','orange'])
plt.title('Count of number of individual sentiments in the dataset',fontsize=17)
plt.ylabel('Count of the sentiment of reviews',fontsize=16)
plt.xlabel('Sentiments',fontsize=16)
plt.xticks(fontsize=15)
for p in ax.patches:
    height = p.get_height()
    ax.text(p.get_x()+p.get_width()/2.,
            height + 4,
            '{:1.2f}%'.format(height/len(Data) * 100),
            ha="center")
plt.grid()
plt.show()


# Preprocessing : 

# In[ ]:


# printing some random reviews
sent_2773 = Data['Text'].values[2773]
print('Review 2773 : \n',sent_2773)
print("="*125)

sent_7530 = Data['Text'].values[7530]
print('Review 7530 : \n',sent_7530)
print("="*125)

sent_1500 = Data['Text'].values[1500]
print('Review 1500 : \n',sent_1500)
print("="*125)

sent_49065 = Data['Text'].values[49065]
print('Review 49065 : \n',sent_49065)
print("="*125)


# In[ ]:


# remove urls from text python: https://stackoverflow.com/a/40823105/4084039
sent_2773 = re.sub(r"http\S+", "", sent_2773)
sent_7530 = re.sub(r"http\S+", "", sent_7530)
sent_1500 = re.sub(r"http\S+", "", sent_1500)
sent_49065 = re.sub(r"http\S+", "", sent_49065)

print(sent_2773)
print('*'*125)
print(sent_7530)
print('*'*125)
print(sent_1500)
print('*'*125)
print(sent_49065)


# In[ ]:


soup = BeautifulSoup(sent_2773)
text = soup.get_text()
print(text)
print("="*50)

soup = BeautifulSoup(sent_7530)
text = soup.get_text()
print(text)
print("="*50)

soup = BeautifulSoup(sent_1500)
text = soup.get_text()
print(text)
print("="*50)

soup = BeautifulSoup(sent_49065)
text = soup.get_text()
print(text)


# In[ ]:


data = Data.head(10000)


# In[ ]:


def remove_shortforms(phrase):
    phrase = re.sub(r"won't", "will not", phrase)
    phrase = re.sub(r"can\'t", "can not", phrase)
    phrase = re.sub(r"n\'t", " not", phrase)
    phrase = re.sub(r"\'re", " are", phrase)
    phrase = re.sub(r"\'s", " is", phrase)
    phrase = re.sub(r"\'d", " would", phrase)
    phrase = re.sub(r"\'ll", " will", phrase)
    phrase = re.sub(r"\'t", " not", phrase)
    phrase = re.sub(r"\'ve", " have", phrase)
    text = re.sub(r"\'m", " am", phrase)
    return text

def remove_html(phrase):
    soup = BeautifulSoup(phrase)
    text = soup.get_text()
    return text

def remove_special_char(text):
    text = re.sub('[^A-Za-z0-9]+'," ",text)
    return text

def remove_wordswithnum(text):
    text = re.sub("\S*\d\S*", "", text).strip()
    return text

def lowercase(text):
    text = text.lower()
    return text

def remove_stop_words(text):
    text = remove_stopwords(text)
    return text

st = SnowballStemmer(language='english')
def stemming(text):
    r= []
    for word in text :
        a = st.stem(word)
        r.append(a)
    return r

def listToString(s):  
    str1 = " "   
    return (str1.join(s))

def remove_punctuations(text):
    text = re.sub(r'[^\w\s]','',text)
    return text

def remove_links(text):
    text = re.sub(r'http\S+', '', text)
    return text

lemmatizer = WordNetLemmatizer()
def lemmatize_words(text):
    text = lemmatizer.lemmatize(text)
    return text


# In[ ]:


start_time = time.time()

text = data['Text'].apply(lambda x : remove_shortforms(x))
text = text.apply(lambda x : remove_html(x))
text = text.apply(lambda x : remove_special_char(x))
text = text.apply(lambda x : remove_wordswithnum(x))
text = text.apply(lambda x : lowercase(x))
text = text.apply(lambda x : remove_stop_words(x))
text = text.apply(lambda x : listToString(stemming(x.split())))
text = text.apply(lambda x : remove_punctuations(x))
text = text.apply(lambda x : remove_links(x))
text = text.apply(lambda x : lemmatize_words(x))

print('Preprocessing of text done in : ',np.round(time.time() - start_time,3), 'seconds')
text[:4]


# In[ ]:


data['preprocessed_text'] = text
data.head()


# In[ ]:


start_time = time.time()

text = data['Summary'].apply(lambda x : remove_shortforms(x))
text = text.apply(lambda x : remove_html(x))
text = text.apply(lambda x : remove_special_char(x))
text = text.apply(lambda x : remove_wordswithnum(x))
text = text.apply(lambda x : lowercase(x))
text = text.apply(lambda x : remove_stop_words(x))
text = text.apply(lambda x : listToString(stemming(x.split())))
text = text.apply(lambda x : remove_punctuations(x))
text = text.apply(lambda x : remove_links(x))
text = text.apply(lambda x : lemmatize_words(x))

print('Preprocessing of summary done in : ',np.round(time.time() - start_time,3), 'seconds')
text[:4]


# In[ ]:


data['preprocessed_summary'] = text
data.head()


# In[ ]:


data = data.reset_index(drop=True)


# In[ ]:


actual_data = [None] * len(data)
for i in range(len(data)):
    actual_data[i] = data['preprocessed_text'][i] +' '+ data['preprocessed_summary'][i] +' '+ data['preprocessed_summary'][i] +' '+ data['preprocessed_summary'][i]
actual_data[:2]


# Bag of words :

# In[ ]:


start_time = time.time()
bow = CountVectorizer(ngram_range=(1,3), min_df=10)
bow.fit(actual_data)
print('Some feature names in Bag of Words : ',bow.get_feature_names()[:10])
print('='*125)

bow_text = bow.transform(actual_data)
print('Time taken to train BOW model : ',np.round(time.time()-start_time,3),' seconds')
print("The type of count vectorizer ",type(bow_text))
print("The shape of text BOW vectorizer ",bow_text.get_shape())
print("The number of unique words ", bow_text.get_shape()[1])


# TF-IDF :

# In[ ]:


start_time = time.time()
tfidf = TfidfVectorizer(ngram_range=(1,3), min_df=10)
tfidf.fit(actual_data)
print("Some sample features in TF-IDF : ",tfidf.get_feature_names()[0:10])
print('='*125)

tfidf_text = tfidf.transform(actual_data)
print('Time taken to train TF-IDF model : ',np.round(time.time()-start_time,3),' seconds')
print("the type of count vectorizer ",type(tfidf_text))
print("the shape of out text TFIDF vectorizer ",tfidf_text.get_shape())
print("the number of unique words including both unigrams and bigrams ", tfidf_text.get_shape()[1])


# GLOVE Vectors : 

# In[ ]:


# print('Loading GloVe vectors...')
# glove = {}
# with open(os.path.join('../input/glove-global-vectors-for-word-representation/glove.6B.200d.txt'), encoding = "utf-8") as f:
#     for line in f:
#         values = line.split()
#         word = values[0]
#         vec = np.asarray(values[1:], dtype='float32')
#         glove[word] = vec
# print(f'Found {len(glove)} word vectors.')


# In[ ]:


# tokenizer = Tokenizer()
# tokenizer.fit_on_texts(actual_data)
# sequences = tokenizer.texts_to_sequences(actual_data)
# word2index = tokenizer.word_index
# print("Number of unique tokens : ",len(word2index))


# In[ ]:


# data_padded = pad_sequences(sequences,1500)
# print(data_padded.shape)
# print(data_padded[0])


# In[ ]:


# embedding_matrix = np.zeros((len(word2index)+1,200))

# embedding_vec=[]
# for word, i in tqdm(word2index.items()):
#     embedding_vec = word2vec.get(word)
#     if embedding_vec is not None:
#         embedding_matrix[i] = embedding_vec
# print(embedding_matrix.shape)
# print(embedding_matrix[1])


# In[ ]:


labels = data['Score']
labels.shape


# Time based splitting :

# In[ ]:


bow_text.shape[0]


# In[ ]:


m = int(bow_text.shape[0]*0.8)
m


# In[ ]:


train_bow = bow_text[:m]
test_bow = bow_text[m:]
train_labels = labels[:m]
test_labels = labels[m:]
train_tfidf = tfidf_text[:m]
test_tfidf = tfidf_text[m:]


# Class distribution :

# In[ ]:


train_class_distribution = train_labels.value_counts()
test_class_distribution = test_labels.value_counts()
index = test_class_distribution.index.values

my_colors = ['r','b','k']
train_class_distribution.plot(kind='bar',color=my_colors)
plt.xlabel('Class')
plt.ylabel('Data points per Class')
plt.title('Distribution of yi in train data')
plt.grid()
plt.show()

sorted_yi = np.argsort(-train_class_distribution.values)
for i in sorted_yi:
    print('Number of data points in class', index[i], ':',train_class_distribution.values[i], '(', np.round((train_class_distribution.values[i]/train_bow.shape[0]*100), 3), '%)')

    
print('-'*80)
my_colors = ['y','m','c']
test_class_distribution.plot(kind='bar',color=my_colors)
plt.xlabel('Class')
plt.ylabel('Data points per Class')
plt.title('Distribution of yi in test data')
plt.grid()
plt.show()


sorted_yi = np.argsort(-test_class_distribution.values)
for i in sorted_yi:
    print('Number of data points in class', index[i], ':',test_class_distribution.values[i], '(', np.round((test_class_distribution.values[i]/test_bow.shape[0]*100), 3), '%)')


# In[ ]:


def plot_confusion_matrix(test_y,predicted_y):
    C = confusion_matrix(test_y,predicted_y)
    A =(((C.T)/(C.sum(axis=1))).T)
    B =(C/C.sum(axis=0))
    class_labels = [1,2,3]
    
    print('\n')
    print("-"*50, "Confusion matrix", "-"*50)
    plt.figure(figsize=(16,6))
    sns.heatmap(C, annot=True, cmap="viridis", fmt=".3f", xticklabels=class_labels, yticklabels=class_labels)
    plt.xlabel('Predicted Class',fontsize=15)
    plt.ylabel('Original Class',fontsize=15)
    plt.show()
    
    print('\n')
    print("-"*50, "Precision matrix", "-"*50)
    plt.figure(figsize=(16,6))
    sns.heatmap(B, annot=True, cmap="RdBu", fmt=".3f", xticklabels=class_labels, yticklabels=class_labels)
    plt.xlabel('Predicted Class',fontsize=15)
    plt.ylabel('Original Class',fontsize=15)
    plt.show()
    
    print('\n')
    print("-"*50, "Recall matrix", "-"*50)
    plt.figure(figsize=(16,6))
    sns.heatmap(A, annot=True, cmap="BuGn", fmt=".3f", xticklabels=class_labels, yticklabels=class_labels)
    plt.xlabel('Predicted Class',fontsize=15)
    plt.ylabel('Original Class',fontsize=15)
    plt.show()


# Random Model : 

# In[ ]:


train_temp,val_temp,train_labels_temp,val_labels_temp = train_test_split(train_bow,train_labels,test_size=0.2)

val_predictions = np.zeros((val_temp.shape[0],3))
for i in range(val_temp.shape[0]):
    rand_probs = np.random.rand(1,3)
    val_predictions[i] = (rand_probs/sum(sum(rand_probs)))[0]
print("Log loss on Validation Data using Random Model : ",log_loss(val_labels_temp,val_predictions))

test_predictions = np.zeros((test_bow.shape[0],3))
for i in range(test_bow.shape[0]):
    rand_probs = np.random.rand(1,3)
    test_predictions[i] = ((rand_probs/sum(sum(rand_probs)))[0])
print("Log loss on Test Data using Random Model : ",log_loss(test_labels,test_predictions))

test_labels = test_labels.reset_index(drop=True)
label = [None]*len(test_predictions)
count=0
for k in range(len(test_predictions)):
    value = test_predictions[k].max()
    for i in range(3):
        if test_predictions[k][i] == value:
            label[k] = i+1
            if label[k] != test_labels[k]:
                count = count+1
predictions = label
acc_random = accuracy_score(test_labels,predictions)
print('Accuracy of Random Model : ',acc_random)
miss_random = count
print('Number of Missclassified points : ',count)
loss_random = log_loss(test_labels,test_predictions)
predicted_y =np.argmax(test_predictions, axis=1)
plot_confusion_matrix(test_labels, predicted_y+1)


# Logistic Regression (without class balancing) :

# In[ ]:


kfold = KFold(n_splits=4)
loss_per_fold = []
best_loss_index = []

for n,(train_idx,val_idx) in enumerate(kfold.split(train_bow,train_labels)):
    
    loss_per_C = []
    
    print('Fold started : ',n)
    train_temp = train_bow[train_idx]
    val_temp = train_bow[val_idx]    
    train_labels_temp = train_labels[train_idx]    
    val_labels_temp = train_labels[val_idx]
    
    C = [0.001,0.01,0.1,1,10,100,1000]
              
    for i in C:
        lr= LogisticRegression(C=i,penalty='l2',max_iter=10000)
        lr.fit(train_temp,train_labels_temp)
        calib_lr = CalibratedClassifierCV(lr)
        calib_lr.fit(train_temp,train_labels_temp)
        val_predictions = calib_lr.predict_proba(val_temp)
        logloss = log_loss(val_labels_temp,val_predictions)
        print(f'Log loss for fold {n} for logistic regression with C equal to {i} with BOW data is {logloss}')
        loss_per_C.append(logloss)
        
    min_loss = min(loss_per_C)
    best_loss_index.append(np.argmin(loss_per_C))
    print(f'Minimum validation log loss for fold {n} is {min_loss}')
    loss_per_fold.append(min_loss)
    
    fig, ax = plt.subplots(figsize=(12,4))
    ax.plot(C, loss_per_C,c='r')
    for i, txt in enumerate(np.round(loss_per_C,3)):
        ax.annotate((C[i],np.round(txt,3)), (C[i],loss_per_C[i]))
    plt.grid()
    plt.title(f'Cross Validation loss for each C for Fold {n}',fontsize=14)
    plt.xlabel("C's",fontsize=12)
    plt.ylabel("Loss measure",fontsize=12)
    plt.show()
    print('*'*125)
    
N = [0,1,2,3]
fig, ax = plt.subplots(figsize=(15,5))
ax.plot(N, loss_per_fold,c='g')
for i, txt in enumerate(np.round(loss_per_fold,3)):
    ax.annotate((N[i],np.round(txt,3)), (N[i],loss_per_fold[i]))
plt.grid()
plt.title(f'Cross Validation loss for each Fold',fontsize=15)
plt.xlabel("Fold number",fontsize=13)
plt.ylabel("Loss measure",fontsize=13)
plt.show()

predict_train = calib_lr.predict_proba(train_bow)
print("The train log loss is : ",log_loss(train_labels, predict_train, labels=calib_lr.classes_, eps=1e-15))
predict_y = calib_lr.predict_proba(test_bow)
print("The test log loss is : ",log_loss(test_labels, predict_y, labels=calib_lr.classes_, eps=1e-15))

predict = calib_lr.predict(test_bow)
acc_lr_bow = accuracy_score(test_labels,predict)
print('Accuracy :',acc_lr_bow)
miss_lr_bow = sum(predict!=test_labels)
print('Number of missclassified points : ',sum(predict!=test_labels))
print('Classification  Report :\n',classification_report(test_labels,predict))
loss_lr_bow = log_loss(test_labels, predict_y, labels=calib_lr.classes_, eps=1e-15)
predicted_y = np.argmax(predict_y, axis=1)
plot_confusion_matrix(test_labels,predicted_y+1)


# In[ ]:


kfold = KFold(n_splits=4)
loss_per_fold = []
best_loss_index = []

for n,(train_idx,val_idx) in enumerate(kfold.split(train_tfidf,train_labels)):
    
    loss_per_C = []
    
    print('Fold started : ',n)
    train_temp = train_tfidf[train_idx]
    val_temp = train_tfidf[val_idx]    
    train_labels_temp = train_labels[train_idx]    
    val_labels_temp = train_labels[val_idx]
    
    C = [0.001,0.01,0.1,1,10,100,1000]
              
    for i in C:
        lr= LogisticRegression(C=i,penalty='l2',max_iter=10000)
        lr.fit(train_temp,train_labels_temp)
        calib_lr = CalibratedClassifierCV(lr)
        calib_lr.fit(train_temp,train_labels_temp)
        val_predictions = calib_lr.predict_proba(val_temp)
        logloss = log_loss(val_labels_temp,val_predictions)
        print(f'Log loss for fold {n} for logistic regression with C equal to {i} with TFIDF data is {logloss}')
        loss_per_C.append(logloss)
        
    min_loss = min(loss_per_C)
    best_loss_index.append(np.argmin(loss_per_C))
    print(f'Minimum validation log loss for fold {n} is {min_loss}')
    loss_per_fold.append(min_loss)
    
    fig, ax = plt.subplots(figsize=(12,4))
    ax.plot(C, loss_per_C,c='r')
    for i, txt in enumerate(np.round(loss_per_C,3)):
        ax.annotate((C[i],np.round(txt,3)), (C[i],loss_per_C[i]))
    plt.grid()
    plt.title(f'Cross Validation loss for each C for Fold {n}',fontsize=14)
    plt.xlabel("C's",fontsize=12)
    plt.ylabel("Loss measure",fontsize=12)
    plt.show()
    print('*'*125)
    
N = [0,1,2,3]
fig, ax = plt.subplots(figsize=(15,5))
ax.plot(N, loss_per_fold,c='g')
for i, txt in enumerate(np.round(loss_per_fold,3)):
    ax.annotate((N[i],np.round(txt,3)), (N[i],loss_per_fold[i]))
plt.grid()
plt.title(f'Cross Validation loss for each Fold',fontsize=15)
plt.xlabel("Fold number",fontsize=13)
plt.ylabel("Loss measure",fontsize=13)
plt.show()

predict_train = calib_lr.predict_proba(train_tfidf)
print("The train log loss is : ",log_loss(train_labels, predict_train, labels=calib_lr.classes_, eps=1e-15))
predict_y = calib_lr.predict_proba(test_tfidf)
print("The test log loss is : ",log_loss(test_labels, predict_y, labels=calib_lr.classes_, eps=1e-15))

predict = calib_lr.predict(test_tfidf)
acc_lr_tfidf = accuracy_score(test_labels,predict)
print('Accuracy :',acc_lr_tfidf)
miss_lr_tfidf = sum(predict!=test_labels)
print('Number of missclassified points : ',sum(predict!=test_labels))
print('Classification  Report :\n',classification_report(test_labels,predict))
loss_lr_tfidf = log_loss(test_labels, predict_y, labels=calib_lr.classes_, eps=1e-15)
predicted_y = np.argmax(predict_y, axis=1)
plot_confusion_matrix(test_labels,predicted_y+1)


# Logistic Regression (with class balancing) :

# In[ ]:


kfold = KFold(n_splits=4)
loss_per_fold = []
best_loss_index = []

for n,(train_idx,val_idx) in enumerate(kfold.split(train_bow,train_labels)):
    
    loss_per_C = []
    
    print('Fold started : ',n)
    train_temp = train_bow[train_idx]
    val_temp = train_bow[val_idx]    
    train_labels_temp = train_labels[train_idx]    
    val_labels_temp = train_labels[val_idx]
    
    C = [0.01,0.1,1,10,50]
              
    for i in C:
        lr= LogisticRegression(C=i,class_weight='balanced',penalty='l2',max_iter=10000)
        lr.fit(train_temp,train_labels_temp)
        calib_lr = CalibratedClassifierCV(lr)
        calib_lr.fit(train_temp,train_labels_temp)
        val_predictions = calib_lr.predict_proba(val_temp)
        logloss = log_loss(val_labels_temp,val_predictions)
        print(f'Log loss for fold {n} for logistic regression with C equal to {i} with BOW data is {logloss}')
        loss_per_C.append(logloss)
        
    min_loss = min(loss_per_C)
    best_loss_index.append(np.argmin(loss_per_C))
    print(f'Minimum validation log loss for fold {n} is {min_loss}')
    loss_per_fold.append(min_loss)
    
    fig, ax = plt.subplots(figsize=(12,4))
    ax.plot(C, loss_per_C,c='r')
    for i, txt in enumerate(np.round(loss_per_C,3)):
        ax.annotate((C[i],np.round(txt,3)), (C[i],loss_per_C[i]))
    plt.grid()
    plt.title(f'Cross Validation loss for each C for Fold {n}',fontsize=14)
    plt.xlabel("C's",fontsize=12)
    plt.ylabel("Loss measure",fontsize=12)
    plt.show()
    print('*'*125)
    
N = [0,1,2,3]
fig, ax = plt.subplots(figsize=(15,5))
ax.plot(N, loss_per_fold,c='g')
for i, txt in enumerate(np.round(loss_per_fold,3)):
    ax.annotate((N[i],np.round(txt,3)), (N[i],loss_per_fold[i]))
plt.grid()
plt.title(f'Cross Validation loss for each Fold',fontsize=15)
plt.xlabel("Fold number",fontsize=13)
plt.ylabel("Loss measure",fontsize=13)
plt.show()

predict_train = calib_lr.predict_proba(train_bow)
print("The train log loss is : ",log_loss(train_labels, predict_train, labels=calib_lr.classes_, eps=1e-15))
predict_y = calib_lr.predict_proba(test_bow)
print("The test log loss is : ",log_loss(test_labels, predict_y, labels=calib_lr.classes_, eps=1e-15))

predict = calib_lr.predict(test_bow)
acc_lr_bow_balanced = accuracy_score(test_labels,predict)
print('Accuracy :',acc_lr_bow_balanced)
miss_lr_bow_balanced = sum(predict!=test_labels)
print('Number of missclassified points : ',sum(predict!=test_labels))
print('Classification  Report :\n',classification_report(test_labels,predict))
loss_lr_bow_balanced = log_loss(test_labels, predict_y, labels=calib_lr.classes_, eps=1e-15)
predicted_y = np.argmax(predict_y, axis=1)
plot_confusion_matrix(test_labels,predicted_y+1)


# In[ ]:


kfold = KFold(n_splits=4)
loss_per_fold = []
best_loss_index = []

for n,(train_idx,val_idx) in enumerate(kfold.split(train_tfidf,train_labels)):
    
    loss_per_C = []
    
    print('Fold started : ',n)
    train_temp = train_tfidf[train_idx]
    val_temp = train_tfidf[val_idx]    
    train_labels_temp = train_labels[train_idx]    
    val_labels_temp = train_labels[val_idx]
    
    C = [0.01,0.1,1,10,50]
              
    for i in C:
        lr= LogisticRegression(C=i,class_weight='balanced',penalty='l2',max_iter=10000)
        lr.fit(train_temp,train_labels_temp)
        calib_lr = CalibratedClassifierCV(lr)
        calib_lr.fit(train_temp,train_labels_temp)
        val_predictions = calib_lr.predict_proba(val_temp)
        logloss = log_loss(val_labels_temp,val_predictions)
        print(f'Log loss for fold {n} for logistic regression with C equal to {i} with TFIDF data is {logloss}')
        loss_per_C.append(logloss)
        
    min_loss = min(loss_per_C)
    best_loss_index.append(np.argmin(loss_per_C))
    print(f'Minimum validation log loss for fold {n} is {min_loss}')
    loss_per_fold.append(min_loss)
    
    fig, ax = plt.subplots(figsize=(12,4))
    ax.plot(C, loss_per_C,c='r')
    for i, txt in enumerate(np.round(loss_per_C,3)):
        ax.annotate((C[i],np.round(txt,3)), (C[i],loss_per_C[i]))
    plt.grid()
    plt.title(f'Cross Validation loss for each C for Fold {n}',fontsize=14)
    plt.xlabel("C's",fontsize=12)
    plt.ylabel("Loss measure",fontsize=12)
    plt.show()
    print('*'*125)
    
N = [0,1,2,3]
fig, ax = plt.subplots(figsize=(15,5))
ax.plot(N, loss_per_fold,c='g')
for i, txt in enumerate(np.round(loss_per_fold,3)):
    ax.annotate((N[i],np.round(txt,3)), (N[i],loss_per_fold[i]))
plt.grid()
plt.title(f'Cross Validation loss for each Fold',fontsize=15)
plt.xlabel("Fold number",fontsize=13)
plt.ylabel("Loss measure",fontsize=13)
plt.show()

predict_train = calib_lr.predict_proba(train_tfidf)
print("The train log loss is : ",log_loss(train_labels, predict_train, labels=calib_lr.classes_, eps=1e-15))
predict_y = calib_lr.predict_proba(test_tfidf)
print("The test log loss is : ",log_loss(test_labels, predict_y, labels=calib_lr.classes_, eps=1e-15))

predict = calib_lr.predict(test_tfidf)
acc_lr_tfidf_balanced = accuracy_score(test_labels,predict)
print('Accuracy :',acc_lr_tfidf_balanced)
miss_lr_tfidf_balanced = sum(predict!=test_labels)
print('Number of missclassified points : ',sum(predict!=test_labels))
print('Classification  Report :\n',classification_report(test_labels,predict))
loss_lr_tfidf_balanced = log_loss(test_labels, predict_y, labels=calib_lr.classes_, eps=1e-15)
predicted_y = np.argmax(predict_y, axis=1)
plot_confusion_matrix(test_labels,predicted_y+1)


# Simple CV strategy : 

# In[ ]:


m = int(bow_text.shape[0]*0.8)
v = int(m*0.8)
m,v


# In[ ]:


train_bow = bow_text[:m]
test_bow = bow_text[m:]
train_labels = labels[:m]
test_labels = labels[m:]
train_tfidf = tfidf_text[:m]
test_tfidf = tfidf_text[m:]

cv_bow = train_bow[v:]
train_bow = train_bow[:v]
cv_tfidf = train_tfidf[v:]
train_tfidf = train_tfidf[:v]
cv_labels = train_labels[v:]
train_labels = train_labels[:v]


# In[ ]:


train_labels.shape,cv_labels.shape,test_labels.shape


# In[ ]:


train_bow.shape,cv_bow.shape,test_bow.shape


# In[ ]:


train_tfidf.shape,cv_tfidf.shape,test_tfidf.shape


# In[ ]:


train_class_distribution = train_labels.value_counts()
cv_class_distribution = cv_labels.value_counts()
test_class_distribution = test_labels.value_counts()
index = train_class_distribution.index.values

my_colors = ['r','b','k']
train_class_distribution.plot(kind='bar',color=my_colors)
plt.xlabel('Class')
plt.ylabel('Data points per Class')
plt.title('Distribution of yi in train data')
plt.grid()
plt.show()

sorted_yi = np.argsort(-train_class_distribution.values)
for i in sorted_yi:
    print('Number of data points in class', index[i], ':',train_class_distribution.values[i], '(', np.round((train_class_distribution.values[i]/train_bow.shape[0]*100), 3), '%)')

    
print('-'*80)
my_colors = ['y','m','c']
cv_class_distribution.plot(kind='bar',color=my_colors)
plt.xlabel('Class')
plt.ylabel('Data points per Class')
plt.title('Distribution of yi in CV data')
plt.grid()
plt.show()


sorted_yi = np.argsort(-test_class_distribution.values)
for i in sorted_yi:
    print('Number of data points in class', index[i], ':',cv_class_distribution.values[i], '(', np.round((cv_class_distribution.values[i]/cv_bow.shape[0]*100), 3), '%)')

print('-'*80)
my_colors = ['b','k','y']
test_class_distribution.plot(kind='bar',color=my_colors)
plt.xlabel('Class')
plt.ylabel('Data points per Class')
plt.title('Distribution of yi in test data')
plt.grid()
plt.show()


sorted_yi = np.argsort(-test_class_distribution.values)
for i in sorted_yi:
    print('Number of data points in class', index[i], ':',test_class_distribution.values[i], '(', np.round((test_class_distribution.values[i]/test_bow.shape[0]*100), 3), '%)')


# RBF SVM : 

# In[ ]:


# params = {
#     'C':[0.01,0.1,1]
# #     'gamma':['scale','auto'],    
#          }

# svm = SVC(kernel='rbf',probability=True,class_weight='balanced')
# svm.fit(train_bow,train_labels)
# clf = GridSearchCV(svm,param_grid=params,scoring='f1_micro',cv=5,verbose=2,n_jobs=-1)
# clf.fit(train_bow,train_labels)

# print('Best Parameters : \n',clf.best_params_)
# print('Best score : \n',clf.best_score_)
# print('Best Estimator : \n',clf.best_estimator_)

# best_params = clf.best_params_
svm = SVC(kernel='rbf',probability=True,C=1,gamma='auto',class_weight='balanced')
svm.fit(train_bow,train_labels)

predict_train = svm.predict_proba(train_bow)
print("The train log loss is : ",log_loss(train_labels, predict_train, labels=svm.classes_, eps=1e-15))
predict_cv = svm.predict_proba(cv_bow)
print("The validation log loss is : ",log_loss(cv_labels, predict_cv, labels=svm.classes_, eps=1e-15))
predict_y = svm.predict_proba(test_bow)
print("The test log loss is : ",log_loss(test_labels, predict_y, labels=svm.classes_, eps=1e-15))

predict = svm.predict(test_bow)
acc_rbf_svm_bow = accuracy_score(test_labels,predict)
print('Accuracy :',acc_rbf_svm_bow)
miss_rbf_svm_bow = sum(predict!=test_labels)
print('Number of missclassified points : ',sum(predict!=test_labels))
print('Classification  Report :\n',classification_report(test_labels,predict))
loss_rbf_svm_bow = log_loss(test_labels, predict_y, labels=svm.classes_, eps=1e-15)
predicted_y = np.argmax(predict_y, axis=1)
plot_confusion_matrix(test_labels,predicted_y+1)


# In[ ]:


# params = {
#     'C':[0.001,0.01,0.1,1,10,100,1000],
#     'gamma':['scale','auto'],    
#          }

# svm = SVC(kernel='rbf',probability=True,class_weight='balanced')
# svm.fit(train_tfidf,train_labels)
# clf = GridSearchCV(svm,param_grid=params,scoring='f1_micro',cv=5,verbose=2,n_jobs=-1)
# clf.fit(train_bow,train_labels)

# print('Best Parameters : \n',clf.best_params_)
# print('Best score : \n',clf.best_score_)
# print('Best Estimator : \n',clf.best_estimator_)

# best_params = clf.best_params_
svm = SVC(kernel='rbf',probability=True,C=0.1,gamma='auto',class_weight='balanced')
svm.fit(train_tfidf,train_labels)

predict_train = svm.predict_proba(train_tfidf)
print("The train log loss is : ",log_loss(train_labels, predict_train, labels=svm.classes_, eps=1e-15))
predict_cv = svm.predict_proba(cv_tfidf)
print("The validation log loss is : ",log_loss(cv_labels, predict_cv, labels=svm.classes_, eps=1e-15))
predict_y = svm.predict_proba(test_tfidf)
print("The test log loss is : ",log_loss(test_labels, predict_y, labels=svm.classes_, eps=1e-15))

predict = svm.predict(test_tfidf)
acc_rbf_svm_tfidf = accuracy_score(test_labels,predict)
print('Accuracy :',acc_rbf_svm_tfidf)
miss_rbf_svm_tfidf = sum(predict!=test_labels)
print('Number of missclassified points : ',sum(predict!=test_labels))
print('Classification  Report :\n',classification_report(test_labels,predict))
loss_rbf_svm_tfidf = log_loss(test_labels, predict_y, labels=svm.classes_, eps=1e-15)
predicted_y = np.argmax(predict_y, axis=1)
plot_confusion_matrix(test_labels,predicted_y+1)


# Linear SVM :

# In[ ]:


# params = {
#     'C':[0.01,0.1,1]
# #     'gamma':['scale','auto'],    
#          }

# svm = SVC(kernel='linear',probability=True,class_weight='balanced')
# svm.fit(train_bow,train_labels)
# clf = GridSearchCV(svm,param_grid=params,scoring='f1_micro',cv=5,verbose=2,n_jobs=-1)
# clf.fit(train_bow,train_labels)

# print('Best Parameters : \n',clf.best_params_)
# print('Best score : \n',clf.best_score_)
# print('Best Estimator : \n',clf.best_estimator_)

# best_params = clf.best_params_
svm = SVC(kernel='linear',probability=True,C=0.1,gamma='auto',class_weight='balanced')
svm.fit(train_bow,train_labels)

predict_train = svm.predict_proba(train_bow)
print("The train log loss is : ",log_loss(train_labels, predict_train, labels=svm.classes_, eps=1e-15))
predict_cv = svm.predict_proba(cv_bow)
print("The validation log loss is : ",log_loss(cv_labels, predict_cv, labels=svm.classes_, eps=1e-15))
predict_y = svm.predict_proba(test_bow)
print("The test log loss is : ",log_loss(test_labels, predict_y, labels=svm.classes_, eps=1e-15))

predict = svm.predict(test_bow)
acc_linear_svm_bow = accuracy_score(test_labels,predict)
print('Accuracy :',acc_linear_svm_bow)
miss_linear_svm_bow = sum(predict!=test_labels)
print('Number of missclassified points : ',sum(predict!=test_labels))
print('Classification  Report :\n',classification_report(test_labels,predict))
loss_linear_svm_bow = log_loss(test_labels, predict_y, labels=svm.classes_, eps=1e-15)
predicted_y = np.argmax(predict_y, axis=1)
plot_confusion_matrix(test_labels,predicted_y+1)


# In[ ]:


# params = {
#     'C':[0.001,0.01,0.1,1,10,100,1000],
#     'gamma':['scale','auto'],    
#          }

# svm = SVC(kernel='linear',probability=True,class_weight='balanced')
# svm.fit(train_tfidf,train_labels)
# clf = GridSearchCV(svm,param_grid=params,scoring='f1_micro',cv=5,verbose=2,n_jobs=-1)
# clf.fit(train_tfidf,train_labels)

# print('Best Parameters : \n',clf.best_params_)
# print('Best score : \n',clf.best_score_)
# print('Best Estimator : \n',clf.best_estimator_)

# best_params = clf.best_params_
svm = SVC(kernel='linear',probability=True,C=0.1,gamma='auto',class_weight='balanced')
svm.fit(train_tfidf,train_labels)

predict_train = svm.predict_proba(train_tfidf)
print("The train log loss is : ",log_loss(train_labels, predict_train, labels=svm.classes_, eps=1e-15))
predict_cv = svm.predict_proba(cv_tfidf)
print("The validation log loss is : ",log_loss(cv_labels, predict_cv, labels=svm.classes_, eps=1e-15))
predict_y = svm.predict_proba(test_tfidf)
print("The test log loss is : ",log_loss(test_labels, predict_y, labels=svm.classes_, eps=1e-15))

predict = svm.predict(test_tfidf)
acc_linear_svm_tfidf = accuracy_score(test_labels,predict)
print('Accuracy :',acc_linear_svm_tfidf)
miss_linear_svm_tfidf = sum(predict!=test_labels)
print('Number of missclassified points : ',sum(predict!=test_labels))
print('Classification  Report :\n',classification_report(test_labels,predict))
loss_linear_svm_tfidf = log_loss(test_labels, predict_y, labels=svm.classes_, eps=1e-15)
predicted_y = np.argmax(predict_y, axis=1)
plot_confusion_matrix(test_labels,predicted_y+1)


# GBDT :

# In[ ]:


# params = {
#     'learning_rate':[0.01,0.1,1,10],
#          'n_estimators' : [2,5,10,50,100,200,500]
#          'max_depth' : [2,5,10,50,70,100],
#          'min_samples_leaf':[1,2,5,8,10],
#          'subsample':[0.1,0.5,0.8,1]
#          }

# gbdt = GradientBoostingClassifier(learning_rate=0.1,n_estimators=100,max_depth=100)
# gbdt.fit(train_bow,train_labels)
# clf = GridSearchCV(gbdt,param_grid=params,scoring='f1_micro',cv=5,verbose=2,n_jobs=-1)
# clf.fit(train_bow,train_labels)

# print('Best Parameters : \n',clf.best_params_)
# print('Best score : \n',clf.best_score_)
# print('Best Estimator : \n',clf.best_estimator_)

# best_params = clf.best_params_
gbdt = GradientBoostingClassifier(learning_rate = 0.1,n_estimators=100,max_depth=100,min_samples_leaf=5,subsample=0.8)
gbdt.fit(train_bow,train_labels)

predict_train = gbdt.predict_proba(train_bow)
print("The train log loss is : ",log_loss(train_labels, predict_train, labels=gbdt.classes_, eps=1e-15))
predict_cv = gbdt.predict_proba(cv_bow)
print("The validation log loss is : ",log_loss(cv_labels, predict_cv, labels=gbdt.classes_, eps=1e-15))
predict_y = gbdt.predict_proba(test_bow)
print("The test log loss is : ",log_loss(test_labels, predict_y, labels=gbdt.classes_, eps=1e-15))

predict = gbdt.predict(test_bow)
acc_gbdt_bow = accuracy_score(test_labels,predict)
print('Accuracy :',acc_gbdt_bow)
miss_gbdt_bow = sum(predict!=test_labels)
print('Number of missclassified points : ',sum(predict!=test_labels))
print('Classification  Report :\n',classification_report(test_labels,predict))
loss_gbdt_bow = log_loss(test_labels, predict_y, labels=gbdt.classes_, eps=1e-15)
predicted_y = np.argmax(predict_y, axis=1)
plot_confusion_matrix(test_labels,predicted_y+1)


# GBDT with BOW is overfitting

# In[ ]:


# params = {'learning_rate':[0.01,0.1,1,10],
#          'n_estimators' : [2,5,10,50,100],
#          'max_depth' : [2,5,10,50,100],
#          'min_samples_leaf':[1,2,5,8,10],
#          'subsample':[0.1,0.5,0.8,1]
#          }

# gbdt = GradientBoostingClassifier()
# gbdt.fit(train_tfidf,train_labels)
# clf = GridSearchCV(gbdt,param_grid=params,scoring='f1_micro',cv=10,verbose=2,n_jobs=-1)
# clf.fit(train_tfidf,train_labels)

# print('Best Parameters : \n',clf.best_params)
# print('Best score : \n',clf.best_score_)
# print('Best Estimator : \n',clf.best_estimator_)
# best_params = clf.best_params_

gbdt = GradientBoostingClassifier(learning_rate = 0.1,n_estimators=100,max_depth=100,min_samples_leaf=5,subsample=0.8)
gbdt.fit(train_tfidf,train_labels)

predict_train = gbdt.predict_proba(train_tfidf)
print("The train log loss is : ",log_loss(train_labels, predict_train, labels=gbdt.classes_, eps=1e-15))
predict_cv = gbdt.predict_proba(cv_tfidf)
print("The validation log loss is : ",log_loss(cv_labels, predict_cv, labels=gbdt.classes_, eps=1e-15))
predict_y = gbdt.predict_proba(test_tfidf)
print("The test log loss is : ",log_loss(test_labels, predict_y, labels=gbdt.classes_, eps=1e-15))

predict = gbdt.predict(test_tfidf)
acc_gbdt_tfidf = accuracy_score(test_labels,predict)
print('Accuracy :',acc_gbdt_tfidf)
miss_gbdt_tfidf = sum(predict!=test_labels)
print('Number of missclassified points : ',sum(predict!=test_labels))
print('Classification  Report :\n',classification_report(test_labels,predict))
loss_gbdt_tfidf = log_loss(test_labels, predict_y, labels=gbdt.classes_, eps=1e-15)
predicted_y = np.argmax(predict_y, axis=1)
plot_confusion_matrix(test_labels,predicted_y+1)


# GBDT with TF-IDF is overfitting

# Random Forest :

# In[ ]:


# params = {
#     'criterion':['gini','entropy'],
#          'n_estimators' : [2,5,10,50,100,200,500],
#          'max_depth' : [2,5,10,50,70,100],
#          'min_samples_leaf':[1,2,5,8,10]
#          }

# rf = RandomForestClassifier(class_weight='balanced')
# rf.fit(train_bow,train_labels)
# clf = GridSearchCV(rf,param_grid=params,scoring='f1_micro',cv=5,verbose=2,n_jobs=-1)
# clf.fit(train_bow,train_labels)

# print('Best Parameters : \n',clf.best_params_)
# print('Best score : \n',clf.best_score_)
# print('Best Estimator : \n',clf.best_estimator_)

# best_params = clf.best_params_
rf = RandomForestClassifier(criterion='entropy',n_estimators=500,max_depth=70,min_samples_leaf=2,class_weight='balanced')
rf.fit(train_bow,train_labels)

predict_train = rf.predict_proba(train_bow)
print("The train log loss is : ",log_loss(train_labels, predict_train, labels=rf.classes_, eps=1e-15))
predict_cv = rf.predict_proba(cv_bow)
print("The validation log loss is : ",log_loss(cv_labels, predict_cv, labels=rf.classes_, eps=1e-15))
predict_y = rf.predict_proba(test_bow)
print("The test log loss is : ",log_loss(test_labels, predict_y, labels=rf.classes_, eps=1e-15))

predict = rf.predict(test_bow)
acc_rf_bow = accuracy_score(test_labels,predict)
print('Accuracy :',acc_rf_bow)
miss_rf_bow = sum(predict!=test_labels)
print('Number of missclassified points : ',sum(predict!=test_labels))
print('Classification  Report :\n',classification_report(test_labels,predict))
loss_rf_bow = log_loss(test_labels, predict_y, labels=rf.classes_, eps=1e-15)
predicted_y = np.argmax(predict_y, axis=1)
plot_confusion_matrix(test_labels,predicted_y+1)


# In[ ]:


# params = {
#     'criterion':['gini','entropy'],
#          'n_estimators' : [2,5,10,50,100,200,500],
#          'max_depth' : [2,5,10,50,70,100],
#          'min_samples_leaf':[1,2,5,8,10]
#          }

# rf = RandomForestClassifier(class_weight='balanced')
# rf.fit(train_tfidf,train_labels)
# clf = GridSearchCV(rf,param_grid=params,scoring='f1_micro',cv=5,verbose=2,n_jobs=-1)
# clf.fit(train_tfidf,train_labels)

# print('Best Parameters : \n',clf.best_params_)
# print('Best score : \n',clf.best_score_)
# print('Best Estimator : \n',clf.best_estimator_)

# best_params = clf.best_params_
rf = RandomForestClassifier(criterion='entropy',n_estimators=500,max_depth=70,min_samples_leaf=2,class_weight='balanced')
rf.fit(train_tfidf,train_labels)

predict_train = rf.predict_proba(train_tfidf)
print("The train log loss is : ",log_loss(train_labels, predict_train, labels=rf.classes_, eps=1e-15))
predict_cv = rf.predict_proba(cv_tfidf)
print("The validation log loss is : ",log_loss(cv_labels, predict_cv, labels=rf.classes_, eps=1e-15))
predict_y = rf.predict_proba(test_tfidf)
print("The test log loss is : ",log_loss(test_labels, predict_y, labels=rf.classes_, eps=1e-15))

predict = rf.predict(test_tfidf)
acc_rf_tfidf = accuracy_score(test_labels,predict)
print('Accuracy :',acc_rf_tfidf)
miss_rf_tfidf = sum(predict!=test_labels)
print('Number of missclassified points : ',sum(predict!=test_labels))
print('Classification  Report :\n',classification_report(test_labels,predict))
loss_rf_tfidf = log_loss(test_labels, predict_y, labels=rf.classes_, eps=1e-15)
predicted_y = np.argmax(predict_y, axis=1)
plot_confusion_matrix(test_labels,predicted_y+1)


# KNN : 

# In[ ]:


# params = {
#     'n_neighbors':[1,2,3,4,5,6,7,8,9,10],
#        'algorithm':['kd-tree','auto','brute'],
#         'leaf_size':[2,5,10,20,30,50,70,100],
#          }

# knn = KNeighborsClassifier()
# knn.fit(train_bow,train_labels)
# clf = GridSearchCV(knn,param_grid=params,scoring='f1_micro',cv=5,verbose=2,n_jobs=-1)
# clf.fit(train_bow,train_labels)

# print('Best Parameters : \n',clf.best_params_)
# print('Best score : \n',clf.best_score_)
# print('Best Estimator : \n',clf.best_estimator_)

# best_params = clf.best_params_
knn = KNeighborsClassifier(algorithm='auto',leaf_size=2,n_neighbors=9)
knn.fit(train_bow,train_labels)

predict_train = knn.predict_proba(train_bow)
print("The train log loss is : ",log_loss(train_labels, predict_train, labels=knn.classes_, eps=1e-15))
predict_cv = knn.predict_proba(cv_bow)
print("The validation log loss is : ",log_loss(cv_labels, predict_cv, labels=knn.classes_, eps=1e-15))
predict_y = knn.predict_proba(test_bow)
print("The test log loss is : ",log_loss(test_labels, predict_y, labels=knn.classes_, eps=1e-15))

predict = knn.predict(test_bow)
acc_knn_bow = accuracy_score(test_labels,predict)
print('Accuracy :',acc_knn_bow)
miss_knn_bow = sum(predict!=test_labels)
print('Number of missclassified points : ',sum(predict!=test_labels))
print('Classification  Report :\n',classification_report(test_labels,predict))
loss_knn_bow = log_loss(test_labels, predict_y, labels=knn.classes_, eps=1e-15)
predicted_y = np.argmax(predict_y, axis=1)
plot_confusion_matrix(test_labels,predicted_y+1)


# In[ ]:


# params = {
#     'n_neighbors':[1,2,3,4,5,6,7,8,9,10],
#        'algorithm':['kd-tree','auto','brute'],
#         'leaf_size':[2,5,10,20,30,50,70,100],
#          }

# knn = KNeighborsClassifier()
# knn.fit(train_tfidf,train_labels)
# clf = GridSearchCV(knn,param_grid=params,scoring='f1_micro',cv=5,verbose=2,n_jobs=-1)
# clf.fit(train_tfidf,train_labels)

# print('Best Parameters : \n',clf.best_params_)
# print('Best score : \n',clf.best_score_)
# print('Best Estimator : \n',clf.best_estimator_)

# best_params = clf.best_params_
knn = KNeighborsClassifier(algorithm='auto',leaf_size=2,n_neighbors=9)
knn.fit(train_tfidf,train_labels)

predict_train = knn.predict_proba(train_tfidf)
print("The train log loss is : ",log_loss(train_labels, predict_train, labels=knn.classes_, eps=1e-15))
predict_cv = knn.predict_proba(cv_tfidf)
print("The validation log loss is : ",log_loss(cv_labels, predict_cv, labels=knn.classes_, eps=1e-15))
predict_y = knn.predict_proba(test_tfidf)
print("The test log loss is : ",log_loss(test_labels, predict_y, labels=knn.classes_, eps=1e-15))

predict = knn.predict(test_tfidf)
acc_knn_tfidf = accuracy_score(test_labels,predict)
print('Accuracy :',acc_knn_tfidf)
miss_knn_tfidf = sum(predict!=test_labels)
print('Number of missclassified points : ',sum(predict!=test_labels))
print('Classification  Report :\n',classification_report(test_labels,predict))
loss_knn_tfidf = log_loss(test_labels, predict_y, labels=knn.classes_, eps=1e-15)
predicted_y = np.argmax(predict_y, axis=1)
plot_confusion_matrix(test_labels,predicted_y+1)


# XGBoost : 
# 

# In[ ]:


# params = {
# #     'booster':['gbtree'],
# #     'eta':[0.01,0.1,0.3,0.5,0.7,0.9],
# #     'gamma':[0,0.3,0.5,0.8],
# #     'max_depth':[2,5,10,20,40,70],
# #     'min_child_weight':[1,3,5,7,9,11,15],
#     'subsample':[0.1,0.3,0.5,0.6,0.8,1],
# #     'colsample_bytree':[0.6,0.8,1],
# #     'reg_lambda':[0.01,0.1,1,10],
# #     'reg_alpha':[0.01,0.1,1,10],
# #     'max_leaves':[1,2,3,5],
#          }

# xg = xgb.XGBClassifier(booster='gbtree',eta=0.01,gamma=0.8,max_depth=40,min_child_weight=3)
# xg.fit(train_bow,train_labels)
# clf = GridSearchCV(xg,param_grid=params,scoring='f1_micro',cv=5,verbose=2,n_jobs=-1)
# clf.fit(train_bow,train_labels)

# print('Best Parameters : \n',clf.best_params_)
# print('Best score : \n',clf.best_score_)
# print('Best Estimator : \n',clf.best_estimator_)

# best_params = clf.best_params_
xg = xgb.XGBClassifier(booster='gbtree',
    eta=0.3,
    gamma=0.3,
    max_depth=5,
    min_child_weight=5,
    subsample=0.8,   
    colsample_bytree=0.8,
    reg_lambda=0.3,
    reg_alpha=0.3,
    max_leaves=5)
xg.fit(train_bow,train_labels)

predict_train = xg.predict_proba(train_bow)
print("The train log loss is : ",log_loss(train_labels, predict_train, labels=xg.classes_, eps=1e-15))
predict_cv = xg.predict_proba(cv_bow)
print("The validation log loss is : ",log_loss(cv_labels, predict_cv, labels=xg.classes_, eps=1e-15))
predict_y = xg.predict_proba(test_bow)
print("The test log loss is : ",log_loss(test_labels, predict_y, labels=xg.classes_, eps=1e-15))

predict = xg.predict(test_bow)
acc_xgb_bow = accuracy_score(test_labels,predict)
print('Accuracy :',acc_xgb_bow)
miss_xgb_bow = sum(predict!=test_labels)
print('Number of missclassified points : ',sum(predict!=test_labels))
print('Classification  Report :\n',classification_report(test_labels,predict))
loss_xgb_bow = log_loss(test_labels, predict_y, labels=xg.classes_, eps=1e-15)
predicted_y = np.argmax(predict_y, axis=1)
plot_confusion_matrix(test_labels,predicted_y+1)


# In[ ]:


# params = {
# #     'booster':['gbtree'],
# #     'eta':[0.01,0.1,0.3,0.5,0.7,0.9],
# #     'gamma':[0,0.3,0.5,0.8],
# #     'max_depth':[2,5,10,20,40,70],
# #     'min_child_weight':[1,3,5,7,9,11,15],
#     'subsample':[0.1,0.3,0.5,0.6,0.8,1],
# #     'colsample_bytree':[0.6,0.8,1],
# #     'reg_lambda':[0.01,0.1,1,10],
# #     'reg_alpha':[0.01,0.1,1,10],
# #     'max_leaves':[1,2,3,5],
#          }

# xg = xgb.XGBClassifier(booster='gbtree',eta=0.01,gamma=0.8,max_depth=40,min_child_weight=3)
# xg.fit(train_tfidf,train_labels)
# clf = GridSearchCV(xg,param_grid=params,scoring='f1_micro',cv=5,verbose=2,n_jobs=-1)
# clf.fit(train_tfidf,train_labels)

# print('Best Parameters : \n',clf.best_params_)
# print('Best score : \n',clf.best_score_)
# print('Best Estimator : \n',clf.best_estimator_)

# best_params = clf.best_params_
xg = xgb.XGBClassifier(booster='gbtree',
    eta=0.3,
    gamma=0.3,
    max_depth=5,
    min_child_weight=5,
    subsample=0.8,   
    colsample_bytree=0.8,
    reg_lambda=0.3,
    reg_alpha=0.3,
    max_leaves=5)
xg.fit(train_tfidf,train_labels)

predict_train = xg.predict_proba(train_tfidf)
print("The train log loss is : ",log_loss(train_labels, predict_train, labels=xg.classes_, eps=1e-15))
predict_cv = xg.predict_proba(cv_tfidf)
print("The validation log loss is : ",log_loss(cv_labels, predict_cv, labels=xg.classes_, eps=1e-15))
predict_y = xg.predict_proba(test_tfidf)
print("The test log loss is : ",log_loss(test_labels, predict_y, labels=xg.classes_, eps=1e-15))

predict = xg.predict(test_tfidf)
acc_xgb_tfidf = accuracy_score(test_labels,predict)
print('Accuracy :',acc_xgb_tfidf)
miss_xgb_tfidf = sum(predict!=test_labels)
print('Number of missclassified points : ',sum(predict!=test_labels))
print('Classification  Report :\n',classification_report(test_labels,predict))
loss_xgb_tfidf = log_loss(test_labels, predict_y, labels=xg.classes_, eps=1e-15)
predicted_y = np.argmax(predict_y, axis=1)
plot_confusion_matrix(test_labels,predicted_y+1)


# Light GBM :

# In[ ]:


X_train_bow = train_bow.astype('float32')
X_test_bow = test_bow.astype('float32')
X_cv_bow = cv_bow.astype('float32')   
Y_train = train_labels.astype('float32')
Y_cv = cv_labels.astype('float32')
Y_test = test_labels.astype('float32')

# params = {
#     'booster':['gbdt','rf'],
#     'learning_rate':[0.01,0.1,0.3,0.5,0.7,0.9],
#     'n_estimators':[2,4,10,30,50,100,200],
#     'subsample':[0.2,0.3,0.5,0.8,1],
#     'max_depth':[2,5,10,20,40,70],
#     'min_child_weight':[1,3,5,7,9,11,15],
#     'colsample_bytree':[0.6,0.8,1],
#     'reg_lambda':[0.01,0.1,1,10],
#     'reg-alpha':[0.01,0.1,1,10],
#     'num_leaves':[1,2,3,5,10,20,50,100],
#          }

# lg = lgb.LGBMClassifier(class_weight='balanced')
# lg.fit(X_train_bow,Y_train)
# clf = GridSearchCV(lg,param_grid=params,scoring='f1_micro',cv=5,verbose=2,n_jobs=-1)
# clf.fit(X_train_bow,Y_train)

# print('Best Parameters : \n',clf.best_params_)
# print('Best score : \n',clf.best_score_)
# print('Best Estimator : \n',clf.best_estimator_)

# best_params = clf.best_params_
lg = lgb.LGBMClassifier(class_weight='balanced',
                        booster='gbdt',
                        learning_rate=0.7,
                        n_estimators=40,
                        subsample=0.2,
                        num_leaves=100,
                        max_depth=40,
                        min_child_weight=1 , 
                        colsample_bytree=0.8,
                        reg_lambda=0.2,
                        reg_alpha=0.2,)
lg.fit(X_train_bow,Y_train)

predict_train = lg.predict_proba(X_train_bow)
print("The train log loss is : ",log_loss(Y_train, predict_train, labels=lg.classes_, eps=1e-15))
predict_cv = lg.predict_proba(X_cv_bow)
print("The validation log loss is : ",log_loss(Y_cv, predict_cv, labels=lg.classes_, eps=1e-15))
predict_y = lg.predict_proba(X_test_bow)
print("The test log loss is : ",log_loss(Y_test ,predict_y, labels=lg.classes_, eps=1e-15))

predict = lg.predict(X_test_bow)
acc_lgb_bow = accuracy_score(test_labels,predict)
print('Accuracy :',acc_lgb_bow)
miss_lgb_bow = sum(predict!=test_labels)
print('Number of missclassified points : ',sum(predict!=test_labels))
print('Classification  Report :\n',classification_report(test_labels,predict))
loss_lgb_bow = log_loss(Y_test, predict_y, labels=lg.classes_, eps=1e-15)
predicted_y = np.argmax(predict_y, axis=1)
plot_confusion_matrix(test_labels,predicted_y+1)


# In[ ]:


X_train_tfidf = train_tfidf.astype('float32')
X_test_tfidf = test_tfidf.astype('float32')
X_cv_tfidf = cv_tfidf.astype('float32')

# params = {
#     'booster':['gbdt','rf'],
#     'learning_rate':[0.01,0.1,0.3,0.5,0.7,0.9],
#     'n_estimators':[2,4,10,30,50,100,200],
#     'subsample':[0.2,0.3,0.5,0.8,1],
#     'max_depth':[2,5,10,20,40,70],
#     'min_child_weight':[1,3,5,7,9,11,15],
#     'colsample_bytree':[0.6,0.8,1],
#     'reg_lambda':[0.01,0.1,1,10],
#     'reg-alpha':[0.01,0.1,1,10],
#     'num_leaves':[1,2,3,5,10,20,50,100],
#          }

# lg = lgb.LGBMClassifier(class_weight='balanced')
# lg.fit(X_train_tfidf,Y_train)
# clf = GridSearchCV(lg,param_grid=params,scoring='f1_micro',cv=5,verbose=2,n_jobs=-1)
# clf.fit(X_train_tfidf,Y_train)

# print('Best Parameters : \n',clf.best_params_)
# print('Best score : \n',clf.best_score_)
# print('Best Estimator : \n',clf.best_estimator_)

# best_params = clf.best_params_
lg = lgb.LGBMClassifier(class_weight='balanced',
                        booster='gbdt',
                        learning_rate=0.7,
                        n_estimators=40,
                        subsample=0.2,
                        num_leaves=100,
                        max_depth=40,
                        min_child_weight=1 , 
                        colsample_bytree=0.8,
                        reg_lambda=0.2,
                        reg_alpha=0.2,)
lg.fit(X_train_tfidf,Y_train)

predict_train = lg.predict_proba(X_train_tfidf)
print("The train log loss is : ",log_loss(Y_train, predict_train, labels=lg.classes_, eps=1e-15))
predict_cv = lg.predict_proba(X_cv_tfidf)
print("The validation log loss is : ",log_loss(Y_cv, predict_cv, labels=lg.classes_, eps=1e-15))
predict_y = lg.predict_proba(X_test_tfidf)
print("The test log loss is : ",log_loss(Y_test ,predict_y, labels=lg.classes_, eps=1e-15))

predict = lg.predict(X_test_tfidf)
acc_lgb_tfidf = accuracy_score(test_labels,predict)
print('Accuracy :',acc_lgb_tfidf)
miss_lgb_tfidf = sum(predict!=test_labels)
print('Number of missclassified points : ',sum(predict!=test_labels))
print('Classification  Report :\n',classification_report(test_labels,predict))
loss_lgb_tfidf = log_loss(Y_test, predict_y, labels=lg.classes_, eps=1e-15)
predicted_y = np.argmax(predict_y, axis=1)
plot_confusion_matrix(test_labels,predicted_y+1)


# Word2vec :

# In[20]:


data= Data


# In[21]:


start_time = time.time()

text = data['Text'].apply(lambda x : remove_shortforms(x))
text = text.apply(lambda x : remove_html(x))
text = text.apply(lambda x : remove_special_char(x))
text = text.apply(lambda x : remove_wordswithnum(x))
text = text.apply(lambda x : lowercase(x))
text = text.apply(lambda x : remove_stop_words(x))
text = text.apply(lambda x : listToString(stemming(x.split())))
text = text.apply(lambda x : remove_punctuations(x))
text = text.apply(lambda x : remove_links(x))
text = text.apply(lambda x : lemmatize_words(x))

print('Preprocessing of text done in : ',np.round(time.time() - start_time,3), 'seconds')

data['preprocessed_text'] = text
text[:4]


# In[22]:


start_time = time.time()

text = data['Summary'].apply(lambda x : remove_shortforms(x))
text = text.apply(lambda x : remove_html(x))
text = text.apply(lambda x : remove_special_char(x))
text = text.apply(lambda x : remove_wordswithnum(x))
text = text.apply(lambda x : lowercase(x))
text = text.apply(lambda x : remove_stop_words(x))
text = text.apply(lambda x : listToString(stemming(x.split())))
text = text.apply(lambda x : remove_punctuations(x))
text = text.apply(lambda x : remove_links(x))
text = text.apply(lambda x : lemmatize_words(x))

print('Preprocessing of summary done in : ',np.round(time.time() - start_time,3), 'seconds')
data['preprocessed_summary'] = text
text[:4]


# In[23]:


data.head()


# In[24]:


data = data.reset_index(drop=True)


# In[25]:


actual_data = [None] * len(data)
for i in range(len(data)):
    actual_data[i] = data['preprocessed_text'][i] +' '+ data['preprocessed_summary'][i] +' '+ data['preprocessed_summary'][i] +' '+ data['preprocessed_summary'][i]
actual_data[:2]


# Word 2 Vec :

# In[27]:


actual_data[5]


# In[30]:


i=0
list_of_sentence=[] 
for sentence in tqdm(actual_data):
    list_of_sentence.append(sentence.split())

w2v_model = Word2Vec(list_of_sentence,min_count=1,size=300,workers=4)
print(w2v_model.wv.most_similar('great'))
print('='*125)
print(w2v_model.wv.most_similar('worst'))   
print('='*125)
print(w2v_model.wv.most_similar('boy'))   
print('='*125)
w2v_words = list(w2v_model.wv.vocab)
print("number of words ",len(w2v_words))
print("sample words ", w2v_words[0:50])


# In[ ]:


# sent_vectors = []
# for sent in tqdm(list_of_sentance):  
#     sent_vec = np.zeros(100) 
#     cnt_words = 0
#     for word in sent: 
#         if word in w2v_words:
#             vec = w2v_model.wv[word]
#             sent_vec += vec
#             cnt_words += 1
#     if cnt_words != 0:
#         sent_vec /= cnt_words
#     sent_vectors.append(sent_vec)
# print(len(sent_vectors))
# print(len(sent_vectors[0]))


# In[31]:


model = TfidfVectorizer()
tf_idf_matrix = model.fit_transform(actual_data)
dictionary = dict(zip(model.get_feature_names(), list(model.idf_)))


# In[39]:


tfidf_feat = model.get_feature_names()
tfidf_sent_vectors = []
row=0;
for sent in tqdm(list_of_sentence):  
    sent_vec = np.zeros(300) 
    weight_sum =0; 
    for word in sent: 
        if word in w2v_words and word in tfidf_feat:
            vec = w2v_model.wv[word]
            tf_idf = dictionary[word]*(sent.count(word)/len(sent))
            sent_vec += (vec * tf_idf)
            weight_sum += tf_idf
    if weight_sum != 0:
        sent_vec /= weight_sum
    tfidf_sent_vectors.append(sent_vec)
    row += 1


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:


model_names = ['Random Model','Logistic Regression with BOW(without class balancing)',
              'Logistic Regression with TFIDF(without class balancing)','Logistic Regression with BOW(with class balancing)',
              'Logistic Regression with TFIDF(with class balancing)','RBF SVM with BOW','RBF SVM with TFIDF',
              'Linear SVM with BOW ','Linear SVM with TFIDF','Gradient Boosting Decision Trees with BOW',
              'Gradient Boosting Decision Trees with TFIDF','K-Nearest Neighbors with BOW','K-Nearest Neighbors with TFIDF',
              'Random Forest with BOW ','Random Forest with TFIDF','XG Boost with BOW','XG Boost with TFIDF',
              'Light GBM with BOW','Light GBM with TFIDF']
loss_values = [np.round(loss_random),np.round(loss_lr_bow,4),np.round(loss_lr_tfidf,4),np.round(loss_lr_bow_balanced,4),
              np.round(loss_lr_tfidf_balanced,4),np.round(loss_rbf_svm_bow,4),np.round(loss_rbf_svm_tfidf,4),
              np.round(loss_linear_svm_bow,4),np.round(loss_linear_svm_tfidf,4),np.round(loss_gbdt_bow,4),
              np.round(loss_gbdt_tfidf,4),np.round(loss_knn_bow,4),np.round(loss_knn_tfidf,4),np.round(loss_rf_bow,4),
              np.round(loss_rf_tfidf,4),np.round(loss_xgb_bow,4),np.round(loss_xgb_tfidf,4),np.round(loss_lgb_bow,4),
              np.round(loss_lgb_tfidf,4)]
missclass_values = [miss_random,miss_lr_bow,miss_lr_tfidf,miss_lr_bow_balanced,miss_lr_tfidf_balanced,miss_rbf_svm_bow,
                    miss_rbf_svm_tfidf,miss_linear_svm_bow,miss_linear_svm_tfidf,miss_gbdt_bow,miss_gbdt_tfidf,miss_knn_bow,miss_knn_tfidf,
                    miss_rf_bow,miss_rf_tfidf,miss_xgb_bow,miss_xgb_tfidf,miss_lgb_bow,miss_lgb_tfidf]
accuracy_values = [acc_random,acc_lr_bow,acc_lr_tfidf,acc_lr_bow_balanced,acc_lr_tfidf_balanced,acc_rbf_svm_bow,
                    acc_rbf_svm_tfidf,acc_linear_svm_bow,acc_linear_svm_tfidf,acc_gbdt_bow,acc_gbdt_tfidf,acc_knn_bow,acc_knn_tfidf,
                    acc_rf_bow,acc_rf_tfidf,acc_xgb_bow,acc_xgb_tfidf,acc_lgb_bow,acc_lgb_tfidf]
summary = pd.DataFrame(model_names,columns=['Model Name'])
summary['Log Loss with Test Data'] = loss_values
summary['Number of missclassified points'] = missclass_values
summary['Accuracy of the model'] = accuracy_values
summary['Status of the model'] = ['Not Overfitting','Not Overfitting','Not Overfitting','Not Overfitting','Not Overfitting',
                                 'Not Overfitting','Not Overfitting','Not Overfitting','Not Overfitting','Overfitting',
                                 'Overfitting','Overfitting','Overfitting','Not Overfitting','Not Overfitting',
                                 'Not Overfitting','Not Overfitting','Overfitting','Overfitting']
abc = summary.style.background_gradient(cmap='Oranges')
abc

