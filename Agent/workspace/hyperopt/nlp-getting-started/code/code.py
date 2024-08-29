#!/usr/bin/env python
# coding: utf-8

# # Natural Langugae Processing with Disaster Tweets: XGBoost and SVC Implementation
# 
# <div style="text-align: center;">
#     <img src="https://miro.medium.com/v2/resize:fit:600/1*iuoT4P9L802xZPg0x1oGgA.jpeg" alt="House Prices" width="600" height="300">
# </div>

# ## Import Libraries

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re
import string
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
nltk.download('stopwords')


# ## Loading Data
FILE_PATH = "./workspace/hyperopt/ps311/data/"

train = pd.read_csv(FILE_PATH+'nlp-getting-started/train.csv')
test = pd.read_csv(FILE_PATH+'nlp-getting-started/test.csv')

train.head()


# ## EDA

# print(train.isnull().sum())


# print(train.describe())


# # Plot the distribution of the target variable
# plt.figure(figsize=(8, 6))
# sns.countplot(x='target', data=train, palette='viridis')
# plt.title('Distribution of Target Variable')
# plt.xlabel('Target')
# plt.ylabel('Count')
# plt.show()


# Visualize the length of tweets
train['text_length'] = train['text'].apply(len)

# plt.figure(figsize=(10, 6))
# sns.histplot(train['text_length'], kde=True, color='purple', bins=30)
# plt.title('Distribution of Tweet Lengths')
# plt.xlabel('Tweet Length')
# plt.ylabel('Frequency')
# plt.show()





# Visualize word count distribution
train['word_count'] = train['text'].apply(lambda x: len(x.split()))

# plt.figure(figsize=(10, 6))
# sns.histplot(train['word_count'], kde=True, color='blue', bins=30)
# plt.title('Distribution of Word Counts in Tweets')
# plt.xlabel('Word Count')
# plt.ylabel('Frequency')
# plt.show()


# Visualize top 20 most frequent words in the tweets (excluding stop words)
from sklearn.feature_extraction.text import CountVectorizer
from collections import Counter

vectorizer = CountVectorizer(stop_words='english')
word_count_vector = vectorizer.fit_transform(train['text'])
word_counts = word_count_vector.toarray().sum(axis=0)
vocab = vectorizer.get_feature_names_out() 


common_words_df = pd.DataFrame(sorted(list(zip(vocab, word_counts)), key=lambda x: x[1], reverse=True)[:20], 
                               columns=['Word', 'Count'])

# plt.figure(figsize=(12, 8))
# sns.barplot(x='Count', y='Word', data=common_words_df, palette='coolwarm')
# plt.title('Top 20 Most Frequent Words')
# plt.xlabel('Count')
# plt.ylabel('Word')
# plt.show()


# ## Preprocess the Text Data

def preprocess(text):
    text = text.lower() # Convert to lowercase
    text = re.sub(r'https?://\S+|www\.\S+', '', text) # Remove URLs
    text = re.sub(r'<.*?>', '', text) # Remove HTML tags
    text = re.sub(r'\d+', '', text) # Remove digits
    text = re.sub(f'[{re.escape(string.punctuation)}]', '', text) # Remove punctuation
    text = re.sub(r'\n', '', text) # Remove newline characters
    text = re.sub(r'\s+', ' ', text).strip() # Remove extra spaces
    return text

train['text'] = train['text'].apply(preprocess)
test['text'] = test['text'].apply(preprocess)


# ## Split the Training Data

# X_train, X_val, y_train, y_val = train_test_split(train['text'], train['target'], test_size=0.2, random_state=42)


# ## Vectorize the Text Data Using TF-IDF

tfidf = TfidfVectorizer(max_features=15000, stop_words=stopwords.words('english'))
X_train_tfidf = tfidf.fit_transform(train['text'])
# X_val_tfidf = tfidf.transform(X_val)
X_test_tfidf = tfidf.transform(test['text'])


# ## Build the Model

# svc = SVC()
# svc.fit(X_train_tfidf, y_train)
# y_pred_svc = svc.predict(X_val_tfidf)


xgb = XGBClassifier(use_label_encoder=False, eval_metric='logloss')
# xgb.fit(X_train_tfidf, y_train)
# y_pred_xgb = xgb.predict(X_val_tfidf)


# # Assuming you have these models trained already
# models = {
#     'Support Vector Classifier': svc,
#     'XGBoost': xgb
# }

# # Initialize variables to store the best model and accuracy
# best_model = None
# best_accuracy = 0

# # Evaluate each model and store the one with the best accuracy
# for name, model in models.items():
#     predictions = model.predict(X_val_tfidf)
#     accuracy = accuracy_score(y_val, predictions)
#     print(f'{name} Accuracy: {accuracy}')
    
#     if accuracy > best_accuracy:
#         best_accuracy = accuracy
#         best_model = model

# print(f'\nBest Model: {best_model} with Accuracy: {best_accuracy}')

# # Make predictions on the test data using the best model
# test_predictions = best_model.predict(X_test_tfidf)


# # Create the submission DataFrame
# submission = pd.DataFrame({
#     'id': test['id'],
#     'target': test_predictions
# })

# # Save the submission file
# submission.to_csv('submission.csv', index=False)

