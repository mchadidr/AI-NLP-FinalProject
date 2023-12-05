# -*- coding: utf-8 -*-
"""
Created on Tue Dec  5 15:21:21 2023

@author: migue
"""

import pandas as pd
import os
from sklearn.naive_bayes import MultinomialNB
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from sklearn.metrics import accuracy_score, make_scorer
from sklearn.metrics import confusion_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import cross_val_score

# Set the file path and name
path = "C:/Users/migue\Desktop/Assignments folder/Fall 2023/AI"
filename = "Youtube04-Eminem.csv"
fullpath = os.path.join(path, filename)

# Read the CSV file into a DataFrame
df_eminem = pd.read_csv(fullpath)

# Display the first 3 rows of the DataFrame
print(df_eminem.head(3))

# Display the shape of the DataFrame
print(df_eminem.shape)

# Display the column names
print(df_eminem.columns)

# Check for missing values
print(df_eminem.isnull().sum())

# Drop unnecessary columns
df_eminem.drop(['COMMENT_ID', 'AUTHOR', 'DATE'], axis=1, inplace=True)

# Define the category map
category_map = {0: 'Non Spam', 1: 'Spam'}

# Tokenize and remove stop words using nltk
nltk.download('punkt')
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

df_eminem['TOKENIZED_CONTENT'] = df_eminem['CONTENT'].apply(lambda x: word_tokenize(x.lower()))
df_eminem['TOKENIZED_CONTENT'] = df_eminem['TOKENIZED_CONTENT'].apply(lambda x: [word for word in x if word.isalpha() and word not in stop_words])
df_eminem['PREPROCESSED_CONTENT'] = df_eminem['TOKENIZED_CONTENT'].apply(lambda x: ' '.join(x))

# Shuffle the dataset
shuffled_df = df_eminem.sample(frac=1, random_state=42).reset_index(drop=True)

# Determine the index to split the data (75% for training, 25% for testing)
split_index = int(0.75 * len(shuffled_df))

# Splitting into training and testing sets
train_df, test_df = shuffled_df[:split_index], shuffled_df[split_index:]

# Vectorize using TfidfVectorizer from sklearn
vectorizer = TfidfVectorizer()
X_train = vectorizer.fit_transform(train_df['PREPROCESSED_CONTENT'])
y_train = train_df['CLASS']

# Display the transformed data
print(X_train)

# Display the dimensions of the training data
print("\nDimensions of training data:", X_train.shape)

# Initialize and fit a Naive Bayes classifier
classifier = MultinomialNB().fit(X_train, y_train)

# Transform the testing data
X_test = vectorizer.transform(test_df['PREPROCESSED_CONTENT'])
y_test = test_df['CLASS']

# Predictions on the test set
y_pred_test = classifier.predict(X_test)

# Perform 5-fold cross-validation on the training data
cv_results = cross_val_score(classifier, X_train, y_train, cv=5, scoring=make_scorer(accuracy_score))

# Print the mean accuracy of the cross-validated model
print("\nMean Accuracy (Cross-Validation):", cv_results.mean())

# Print the confusion matrix
conf_matrix = confusion_matrix(y_test, y_pred_test)
print("\nConfusion Matrix:")
print(conf_matrix)

# Calculate accuracy
test_accuracy = accuracy_score(y_test, y_pred_test)
print("\nAccuracy:", test_accuracy)

# Input data for prediction
input_data = [
    'Follow my page pippi54', 
    'Eminen is the best artist over the world!!!',
    'Do you want to see my ass???Click here dasgdkgask.jklashd.net',
    'This song sucks',
    'A feel better when I listen this. Love',
    'Eminem forever!!! I love you and I want to die listening your music',
]

# Transform input data using vectorizer
input_tfidf = vectorizer.transform(input_data)

# Predict the output categories
predictions = classifier.predict(input_tfidf)

# Print the outputs
for sent, category in zip(input_data, predictions):
    print('\nInput:', sent, '\nPredicted category:', category_map[category])