# import numpy as np
# import string
#import nltk
import pandas as pd
from sklearn.ensemble import AdaBoostClassifier
from tempfile import mkdtemp
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.metrics import classification_report
#nltk.download('stopwords')
from nltk.corpus import stopwords


# reading in data
df_test = pd.read_csv("C:\\Users\\roons\\OneDrive\\Documents\\Fall 2020\\STAT 495R\\FakeNews\\test.csv")
df_train = pd.read_csv("C:\\Users\\roons\\OneDrive\\Documents\\Fall 2020\\STAT 495R\\FakeNews\\train.csv")

# fill na articles
df_train['text'].fillna(value='No Article Provided', inplace=True)
df_test['text'].fillna(value='No Article Provided', inplace=True)

# split up data to training and df_test

news_train, news_test, label_train, label_test = \
    train_test_split(df_train['text'], df_train['label'], test_size=0.2)

# list of stop stop_words
stoplist = list(stopwords.words("english"))

# second model created based on grid adaboost
param_grid = dict(classifier__n_estimators=[200, 250, 300],
                  classifier__learning_rate=[.1, .08, .05])
from shutil import rmtree
rmtree(cachedir)
cachedir = mkdtemp()
# mygrid = GridSearchCV(SVC(), param_grid)

gridadapipe = Pipeline([
    ('bow', CountVectorizer(stop_words=stoplist, max_features=50000, ngram_range=(2,3))),  # strings to token integer counts
    ('tfidf', TfidfTransformer()),  # integer counts to weighted TF-IDF scores
    ('classifier', AdaBoostClassifier()),  # train on TF-IDF vectors w/ AdaBoost classifier
], verbose=True, memory=cachedir)

grid = GridSearchCV(gridadapipe, param_grid)
print("grid instantiated")
grid.fit(news_train, label_train)
print("Fit done")
print("building predictions...")
gridadatestpreds = grid.predict(news_test)
print(grid.best_params_)
gridadapreds = grid.predict(df_test['text'])
print(classification_report(gridadatestpreds, label_test))

output = pd.DataFrame(gridadapreds, index=df_test['id'])
output.to_csv('C:/Users/roons/OneDrive/Documents/Fall 2020/STAT 495R/FakeNews/pyFNpredsgridadaatom.csv')
