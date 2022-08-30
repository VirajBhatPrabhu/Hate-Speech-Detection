import pandas as pd
import numpy as np
from nltk.util import pr
import re
import nltk



df = pd.read_csv('Data/Hatespeech.csv')

df['labels'] = df['class'].map({0: 'Hate Speech', 1: 'Offensive Language', 2: 'Neither Hate nor Offensive'})
print(df['labels'].value_counts())

df = df[['tweet', 'labels']]

from nltk.stem.snowball import SnowballStemmer
from nltk.corpus import stopwords
import string

stopwords = set(stopwords.words('english'))
stemmer = SnowballStemmer('english')


def clean_text(text):
    text = str(text).lower()
    text = re.sub('\[.*?\]', '', text)
    text = re.sub('https?://\S+|www\.\S+', '', text)
    text = re.sub('<.*?>+', '', text)
    text = re.sub('[%s]' % re.escape(string.punctuation), '', text)
    text = re.sub('\n', '', text)
    text = re.sub('\w*\d\w*', '', text)
    text = re.sub('rt', '', text)

    text = [word for word in text.split(' ') if word not in stopwords]
    text = " ".join(text)
    text = [stemmer.stem(w) for w in text.split(' ')]
    text = " ".join(text)
    return text


df['tweet'] = df['tweet'].apply(clean_text)

x = np.array(df['tweet'])
y = np.array(df['labels'])

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split

cv = CountVectorizer()
X = cv.fit_transform(x)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)


from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report

model1=DecisionTreeClassifier()
model1.fit(X_train,y_train)
print(model1.score(X_test,y_test))
predictions = model1.predict(X_test)

print(classification_report(y_test, predictions))


import pickle
pickle.dump(model1,open('Hatespeechmodel.pkl','wb'))
pickle.dump(cv,open('countvectorizer.pkl','wb'))
