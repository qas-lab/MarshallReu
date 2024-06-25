import pandas as pd
import numpy as np
import re
import string
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB, ComplementNB
from sklearn.neural_network import MLPClassifier    #MLF classifier
from sklearn.metrics import classification_report, accuracy_score
from nltk.stem.wordnet import WordNetLemmatizer     #lemmentizing words
from nltk.tokenize import word_tokenize             #tokenize words before applying lemmatization
from sklearn.ensemble import RandomForestClassifier # CLF classifier


#function for lemmatization
def lemmatizeText(text):
    tokens = word_tokenize(text)
    lemmatized_tokens = [lemmatizer.lemmatize(token) for token in tokens]
    return ' '.join(lemmatized_tokens)

#removing time stamp
def removeTimeStamp(text):
    timestamp_pattern = r'[a-zA-Z]{3}\s\(\d{2}\.\d{2}\.\d{4}\s\d{2}:\d{2}:\d{2}\s\b{a-zA-Z}{2}\)'
    return re.sub(timestamp_pattern, '', text)

# Function for text cleaning
def clean_text(text):
    text = removeTimeStamp(text)                                      # Remove timestamps
    text = text.lower()                                               # Convert to lowercase
    text = re.sub(r'\d+', '', text)                                   # Remove numbers
    text = text.translate(str.maketrans('', '', string.punctuation))  # Remove punctuation
    text = text.strip()                                               # Remove leading/trailing whitespace
    return text

stops = list(stopwords.words('english'))

# Load your dataset
df = pd.read_csv("eclipse_jdt.csv")

#attepting to drop dups
dups = df.dropna(subset=['Duplicated_issue'])
newData = df.drop(index=dups.index)

#lemmatization
lemmatizer = WordNetLemmatizer()

text = newData['Description'].astype(str).apply(clean_text).apply(lemmatizeText)
trueLabel = newData['Component'].astype(str).apply(clean_text).apply(lemmatizeText)

print(text)

# multiVec = MultinomialNB()
# compVec = ComplementNB()
# clf = MLPClassifier()

# xTrain, xTest, yTrain, yTest = train_test_split(trueLabel, text, test_size=0.2, random_state=42)

# vec = TfidfVectorizer(stop_words=stops)             #Term Freq
# train = vec.fit_transform(yTrain)
# test = vec.transform(yTest)

# #multinomial
# multiVec.fit(train, xTrain)
# predicted = multiVec.predict(test)
# report = classification_report(xTest, predicted)
# print(f'Multinominal Report: \n {report}')

# #complement
# compTrain = compVec.fit(train, xTrain)
# compPredict = compTrain.predict(test)
# compReport = classification_report(xTest, compPredict)
# print(f'Complement Report: \n{compReport}')

# #clf
# clf = RandomForestClassifier(n_estimators=100)
# clfTrain = clf.fit(train, xTrain)
# clfPred = clf.predict(test)
# clfReport = classification_report(xTest, clfPred)
# print(f'CLF Report: {clfReport}\n')

# #mlp 
# clf.fit(train, xTrain)
# clfPredict = clf.predict(test)
# clfReport = classification_report(xTest, clfPredict)
# clfScroe = accuracy_score(xTest, clfPredict)
# print(f'MLP Report: \n{clfScroe} \n')