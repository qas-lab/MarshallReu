import pandas as pd
import numpy as np
import re
import string
from spellchecker import SpellChecker
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import ComplementNB
from sklearn.neural_network import MLPClassifier    #MLF classifier
from sklearn.metrics import classification_report, accuracy_score
from nltk.stem.wordnet import WordNetLemmatizer     #lemmentizing words
from nltk.tokenize import word_tokenize             #tokenize words before applying lemmatization
from sklearn.ensemble import RandomForestClassifier # CLF classifier
from sklearn.linear_model import LogisticRegression # Logistic Regression
from sklearn.ensemble import StackingClassifier


#function for lemmatization
def lemmatizeText(text):
    tokens = word_tokenize(text)
    lemmatized_tokens = [lemmatizer.lemmatize(token) for token in tokens]
    return ' '.join(lemmatized_tokens)

#removing time stamp
def remove(text):
    #timestamp_pattern = r'\([A-Za-z0-9]+\)'
    timestamp_pattern = r'\S([A-Za-z]*\s*)\d{1,2}:\d{2}\s*[APap][Mm]'
    return re.sub(timestamp_pattern, '', text)


# Function for text cleaning
def clean_text(text):
    text = remove(text)                                               # Remove stamps
    text = text.lower()                                               # Convert to lowercase
    text = re.sub(r'\d+', '', text)                                   # Remove numbers                       
    text = text.translate(str.maketrans('', '', string.punctuation))  # Remove punctuation
    text = text.strip()                                               # Remove leading/trailing whitespace 
    return text

stops = list(stopwords.words('english'))

# Load your dataset
df = pd.read_csv("dataset.csv")

#lemmatization
lemmatizer = WordNetLemmatizer()

summary = df['Summary'].astype(str).apply(clean_text).apply(lemmatizeText)
#text2 = df['Product'].astype(str).apply(clean_text).apply(lemmatizeText)
trueLabel = df['Component'].astype(str) 

text = summary + '' + df['Product'].astype(str)

# print(text) #Only for looking at the word strips

compVec = ComplementNB()
clf = RandomForestClassifier()
#mlp = MLPClassifier()
# logisticReg = LogisticRegression()

# # Define a Stacking Classifier
# stackingClf = StackingClassifier(estimators=[
#     ('clf', clf),
#     ('mlp', mlp)
# ], final_estimator=logisticReg)

xTrain, xTest, yTrain, yTest = train_test_split(trueLabel, text, test_size=0.2, random_state=42, shuffle=True)

#vec = CountVectorizer(stop_words=stops)              #count Freq
vec = TfidfVectorizer(stop_words=stops)               #Term Freq
train = vec.fit_transform(yTrain)
test = vec.transform(yTest)

#complement
compTrain = compVec.fit(train, xTrain)
compPredict = compTrain.predict(test)
compReport = classification_report(xTest, compPredict, zero_division=0)
print(f'Complement Report: \n{compReport}')

#clf
clfTrain = clf.fit(train, xTrain)
clfPred = clf.predict(test)
clfReport = classification_report(xTest, clfPred)
print(f'CLF Report: \n{clfReport}\n')

# #mlp 
# mlp.fit(train, xTrain)
# clfPredict = mlp.predict(test)
# clfReport = classification_report(xTest, clfPredict, zero_division=0)
# clfScroe = accuracy_score(xTest, clfPredict)
# print(f'MLP Report: \n{clfScroe} \n')

# # # Train and evaluate the Stacking Classifier
# # stackingClf.fit(train, xTrain)
# # stackingPredict = stackingClf.predict(test)
# # print(f'Stacking Classifier Report:\n{classification_report(xTest, stackingPredict)}')

# # Logistic Regression
# logisticReg.fit(train, xTrain)
# logisticPred = logisticReg.predict(test)
# logisticReport = classification_report(xTest, logisticPred, zero_division=0)
# print(f'Logistic Regression Baseline: \n{logisticReport} \n')