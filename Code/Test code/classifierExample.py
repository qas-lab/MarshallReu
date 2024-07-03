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
from nltk.stem import PorterStemmer


# Function to remove stop words
def removeStopWords(text):
    words = text.split()
    filtered_words = [word for word in words if word not in stops]
    return ' '.join(filtered_words)

# Function to remove code snippets and URLs
def remove_code_snippets(text):
    text = re.sub(r'`[^`]*`', ' ', text)
    text = re.sub(r'```[^```]*```', ' ', text)
    return text

def remove_urls(text):
    text = re.sub(r'http\S+|www\S+|https\S+', ' ', text, flags=re.MULTILINE)
    return text

# Function to replace punctuation with spaces
def replace_punctuation_with_space(text):
    translator = str.maketrans(string.punctuation, ' ' * len(string.punctuation))
    return text.translate(translator)

# Preprocessing function
def preProcessing(text):

    text = remove_code_snippets(text)                                                   # Remove code snippets
    text = remove_urls(text)                                                            # Remove URLs
    text = text.lower()                                                                 # Lowercase
    text = re.sub(r'\d+', ' ', text)                                                    # Remove numbers
    text = removeStopWords(text)                                                        # Remove stop words
    text = replace_punctuation_with_space(text)                                         # Replace punctuation with spaces
    text = re.sub(r'\b[a-z]\b', ' ', text)                                              # Remove single characters
    text = re.sub(r'\s+', ' ', text).strip()                                            # Remove extra spaces
    text = ' '.join([stem.stem(word) for word in text.split()])                         # Apply stemming

    return text

stops = list(stopwords.words('english'))

# Load your dataset
df = pd.read_csv("eclipse_jdt.csv")

#lemmatization
lemmatizer = WordNetLemmatizer()
stem = PorterStemmer()

summary = df['Title'].astype(str).apply(preProcessing)
#text2 = df['Product'].astype(str).apply(clean_text).apply(lemmatizeText)
trueLabel = df['Description'].astype(str).apply(preProcessing)

text = summary + '' + trueLabel



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