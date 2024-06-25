import pandas as pd
import numpy as np
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB, ComplementNB
from sklearn.neural_network import MLPClassifier #added this to try out
from sklearn.metrics import classification_report, accuracy_score
from nltk.stem.wordnet import WordNetLemmatizer     #lemmentizing words
from nltk.tokenize import word_tokenize             #tokenize words before applying lemmatization

#function for lemmatization (Fix the for loops to what I write)
def lemmatizeText(text):
    tokens = word_tokenize(text)
    lemmatized_tokens = [lemmatizer.lemmatize(token) for token in tokens]
    return ' '.join(lemmatized_tokens)

stops = list(stopwords.words('english'))

# Load your dataset
df = pd.read_csv("eclipse_jdt.csv")

#attepting to drop dups
dups = df.dropna(subset=['Duplicated_issue'])
newData = df.drop(index=dups.index)

#lemmatization
lemmatizer = WordNetLemmatizer()

text = newData['Description'].astype(str).apply(lemmatizeText)
trueLabel = newData['Component'].astype(str).apply(lemmatizeText)

multiVec = MultinomialNB()
compVec = ComplementNB()
clf = MLPClassifier()

xTrain, xTest, yTrain, yTest = train_test_split(trueLabel, text, test_size=0.20, random_state=42)

vec = TfidfVectorizer(stop_words=stops)             #Term Freq
train = vec.fit_transform(yTrain)
test = vec.transform(yTest)

#multinomial
multiVec.fit(train, xTrain)
predicted = multiVec.predict(test)
report = classification_report(xTest, predicted)
print(f'Multinominal Report: \n {report}')

#complement
compTrain = compVec.fit(train, xTrain)
compPredict = compTrain.predict(test)
compReport = classification_report(xTest, compPredict)
print(f'Complement Report: \n{compReport}')

# #mlp 
# clf.fit(train, xTrain)
# clfPredict = clf.predict(test)
# clfReport = classification_report(xTest, clfPredict)
# clfScroe = accuracy_score(xTest, clfPredict)
# print(f'CLF Report: \n{clfScroe}')