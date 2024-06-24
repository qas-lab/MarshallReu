import pandas as pd
import numpy as np
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB, CategoricalNB, ComplementNB
from sklearn.linear_model import LogisticRegression # See what this is
from sklearn.metrics import accuracy_score, classification_report

stops = list(stopwords.words('english'))

# Load your dataset
df = pd.read_csv("eclipse_jdt.csv")

#attepting to drop dups
dups = df.dropna(subset=['Duplicated_issue'])
newData = df.drop(index=dups.index)

text = newData['Description']
trueLabel = newData['Component']

# #This is strickly for testing purposes
# tempDup = df.drop_duplicates(subset=['Component'])
# labels = list(tempDup['Component'])
# print(f'The labels are {labels} and the size is {len(labels)}\n')

#data = text.astype(str) + ' ' + trueLabel.astype(str)

multiVec = MultinomialNB()
catVec = CategoricalNB()
compVec = ComplementNB()

xTrain, xTest, yTrain, yTest = train_test_split(trueLabel.astype(str), text.astype(str), test_size=0.2, random_state=42)

vec = TfidfVectorizer(stop_words=stops)             #Term Freq
train = vec.fit_transform(yTrain)
test = vec.transform(yTest)

countVec = CountVectorizer(stop_words=stops)        #Count Vec
countTrain = countVec.fit_transform(yTrain)
countTest = countVec.transform(yTest)

#multinomial
multiVec.fit(train, xTrain)
predicted = multiVec.predict(test)
acc = accuracy_score(xTest, predicted)
report = classification_report(xTest, predicted)
print(f'Multinominal Accuracy Score is: {acc} \n')
print(f'Multinominal Report {report}')