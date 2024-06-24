import pandas as pd
import numpy as np
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB, CategoricalNB, ComplementNB
from sklearn.linear_model import LogisticRegression # See what this is
from sklearn.metrics import accuracy_score

stops = list(stopwords.words('english'))

# Load your dataset
df = pd.read_csv("eclipse_jdt.csv")

#attepting to drop dups
dups = df.dropna(subset=['Duplicated_issue'])
newData = df.drop(index=dups.index)

text = newData['Description']
trueLabel = newData['Component']

#This is strickly for testing purposes
tempDup = df.drop_duplicates(subset=['Component'])
labels = list(tempDup['Component'])
print(labels)

data = text.astype(str) + ' ' + trueLabel.astype(str)

multiVec = MultinomialNB()
#catVec = CategoricalNB()
#compVec = ComplementNB()

x,y = train_test_split(data, test_size=0.2)

print(y)