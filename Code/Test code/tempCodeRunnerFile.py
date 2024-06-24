import pandas as pd
import numpy as np
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB, CategoricalNB, ComplementNB
from sklearn.linear_model import LogisticRegression # See what this is
from sklearn.metrics import accuracy_score

stops = list(stopwords.words('english'))
print(stops)

# Load your dataset
df = pd.read_csv("eclipse_jdt.csv")

#attepting to drop dups
dups = df.dropna(subset=['Duplicated_issue'])
newData = df.drop(index=dups.index)

text = newData['Description']
trueLabel = newData['Component']

data1 = np.array([trueLabel, text])  #You can also use datafram from panda to convert to array

data = word_tokenize(data1)

multiVec = MultinomialNB()
catVec = CategoricalNB()
compVec = ComplementNB()

x,y = train_test_split(data, test_size=0.2)

vec = TfidfVectorizer(stop_words=stops)
train = vec.fit_transform(x)