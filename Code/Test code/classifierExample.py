import pandas as pd
import numpy as np
from nltk.corpus import stopwords
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

data = np.array([trueLabel, text])  #You can also use datafram from panda to convert to array

multiVec = MultinomialNB()
catVec = CategoricalNB()
compVec = ComplementNB()

x,y = train_test_split(data, test_size=0.2)

vec = TfidfVectorizer(stop_words=stops)
train = vec.fit_transform(x)
test = vec.transform(y)

multiVec.fit(train, x)

predicted = multiVec.predict(test)

acc = accuracy_score(y, predicted)
print(f'Accuracy Score is: {acc}')
print(predicted)

# compTrain = compVec.fit(train, x)
# compPredict = compTrain.predict(test)

# accComp = accuracy_score(y, compPredict)
# print(f'\nComplementNB Accuracy Score is : {accComp}')
# print(compPredict, '\n')