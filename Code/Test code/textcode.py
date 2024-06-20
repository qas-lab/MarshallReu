import pandas as pd
import numpy as np
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Print the top words for each topic
def print_top_words(model, feature_names, n_top_words):
    for topic_idx, topic in enumerate(model.components_):
        message = f"Topic #{topic_idx}: "
        message += " ".join([feature_names[i] for i in topic.argsort()[:-n_top_words - 1:-1]])
        print(message)
    print()

df = pd.read_csv("dataset.csv")

summary = df['Summary']

x, y = train_test_split(summary, shuffle=True, test_size=0.2, random_state=42)

vec = CountVectorizer(max_df=0.95, stop_words='english')

train = vec.fit_transform(x)
test = vec.transform(y)

numTopics = 10

#lda model
lda = LatentDirichletAllocation(n_components=numTopics, random_state=42)
training = lda.fit(train)

print_top_words(lda, vec.get_feature_names_out(), 10)

trainLda = lda.transform(train)
testing = lda.transform(test)

#printing topic predictions and the accuracy of it
for i, topic in enumerate(testing[:10]):
    maxValue = np.max(topic)
    index = np.argmax(topic)
    print(f"Bug Report #{i} Topic {index} Distrubtion {maxValue}")

#Need to run a text classifier on the topics so that I can output the values.......(Thursday)

nbClass = MultinomialNB()
nbClass.fit(trainLda,x)
predict = nbClass.predict(testing)

acc = accuracy_score(y, predict)
print()
print(f'Naive Bayes Score: {acc}')
