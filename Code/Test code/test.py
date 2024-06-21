import pandas as pd
import numpy as np
import nltk
nltk.download('wordnet')
nltk.download('punkt')
from nltk.stem.wordnet import WordNetLemmatizer     #lemmentizing words
from nltk.tokenize import word_tokenize             #tokenize words before applying lemmatization
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Print the top words for each topic (Fix the for loops to what I write)
def print_top_words(model, feature_names, n_top_words):
    for topic_idx, topic in enumerate(model.components_):
        message = f"Topic #{topic_idx}: "
        message += " ".join([feature_names[i] for i in topic.argsort()[:-n_top_words - 1:-1]])
        print(message)
    print()

#function for lemmatization (Fix the for loops to what I write)
def lemmatizeText(text):
    tokens = word_tokenize(text)
    lemmatized_tokens = [lemmatizer.lemmatize(token) for token in tokens]
    return ' '.join(lemmatized_tokens)

df = pd.read_csv("eclipse_jdt.csv")
df['combine'] = df['Description'].astype(str) + ' ' + df['Title'].astype(str) # + ' ' + df['Component'].astype(str)

# Lemmatize all words in documents.
lemmatizer = WordNetLemmatizer()
df['lemmatizeText'] = df['combine'].apply(lemmatizeText)

summary = df['lemmatizeText'].values.astype('U')

x, y = train_test_split(summary, shuffle=True, test_size=0.2, random_state=42)

#max_df=0.95, 
#vec = CountVectorizer(stop_words='english')
vec = TfidfVectorizer(stop_words='english')

train = vec.fit_transform(x)
test = vec.transform(y)

numTopics = 10

#lda model
#switching max_iter from 10 to 6 gives slightly better prediction accuracy
lda = LatentDirichletAllocation(n_components=numTopics, random_state=42)
training = lda.fit(train)

print_top_words(lda, vec.get_feature_names_out(), 10)

trainLda = lda.transform(train)
testing = lda.transform(test)

average = 0
count = 0

#printing topic predictions and the accuracy of it
for count, topic in enumerate(testing[:10]):
    maxValue = np.max(topic)
    index = np.argmax(topic)
    print(f"Bug Report #{count} Topic {index} Distrubtion {maxValue}")
    average += maxValue

count += 1
average = average / count 
print(f'\nThe average is: {average}\n')

# #Added text classifier, needs to be tuned more
# print()
# nbClass = MultinomialNB()
# nbClass.fit(trainLda,x)
# predict = nbClass.predict(testing)

# acc = accuracy_score(y, predict)
# print(f'Naive Bayes Score: {acc}')