# This is the Baseline LDA model, that I will use for my prototype, This file is only for the LDA 

import pandas as pd
import re
import string
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer,TfidfVectorizer
from sklearn.model_selection import train_test_split
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.tokenize import word_tokenize

#function for lemmztizing
def wordLem(text):

    tokens = word_tokenize(text)
    lemToken = [lem.lemmatize(tokens) for tokens in tokens]

    return ' '.join(lemToken)

# Function for text cleaning
def cleanText(text):
    text = text.lower()                                                    # Convert to lowercase
    text = re.sub(r'\d+', ' ', text)                                       # Remove numbers
    text = re.sub(r'\b[nan]', ' ', text)                                   # Remove nan (Hard Coded)
    #text = text.translate(str.maketrans(' ', ' ', string.punctuation))     # Remove punctuation
    return text

#function for printing topics
def printTopics(model, featNames, topics):

    for topicIdx, topic in enumerate(model.components_):
        message = f'Topic #{topicIdx}: \t'
        index = topic.argsort()[:-topics -1:-1]
        message += ' '.join([featNames[i] for i in index if i < len(featNames)])
        print(message)

    print()

#variables to use within code
stops = list(stopwords.words('english'))
numTopics = 10

#Creating instances of classes
lem = WordNetLemmatizer()
lda = LatentDirichletAllocation(n_components=numTopics, random_state=42, doc_topic_prior=0.6)
#vec = TfidfVectorizer(stop_words=stops, max_df=0.90)
vec = CountVectorizer(stop_words=stops, max_df=0.90, min_df=0.01)

#opening file and removing duplicate reports
df = pd.read_csv('eclipse_jdt.csv')
dupl = df.dropna(subset=['Duplicated_issue'])
newData = df.drop(index=dupl.index)
text = newData['Description'].astype(str).apply(cleanText).apply(wordLem) + ' ' + newData['Component'].astype(str)

#splitting testing data
train, test = train_test_split(text, shuffle=True, random_state=42, test_size=0.2)

xTrain = vec.fit_transform(train)
yYest = vec.transform(test)

model = lda.fit_transform(xTrain)
names = list(vec.get_feature_names_out())

# Only for testing to see the topics that are printed out
printTopics(lda, names, 10)
print()

# Extract top words for each topic
def get_top_words(model, feature_names, n_top_words):
    top_words = []
    for topic_idx, topic in enumerate(model.components_):
        top_words_for_topic = [feature_names[i] for i in topic.argsort()[:-n_top_words - 1:-1]]
        top_words.append(top_words_for_topic)
    return top_words

n_top_words = 400
feature_names = vec.get_feature_names_out()
top_words = get_top_words(lda, feature_names, n_top_words)

# Create a dataframe to store topics and their top words
topic_names = [f'Topic_{i}' for i in range(lda.n_components)]
top_words_df = pd.DataFrame(top_words, index=topic_names)
top_words_df.columns = [f'Word_{i}' for i in range(1, n_top_words + 1)]

# Save the dataframe to a CSV file
top_words_df.to_csv("topics.csv", index=True)

print("Topics and top words saved to 'topics.csv'")
