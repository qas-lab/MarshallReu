# This is the Baseline LDA model, that I will use for my prototype, This file is only for the LDA 

import pandas as pd
import re
import string
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer,TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.tokenize import word_tokenize
import numpy as np
import torch
import torch.nn.functional as F
import pyLDAvis

#function for lemmztizing
def wordLem(text):

    tokens = word_tokenize(text)
    lemToken = [lem.lemmatize(tokens) for tokens in tokens]

    return ' '.join(lemToken)

#function for printing topics
def printTopics(model, featNames, topics):

    for topicIdx, topic in enumerate(model.components_):
        message = f'Topic #{topicIdx}: \t'
        index = topic.argsort()[:-topics -1:-1]
        message += ' '.join([featNames[i] for i in index if i < len(featNames)])
        print(message)

    print()

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
    #text = ' '.join([stem.stem(word) for word in text.split()])                         # Apply stemming

    return text

#variables to use within code
stops = list(stopwords.words('english'))
numTopics = 3

#Creating instances of classes
lem = WordNetLemmatizer()
lda = LatentDirichletAllocation(n_components=numTopics, random_state=42)
#vec = TfidfVectorizer(stop_words=stops, max_df=0.95, ngram_range=(1,2))                                                          #min_df added to test how it works
vec = CountVectorizer(stop_words=stops, max_df=0.95, ngram_range=(1,3), max_features=10000)

#opening file and removing duplicate reports
df = pd.read_csv('eclipse_jdt.csv')
dupl = df.dropna(subset=['Duplicated_issue'])
newData = df.drop(index=dupl.index)
text = newData['Description'].astype(str).apply(preProcessing).apply(wordLem)

#splitting testing data
train, test = train_test_split(text, shuffle=True, random_state=42, test_size=0.2)

xTrain = vec.fit_transform(train)
yTest = vec.transform(test)

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

n_top_words = 1000
feature_names = vec.get_feature_names_out()
top_words = get_top_words(lda, feature_names, n_top_words)

# Create a dataframe to store topics and their top words
topic_names = [f'Topic_{i}' for i in range(lda.n_components)]
top_words_df = pd.DataFrame(top_words, index=topic_names)
top_words_df.columns = [f'Word_{i}' for i in range(1, n_top_words + 1)]

# Save the dataframe to a CSV file
top_words_df.to_csv("topics.csv", index=True)

output = torch.rand(1, 10)
target = torch.randint(10, (1,))
loss = F.cross_entropy(output, target)
perp = torch.exp(loss)
print(f'\n Torch Perplexity: {perp}\n')



# # Calculate Coherence Score using gensim
# # Convert sklearn's LDA output to gensim format
# corpus = [gensim.matutils.sparse2full(c, numTerms=len(names)) for c in xTrain]
# text_list = [doc.split() for doc in train]

# # Create a dictionary
# dictionary = gensim.corpora.Dictionary(text_list)

# # Generate coherence score
# lda_gensim = gensim.models.ldamodel.LdaModel(id2word=dictionary, num_topics=numTopics, random_state=42)
# lda_gensim.update(corpus)
# coherence_model_lda = CoherenceModel(model=lda_gensim, texts=text_list, dictionary=dictionary, coherence='c_v')
# coherence_lda = coherence_model_lda.get_coherence()
# print(f'Coherence Score: {coherence_lda:.4f}')

