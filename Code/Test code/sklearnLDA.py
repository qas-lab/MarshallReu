# This is the Baseline LDA model, that I will use for my prototype, This file is only for the LDA 

from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer,TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from nltk.stem.wordnet import WordNetLemmatizer
import numpy as np
import torch
import torch.nn.functional as F
import pyLDAvis
from cleanData import text
from sklearn.datasets import fetch_20newsgroups

#function for printing topics
def printTopics(model, featNames, topics):

    for topicIdx, topic in enumerate(model.components_):
        message = f'Topic #{topicIdx}: \t'
        index = topic.argsort()[:-topics -1:-1]
        message += ' '.join([featNames[i] for i in index if i < len(featNames)])
        print(message)

    print()

numTopics = 10

#Creating instances of classes
lda = LatentDirichletAllocation(n_components=numTopics, random_state=100)
vec = TfidfVectorizer()                                                          #min_df added to test how it works
#vec = CountVectorizer(max_df=0.90, min_df=0.1)

#Doing this to ensure I am intializing data
text = text.astype(str)

# docs = fetch_20newsgroups(subset='all',  remove=('headers', 'footers', 'quotes'))['data']
# text = docs

#splitting testing data
train, test = train_test_split(text, shuffle=True, random_state=42, test_size=0.2)

xTrain = vec.fit_transform(train)
yTest = vec.transform(test)

model = lda.fit_transform(xTrain)
names = list(vec.get_feature_names_out())

# Only for testing to see the topics that are printed out
printTopics(lda, names, 10)
print()

# # Extract top words for each topic
# def get_top_words(model, feature_names, n_top_words):
#     top_words = []
#     for topic_idx, topic in enumerate(model.components_):
#         top_words_for_topic = [feature_names[i] for i in topic.argsort()[:-n_top_words - 1:-1]]
#         top_words.append(top_words_for_topic)
#     return top_words

# n_top_words = 100
# feature_names = vec.get_feature_names_out()
# top_words = get_top_words(lda, feature_names, n_top_words)

# # Create a dataframe to store topics and their top words
# topic_names = [f'Topic_{i}' for i in range(lda.n_components)]
# top_words_df = pd.DataFrame(top_words, index=topic_names)
# top_words_df.columns = [f'Word_{i}' for i in range(1, n_top_words + 1)]

# # Save the dataframe to a CSV file
# top_words_df.to_csv("topics.csv", index=True)

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

