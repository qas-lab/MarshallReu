# print(f'Top 1 Accuracy: {top_k_accuracy_score(y_test, y_pred, k=1)}')
# print(f'Top 5 Accuracy: {top_k_accuracy_score(y_test, y_pred, k=5)}')

import gensim
from gensim import corpora
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, top_k_accuracy_score
from cleanData import text

minWords = 10
numTopics = 10
passes = 10

text_list = text.tolist()  
textDic = corpora.Dictionary(text_list)
corpus = [textDic.doc2bow(doc) for doc in text_list]

lda = gensim.models.LdaModel(corpus, num_topics=numTopics, id2word=textDic, passes=passes)
topics = [lda[doc] for doc in corpus]

# Extract the topic distribution for each document and the dominant topic as the label
X = [[topic[1] for topic in doc] for doc in topics]
y = [max(doc, key=lambda item: item[1])[0] for doc in topics]  # dominant topic index as label

# Check if lengths match
print(f"Length of X: {len(X)}")
print(f"Length of y: {len(y)}")