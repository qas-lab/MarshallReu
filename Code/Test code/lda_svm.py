# SVM

import gensim
from gensim import corpora
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, top_k_accuracy_score
from cleanData import text

# Parameters
minWords = 10
numTopics = 10
passes = 10

# Ensure text_list is tokenized and cleaned
text_list = text
if isinstance(text_list, list):
    text_list = [doc.split() for doc in text_list]

# Create dictionary and corpus
textDic = corpora.Dictionary(text_list)
corpus = [textDic.doc2bow(doc) for doc in text_list]

# Train LDA model
lda = gensim.models.LdaModel(corpus, num_topics=numTopics, id2word=textDic, passes=passes)

# Convert topics to fixed-size feature vectors
def get_topic_vector(lda_model, bow, num_topics):
    topics = lda_model.get_document_topics(bow, minimum_probability=0.0)
    return [prob for _, prob in sorted(topics, key=lambda x: x[0])]

X = [get_topic_vector(lda, doc, numTopics) for doc in corpus]

# Dummy labels (replace with actual labels)
# Ensure y has the same length as X
y = [i % numTopics for i in range(len(X))]  # Creating dummy labels, modify as needed

# Ensure all classes are present in the labels
unique_labels = list(set(y))
if len(unique_labels) < numTopics:
    raise ValueError(f"Not all classes are present in the labels. Found {len(unique_labels)} unique classes.")

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, shuffle=True)

# Train SVM classifier
svm = SVC(kernel='linear')
svm.fit(X_train, y_train)

# Predict and evaluate
y_pred = svm.predict(X_test)
print(classification_report(y_test, y_pred, zero_division=0))

# Calculate top-k accuracy with the known classes parameter
known_classes = list(range(numTopics))
print(f'Top 1 Accuracy: {top_k_accuracy_score(y_test, svm.decision_function(X_test), k=1, labels=known_classes)}')
print(f'Top 5 Accuracy: {top_k_accuracy_score(y_test, svm.decision_function(X_test), k=5, labels=known_classes)}')
