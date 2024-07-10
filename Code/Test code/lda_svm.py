import gensim
from gensim import corpora
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, top_k_accuracy_score
from cleanData import text

minWords = 10
numTopics = 10
passes = 15

text_list = text
text_list = text.tolist()  
textDic = corpora.Dictionary(text_list)
corpus = [textDic.doc2bow(doc) for doc in text_list]

lda = gensim.models.LdaModel(corpus, num_topics=numTopics, id2word=textDic, passes=passes)
topics = [lda[doc] for doc in corpus]

X = [[topic[1] for topic in doc] for doc in topics]
y = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]  # Not enough labels 

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

svm = SVC(kernel='linear')
svm.fit(X_train, y_train)

y_pred = svm.predict(X_test)
print(classification_report(y_test, y_pred))

print(f'Top 1 Accuracy: {top_k_accuracy_score(y_test, y_pred, k=1)}')
print(f'Top 5 Accuracy: {top_k_accuracy_score(y_test, y_pred, k=5)}')
