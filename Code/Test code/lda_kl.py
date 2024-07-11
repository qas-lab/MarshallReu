from gensim import corpora, models
from gensim.models.ldamodel import LdaModel
from gensim.matutils import kullback_leibler
import numpy as np
from cleanData import text

text_list = text
text_list = text.tolist()  
textDic = corpora.Dictionary(text_list)
corpus = [textDic.doc2bow(doc) for doc in text_list]

# Build LDA model
lda_model = LdaModel(corpus, id2word=textDic, num_topics=10)

# Compute topic distributions for each document
topic_distributions = [lda_model[doc] for doc in corpus]

# Compute KL divergence between pairs of documents
num_docs = len(corpus)
kl_divergences = np.zeros((num_docs, num_docs))
for i in range(num_docs):
    for j in range(num_docs):
        if i != j:
            kl_divergences[i, j] = kullback_leibler(topic_distributions[i], topic_distributions[j])

# Example of accessing KL divergence between document 0 and document 1
kl_divergence_0_1 = kl_divergences[0, 1]
