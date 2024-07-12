# Bert topic model

from bertopic import BERTopic
from bertopic.representation import KeyBERTInspired
from sklearn.metrics import classification_report, accuracy_score
from cleanData import text
from sentence_transformers import SentenceTransformer
from umap import UMAP

docs = text.astype(str).tolist()  

representation = KeyBERTInspired()
sentence_model = SentenceTransformer('all-mpnet-base-v2')
umap_model = UMAP()

topic_model = BERTopic(representation_model=representation,
                       embedding_model=sentence_model)

topics, probs = topic_model.fit_transform(docs)
#new_topics = topic_model.reduce_topics(docs, nr_topics=20)

intertopic = topic_model.visualize_topics()
intertopic.write_html('intertopic.html')

document_info = topic_model.get_topic_info()
print(document_info)


