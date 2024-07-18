# Bert topic model

from bertopic import BERTopic
from bertopic.representation import KeyBERTInspired
from sentence_transformers import SentenceTransformer
from bertopic.vectorizers import ClassTfidfTransformer

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import PCA, NMF
from sklearn.cluster import KMeans

from umap import UMAP
from hdbscan import HDBSCAN
import pandas as pd

from scipy.cluster import hierarchy as sch

# # Load and prepare data
# data = pd.read_csv('eclipse_jdt.csv')
# #data = pd.read_csv('mozilla_firefox.csv')

# docs = data[['Title', 'Description']]
# docs['Title'] = docs['Title'].fillna('')
# docs['Description'] = docs['Description'].fillna('')
# docs['Text'] = docs['Title'] + docs['Description']
# docs = docs['Text'].tolist()

# Load and prepare data
data = pd.read_csv('dataset.csv')

# docs = data['Summary']
# docs = docs.fillna('')
# docs = docs.tolist()

# docs = data[data['Assignee Real Name'] == 'Martin Aeschlimann']
# docs = docs['Summary']
# docs = docs.fillna('')
# docs = docs.tolist()

docs = data[data['Assignee Real Name'] == 'Curtis Windatt']
docs = docs[['Summary','Product','Component']]
docs['Summary'] = docs['Summary'].fillna('')
docs['Product'] = docs['Product'].fillna('')
docs['Component'] = docs['Component'].fillna('')
docs['Text'] = docs['Component'] + ' ' + docs['Product'] + ' ' + docs['Summary']
docs = docs['Text'].tolist()

# groupDocs = data.groupby(['Assignee Real Name','Summary'])
# print(groupDocs.head())

# 1. Embedding
sentence_model = SentenceTransformer('all-mpnet-base-v2')

#2. Dimensionality Reduction
umap_model = UMAP(metric='cosine', 
                  n_neighbors=3, # Change back to 5?
                  min_dist=0.0)

#3. Clustering 
hdbscan_model = HDBSCAN(min_cluster_size=60,
                        min_samples=1, 
                        metric='euclidean', 
                        cluster_selection_method='leaf',
                        prediction_data=True)
# hdbscan_model = KMeans(n_clusters=3)

# 4. Token
vec = CountVectorizer(stop_words='english',
                      ngram_range=(1,2)) 

# 5. Weighting schemes
ctif = ClassTfidfTransformer(reduce_frequent_words=True)

# 6. Fine Tune
representation = KeyBERTInspired()    

# Create and fit BERTopic model
topic_model = BERTopic(embedding_model=sentence_model,          #1
                       umap_model=umap_model,                   #2
                       hdbscan_model=hdbscan_model,             #3
                       vectorizer_model=vec,                    #4
                       ctfidf_model=ctif,                       #5
                       representation_model=representation,     #6
                       )

topics, probs = topic_model.fit_transform(docs)

# topic_model.reduce_topics(docs, nr_topics=3) # reduce topics for HDBSCAN

# Reduce Outliers
# newTopics = topic_model.reduce_outliers(docs, topics, strategy='c-tf-idf')

# Visualize topics
intertopic = topic_model.visualize_topics()
intertopic.write_html('intertopic.html')

# Get topic and document information
document_info = topic_model.get_topic_info()
topic_info = topic_model.get_document_info(docs)
print(document_info)
print(topic_info)

# Hierarchical topics
linkage_function = lambda x: sch.linkage(x, 'single', optimal_ordering=True)
hierarchical_topics = topic_model.hierarchical_topics(docs, linkage_function=linkage_function)
hierarchical = topic_model.visualize_hierarchy(hierarchical_topics=hierarchical_topics)
tree = topic_model.get_topic_tree(hierarchical_topics)
hierarchical.write_html('hierarchcalTopics.html')

# Saving topic model
topic_model.save('bert_topic_model')

################################################################################################################################################################
################################################################################################################################################################

# kmeansModel = BERTopic.load('bert_topic_model_KMeans')
# bertBigram = BERTopic.load('bert_bigram_good_cluster')
# bertGoodCluster2 = BERTopic.load('bert_topic_good_clusters')
# bertHDBSCAN = BERTopic.load('bert_topic_model_HDBSCAN_238')

# # # Visualize topics
# # kmeansIntertopic = kmeansModel.visualize_topics()
# # kmeansIntertopic.write_html('kmeansIntertopic.html')

# # bertIntertopic = bertBigram.visualize_topics()
# # bertIntertopic.write_html('bigramIntertopic.html')

# # goodCluster2 = bertGoodCluster2.visualize_topics()
# # goodCluster2.write_html('cluster_2_intertopic.html')

# # hdbscanIntertopic = bertHDBSCAN.visualize_topics()
# # hdbscanIntertopic.write_html('hdbscanIntertopic.html')

# kmeansTopicInfo = kmeansModel.get_topic_info()
# kmeansDocInfo = kmeansModel.get_document_info(docs)

# bertTopicInfo = bertBigram.get_topic_info()
# bertDocInfo = bertBigram.get_document_info(docs)

# goodTopicInfo = bertGoodCluster2.get_topic_info()
# goodDocInfo = bertGoodCluster2.get_document_info(docs)

# hdbTopicInfo = bertHDBSCAN.get_topic_info()
# hdbDocInfo = bertHDBSCAN.get_document_info(docs)

# print(f'\n{kmeansTopicInfo}\n')
# print(f'\n{kmeansDocInfo}\n')
# print(f'\n{bertTopicInfo}\n')
# print(f'\n{bertDocInfo}\n')
# print(f'\n{goodTopicInfo}\n')
# print(f'\n{goodDocInfo}\n')
# print(f'\n{hdbTopicInfo}\n')
# print(f'\n{hdbDocInfo}\n')