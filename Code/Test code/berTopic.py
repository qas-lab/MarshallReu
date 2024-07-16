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

# Load and prepare data
data = pd.read_csv('eclipse_jdt.csv')
#data = pd.read_csv('mozilla_firefox.csv')

docs = data[['Title', 'Description']]
docs['Title'] = docs['Title'].fillna('')
docs['Description'] = docs['Description'].fillna('')
docs['Text'] = docs['Title'] + docs['Description']
docs = docs['Text'].tolist()

# hdbscanModel = BERTopic.load('bert_topic_model_HDBSCAN_238')

# topicsToMerge1 = [182,35,5,6,85,143,39,208,22,60,112,18,105,113,158,16,154,207,181]
# topicsToMerge2 = [191,153,13,9,100,147,146,161,14,41,11,151,88,123]
# # topicsToMerge3 = [117,10,25,219,101,171,125,201,199]
# # topicsToMerge4 = [74,167,42,227,180,202,121,65,166]

# hdbscanModel.merge_topics(docs, topicsToMerge1)
# hdbscanModel.merge_topics(docs, topicsToMerge2)
# # hdbscanModel.merge_topics(docs, topicsToMerge3)
# # hdbscanModel.merge_topics(docs, topicsToMerge4)

# # Visualize topics
# intertopic = hdbscanModel.visualize_topics()
# intertopic.write_html('intertopicMerged.html')

# # Hierarchical topics
# linkage_function = lambda x: sch.linkage(x, 'single', optimal_ordering=True)
# hierarchical_topics = hdbscanModel.hierarchical_topics(docs, linkage_function=linkage_function)
# hierarchical = hdbscanModel.visualize_hierarchy(hierarchical_topics=hierarchical_topics)
# tree = hdbscanModel.get_topic_tree(hierarchical_topics)
# hierarchical.write_html('MergedhierarchcalTopics.html')


# kMeans = BERTopic.load('bert_topic_model_KMeans')

# print(hdbscanModel.get_topic_info())
# print(kMeans.get_topic_info())


# 1. Embedding
sentence_model = SentenceTransformer('all-mpnet-base-v2')

#2. Dimensionality Reduction
umap_model = UMAP(metric='cosine', 
                  n_neighbors=5, 
                  min_dist=0.0)

# 3. Clustering 
hdbscan_model = HDBSCAN(min_cluster_size=60,
                        min_samples=1, 
                        metric='euclidean', 
                        cluster_selection_method='leaf', # TRY LEAF NEXT
                        prediction_data=True)
#hdbscan_model = KMeans(n_clusters=50)

# 4. Token
vec = CountVectorizer(stop_words='english',
                      min_df=10, 
                      max_df=0.90,
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

topic_model.reduce_topics(docs, nr_topics=100) # reduce topics for HDBSCAN

# Reduce Outliers
#newTopics = topic_model.reduce_outliers(docs, topics, strategy='c-tf-idf')

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



