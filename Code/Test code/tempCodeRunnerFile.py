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

# # bugReports = data[['Issue_id', 'Component' ,'Title', 'Description']]
# # print(bugReports)

# # 1. Embedding
# sentence_model = SentenceTransformer('all-mpnet-base-v2')
# # sentence_model = SentenceTransformer('Salesforce/SFR-Embedding-2_R')

# #2. Dimensionality Reduction
# umap_model = UMAP(metric='cosine', 
#                   n_neighbors=10, 
#                   min_dist=0.0)

# # 3. Clustering 
# # hdbscan_model = HDBSCAN(min_cluster_size=15, metric='euclidean', cluster_selection_method='eom', prediction_data=True)
# hdbscan_model = KMeans(n_clusters=50)

# # 4. Token
# vec = CountVectorizer(stop_words='english',
#                       min_df=10, 
#                       max_df=0.90,
#                       ngram_range=(1,2)) 

# # 5. Weighting schemes
# ctif = ClassTfidfTransformer(reduce_frequent_words=True)

# # 6. Fine Tune
# representation = KeyBERTInspired()    

# # Create and fit BERTopic model
# topic_model = BERTopic(embedding_model=sentence_model,          #1
#                        umap_model=umap_model,                   #2
#                        hdbscan_model=hdbscan_model,             #3
#                        vectorizer_model=vec,                    #4
#                        ctfidf_model=ctif,                       #5
#                        representation_model=representation,     #6
#                        )

# topics, probs = topic_model.fit_transform(docs)

# # Reduce Outliers
# #newTopics = topic_model.reduce_outliers(docs, topics, strategy='c-tf-idf')

# # Visualize topics
# intertopic = topic_model.visualize_topics()
# intertopic.write_html('intertopic.html')

# # Get topic and document information
# document_info = topic_model.get_topic_info(4)
# topic_info = topic_model.get_document_info(docs)
# print(document_info)
# print(topic_info)

# # Hierarchical topics
# linkage_function = lambda x: sch.linkage(x, 'single', optimal_ordering=True)
# hierarchical_topics = topic_model.hierarchical_topics(docs, linkage_function=linkage_function)
# hierarchical = topic_model.visualize_hierarchy(hierarchical_topics=hierarchical_topics)
# tree = topic_model.get_topic_tree(hierarchical_topics)
# hierarchical.write_html('hierarchcalTopics.html')

# # Saving topic model
# topic_model.save('bert_topic_model')

myModel = BERTopic.load('bert_topic_model')

info = myModel.get_topic_info()
print(info)

df = pd.read_csv('dataset.csv')
df = df.head(30)

print(df)