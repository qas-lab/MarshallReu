# This file is for creating bert models for each dev

import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0' # Suppress the oneDNN custom operations message

from bertopic import BERTopic
from bertopic.representation import KeyBERTInspired
from sentence_transformers import SentenceTransformer
from bertopic.vectorizers import ClassTfidfTransformer

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import PCA, NMF
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split

from umap import UMAP
from hdbscan import HDBSCAN
import pandas as pd

from scipy.cluster import hierarchy as sch

# Load and prepare data
data = pd.read_csv('dataset.csv')

# Get developers with more than 100 documents
devNum = data['Assignee Real Name'].value_counts()
devNum = devNum[devNum > 100]

def devDocList(name, data):
   docs = data[data['Assignee Real Name'] == name]
   docs = docs[['Summary', 'Product', 'Component']]
   docs['Summary'] = docs['Summary'].fillna('')
   docs['Product'] = docs['Product'].fillna('')
   docs['Component'] = docs['Component'].fillna('')
   docs['Text'] = docs['Product'] + ' ' + docs['Component'] + ' ' + docs['Summary']
   return docs

# 1. Embedding
sentence_model = SentenceTransformer('all-mpnet-base-v2')

# 2. Dimensionality Reduction
#umap_model = PCA(n_components=1)
umap_model = UMAP(n_components=1,
                  n_neighbors=15,
                  min_dist=0.1,
                  metric='cosine')

# 3. Clustering 
hdbscan_model = HDBSCAN(min_cluster_size=20,
                        min_samples=1, 
                        metric='euclidean', 
                        cluster_selection_method='leaf',
                        prediction_data=True)

# 4. Token
vec = CountVectorizer(stop_words='english',
                      ngram_range=(1,2))

# 5. Weighting schemes
ctif = ClassTfidfTransformer(reduce_frequent_words=True)

# 6. Fine Tune
representation = KeyBERTInspired()


test_documents = []

count = 0

for dev in devNum.index:

   print(f"Training model for {dev}...")
   
   docs_df = devDocList(dev, data)
   
   docTrain_df, docTest_df = train_test_split(docs_df, test_size=0.2)
   

   docTest_df['Developer'] = dev
   test_documents.append(docTest_df)
   
  
   docTrain = docTrain_df['Text'].tolist()
   
   # Create and fit BERTopic model
   topic_model = BERTopic(embedding_model=sentence_model,          #1
                          umap_model=umap_model,                   #2
                          hdbscan_model=hdbscan_model,             #3
                          vectorizer_model=vec,                    #4
                          ctfidf_model=ctif,                       #5
                          representation_model=representation)     #6

   topics, probs = topic_model.fit_transform(docTrain)
   topicsTest, topicsProb = topic_model.transform(docTrain_df['Text'].tolist())
   
   # Print topic information
   print(topic_model.get_topic_info())
   print(topic_model.get_document_info(docTrain))
   print(topicsProb.mean())
   
   # Save the topic model
   topic_model.save(f'{count}_{dev}')

   count += 1

print("All models trained and saved.")

# Concatenate all test documents into a single DataFrame
test_documents_df = pd.concat(test_documents, ignore_index=True)

# Save the test documents to a CSV file
test_documents_df.to_csv('test_documents.csv', index=False)


