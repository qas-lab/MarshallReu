# Bert topic model

from bertopic import BERTopic
from bertopic.representation import KeyBERTInspired
from sentence_transformers import SentenceTransformer
from bertopic.vectorizers import ClassTfidfTransformer

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

from umap import UMAP
from hdbscan import HDBSCAN
import pandas as pd

# Load and prepare data
data = pd.read_csv('eclipse_jdt.csv')
#data = pd.read_csv('mozilla_firefox.csv')

docs = data[['Title', 'Description']]
docs['Title'] = docs['Title'].fillna('')
docs['Description'] = docs['Description'].fillna('')
docs['Text'] = docs['Title'] + docs['Description']
docs = docs['Text'].tolist()

# bugReports = data[['Issue_id', 'Component' ,'Title', 'Description']]
# print(bugReports)

# 1. Embedding
sentence_model = SentenceTransformer('all-mpnet-base-v2')

# 2. Dimensionality Reduction
umap_model = UMAP(metric='cosine', n_neighbors=20)

# 3. Clustering 
hdbscan_model = HDBSCAN(min_cluster_size=15, metric='euclidean', cluster_selection_method='eom', prediction_data=True)

# 4. Token
vec = CountVectorizer(stop_words='english', ngram_range=(1,3))

# 5. Weighting schemes
ctif = ClassTfidfTransformer()

# 6. Fine Tune
representation = KeyBERTInspired()    

# Create and fit BERTopic model
topic_model = BERTopic(representation_model=representation,
                       embedding_model=sentence_model,
                       umap_model=umap_model,
                       vectorizer_model=vec,
                       ctfidf_model=ctif,
                       hdbscan_model=hdbscan_model
                       )

topics, probs = topic_model.fit_transform(docs)

# Visualize topics
intertopic = topic_model.visualize_topics()
intertopic.write_html('intertopicBefore.html')

# Get topic and document information
document_info = topic_model.get_topic_info(4)
topic_info = topic_model.get_document_info(docs)
print(document_info)
print(topic_info)

# Calculate average probability
average_probability = probs.mean()
print(f'Average topic probability before reduction: {average_probability:.4f}')

topic_model.save('bert_topic_model')