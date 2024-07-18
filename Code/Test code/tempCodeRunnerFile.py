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

docs = data[data['Assignee Real Name'] == 'Darin Wright']
# docs = docs[['Summary','Product','Component']]
# docs['Summary'] = docs['Summary'].fillna('')
# docs['Product'] = docs['Product'].fillna('')
# docs['Component'] = docs['Component'].fillna('')
# docs['Text'] = docs['Component'] + ' ' + docs['Product'] + ' ' + docs['Summary']
# docs = docs['Text'].tolist()

print(docs)