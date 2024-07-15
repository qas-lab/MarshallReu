# Bert topic model

from bertopic import BERTopic
from bertopic.representation import KeyBERTInspired
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import CountVectorizer
from bertopic.vectorizers import ClassTfidfTransformer
from cleanData import text
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

bugReports = data[['Issue_id', 'Component' ,'Title', 'Description']]

print(bugReports)

print(f'Data transformed to list \n')