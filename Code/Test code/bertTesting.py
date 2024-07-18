# This file is for testing loading and testing BERTopic models

from bertopic import BERTopic
from bertopic.representation import KeyBERTInspired
from sentence_transformers import SentenceTransformer
from bertopic.vectorizers import ClassTfidfTransformer

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import KFold

import numpy as np
from umap import UMAP
from hdbscan import HDBSCAN
import pandas as pd

from gensim.corpora.dictionary import Dictionary
from gensim.models.coherencemodel import CoherenceModel

martin = BERTopic.load('Martin_model')
darwin = BERTopic.load('Darwin_model')
curtis = BERTopic.load('Curtis_model_1')

data = pd.read_csv('dataset.csv')

data = data.sample(frac=1, random_state=42).reset_index(drop=True)

docs = data[['Summary','Product','Component']]
docs['Summary'] = docs['Summary'].fillna('')
docs['Product'] = docs['Product'].fillna('')
docs['Component'] = docs['Component'].fillna('')
docs['Text'] = docs['Component'] + ' ' + docs['Product'] + ' ' + docs['Summary']
testing = docs['Text'].head(20)

testing = testing.tolist()
len = len(testing)

for i in range(len):
    print(testing[i])
    print()