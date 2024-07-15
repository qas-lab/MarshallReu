# # This is a example

# from bertopic import BERTopic
# from bertopic.vectorizers import ClassTfidfTransformer
# from bertopic.dimensionality import BaseDimensionalityReduction
# from sklearn.linear_model import LogisticRegression
# from sklearn.datasets import fetch_20newsgroups

# # Get labeled data
# data = fetch_20newsgroups(subset='all',  remove=('headers', 'footers', 'quotes'))
# docs = data['data']
# y = data['target']

# # Skip over dimensionality reduction, replace cluster model with classifier,
# # and reduce frequent words while we are at it.
# empty_dimensionality_model = BaseDimensionalityReduction()
# clf = LogisticRegression()
# ctfidf_model = ClassTfidfTransformer(reduce_frequent_words=True)

# # Create a fully supervised BERTopic instance
# topic_model= BERTopic(
#         umap_model=empty_dimensionality_model,
#         hdbscan_model=clf,
#         ctfidf_model=ctfidf_model
# )
# topics, probs = topic_model.fit_transform(docs, y=y)

# # Map input `y` to topics
# mappings = topic_model.topic_mapper_.get_mappings()
# mappings = {value: data["target_names"][key] for key, value in mappings.items()}

# # Assign original classes to our topics
# df = topic_model.get_topic_info()
# df["Class"] = df.Topic.map(mappings)

# print(df)

"""

This is my code version of a supervised bert model

"""
# from bertopic import BERTopic
# from bertopic.vectorizers import ClassTfidfTransformer
# from bertopic.dimensionality import BaseDimensionalityReduction
# from sklearn.linear_model import LogisticRegression
# import pandas as pd

# # Load and prepare data
# data = pd.read_csv('eclipse_jdt.csv')
# #data = pd.read_csv('mozilla_firefox.csv')

# docs = data[['Title', 'Description']]
# docs['Title'] = docs['Title'].fillna('')
# docs['Description'] = docs['Description'].fillna('')
# docs['Text'] = docs['Title'] + docs['Description']
# docs = docs['Text'].tolist()

# y = data['Component']

# emptyDim = BaseDimensionalityReduction()
# clf = LogisticRegression()
# ctfidf = ClassTfidfTransformer()

# topic_model = BERTopic(
#     umap_model=emptyDim,
#     hdbscan_model=clf,
#     ctfidf_model=ctfidf
# )

# topics, probs = topic_model.fit_transform(docs, y=y)

# mapping = topic_model.topic_mapper_.get_mappings()
# mappings = {value: data['Component'][key] for key, value in mapping.items()}

# data = topic_model.get_topic_info()
# data['Component'] = data.Topic.map(mapping)
# print(data)

'''

This is the corrected version of supervised bert topic 

'''

from bertopic import BERTopic
from bertopic.vectorizers import ClassTfidfTransformer
from bertopic.dimensionality import BaseDimensionalityReduction
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
import pandas as pd
from cleanData import text

# Load and prepare data
data = pd.read_csv('eclipse_jdt.csv')

docs = data[['Title', 'Description']]
docs['Title'] = docs['Title'].fillna('')
docs['Description'] = docs['Description'].fillna('')
docs['Text'] = docs['Title'] + docs['Description']
docs_list = docs['Text'].tolist()

# Encode the 'Component' column to integer labels
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(data['Component'])

# Define the models
emptyDim = BaseDimensionalityReduction()
clf = LogisticRegression()
ctfidf = ClassTfidfTransformer(reduce_frequent_words=True)

# Initialize BERTopic with the supervised model
topic_model = BERTopic(
    umap_model=emptyDim,
    hdbscan_model=clf,
    ctfidf_model=ctfidf
)

# Fit and transform the documents
topics, probs = topic_model.fit_transform(docs_list, y=y)

# Map the topic labels back to the original component names
mapping = {index: label for index, label in enumerate(label_encoder.classes_)}

# Get the topic information
topic_info = topic_model.get_topic_info()
topic_info['Component'] = topic_info.Topic.map(mapping)
print(topic_info)

prob = probs.mean()
print(prob)
