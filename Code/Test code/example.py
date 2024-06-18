import pandas as pd
import numpy as np
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import accuracy_score

df = pd.read_csv("dataset.csv")
print(df['Assignee'])


'''
vec = CountVectorizer( max_df=0.95, stop_words='english', ngram_range=(1,1))
x = vec.fit_transform(df)

n_topics = 10
lda = LatentDirichletAllocation(n_components=n_topics, max_iter=3)
lda.fit(x)

test = vec.get_feature_names_out()

# Print the top words for each topic
def print_top_words(model, feature_names, n_top_words):
    for topic_idx, topic in enumerate(model.components_):
        message = f"Topic #{topic_idx}: "
        message += " ".join([feature_names[i] for i in topic.argsort()[:-n_top_words - 1:-1]])
        print(message)
    print()

# Number of words to print per topic
n_top_words = 10

# Get feature names (words) from the vectorizer
feature_names = vec.get_feature_names_out()

# Print the topics found by LDA
print(f"Top {n_top_words} words per topic:\n")
print_top_words(lda, feature_names, n_top_words)

'''
