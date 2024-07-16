import pandas as pd
from bertopic import BERTopic


# Load and prepare data
data = pd.read_csv('eclipse_jdt.csv')

docs = data[['Title', 'Description']]
docs['Title'] = docs['Title'].fillna('')
docs['Description'] = docs['Description'].fillna('')
docs['Text'] = docs['Title'] + docs['Description']
docs_list = docs['Text'].tolist()

targetNames = data['Component'].tolist()  
classes = list(set(targetNames))  

topic_model = BERTopic(verbose=True).fit(docs, y=targetNames)

topics_per_class = topic_model.topics_per_class(docs, classes=classes)
fig_unsupervised = topic_model.visualize_topics_per_class(topics_per_class, top_n_topics=10)

fig_unsupervised.write_html('supervised.html')