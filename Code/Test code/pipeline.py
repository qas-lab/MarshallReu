# pipeline

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report
from collections import Counter
from sklearn.naive_bayes import ComplementNB

# Example of reading the current CSV file
#current_df = pd.read_csv('topics.csv', header=None, skiprows=[0])  # Assuming no header in the original file

current_df = pd.read_csv('topics_2.csv', header=None, skiprows=[0])  # Assuming no header in the original file

# Initialize an empty list to store the restructured data
restructured_data = []

# Iterate over each row in the current DataFrame
for index, row in current_df.iterrows():
    topic = row[0]
    words = row[1:].tolist()
    for word in words:
        restructured_data.append((topic, word))

# Create a new DataFrame from the restructured data
new_df = pd.DataFrame(restructured_data, columns=['Label', 'Word'])

#print(new_df)
new_df.to_csv('restructured_topics.csv', index=False)

new_df.dropna(inplace=True)

labels = new_df['Label']
text = new_df['Word']

# counts = Counter(text)
# print(counts)

# Split the data into training and test sets
xTrain, xTrue, yTrain, yTest = train_test_split(labels, text, test_size=0.2, random_state=42, shuffle=True)

# print(f'xTrain : {xTrain.head()} yTrain : {yTrain.head()}')

# Create a pipeline that includes the TF-IDF vectorizer and the classifier
pipeline = Pipeline([
    #('tfidf', TfidfVectorizer()),
    ('count', CountVectorizer()),
    ('clf', MultinomialNB()),
])

# Train the classifier
pipeline.fit(xTrain, yTrain)

# Evaluate the classifier
y_pred = pipeline.predict(yTest)
print(f'Accuracy: {accuracy_score(xTrue, y_pred)}')
print(f'Classification Report: \n{classification_report(xTrue, y_pred, zero_division=0)}')

# Debugging tips
print(f'Predictions: {y_pred[:10]}')
print(f'Actual labels: {yTest[:10].values}')


