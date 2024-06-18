import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from scipy.sparse import hstack

# Load your dataset
df = pd.read_csv("dataset.csv")

# Preview the dataset
print(df.head())

# Prepare the data
texts_summary = df['Summary']
texts_description = df['Assignee']
true_topics = df['Assignee']

# Split data into training and testing sets
X_train_summary, X_test_summary, X_train_description, X_test_description, y_train, y_test = train_test_split(
    texts_summary, texts_description, true_topics, test_size=0.2, random_state=42)

# Vectorize the text data
vec_summary = TfidfVectorizer(max_df=0.95, stop_words='english', ngram_range=(1, 1))
vec_description = TfidfVectorizer(max_df=0.95, stop_words='english', ngram_range=(1, 1))

X_train_summary_vec = vec_summary.fit_transform(X_train_summary)
X_test_summary_vec = vec_summary.transform(X_test_summary)

X_train_description_vec = vec_description.fit_transform(X_train_description)
X_test_description_vec = vec_description.transform(X_test_description)

# Combine the vectorized features
X_train_vec = hstack([X_train_summary_vec, X_train_description_vec])
X_test_vec = hstack([X_test_summary_vec, X_test_description_vec])

# Fit a Naive Bayes classifier
nb_classifier = MultinomialNB()
nb_classifier.fit(X_train_vec, y_train)

# Predict the topics for the test set
nb_predicted_topics = nb_classifier.predict(X_test_vec)

# Compute accuracy by comparing predicted topics with true topics
nb_accuracy = accuracy_score(y_test, nb_predicted_topics)
print(f'Naive Bayes Accuracy: {nb_accuracy:.2f}')

# Fit a Logistic Regression classifier
logistic_classifier = LogisticRegression(max_iter=1000)
logistic_classifier.fit(X_train_vec, y_train)

# Predict the topics for the test set
logistic_predicted_topics = logistic_classifier.predict(X_test_vec)

# Compute accuracy by comparing predicted topics with true topics
logistic_accuracy = accuracy_score(y_test, logistic_predicted_topics)
print(f'Logistic Regression Accuracy: {logistic_accuracy:.2f}')
