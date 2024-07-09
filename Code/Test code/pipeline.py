

#Pipeline

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB, GaussianNB
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier, StackingClassifier, VotingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, top_k_accuracy_score
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler, FunctionTransformer
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF

# # Load data
# current_df = pd.read_csv('lda_topics.csv', header=0)

# # Drop any rows with missing values
# current_df.dropna(inplace=True)

# restructured_data = []

# # Iterate over each row in the current DataFrame
# for index, row in current_df.iterrows():
#     topic = row['Topic']
#     words = row['Words'].split(', ')  # Split words by comma and space
#     for word in words:
#         restructured_data.append((topic, word))

# # Create a new DataFrame from the restructured data
# new_df = pd.DataFrame(restructured_data, columns=['Topic', 'Word'])

new_df = pd.read_csv('topics_words.csv')
new_df.drop(1)

# Define labels and text
labels = new_df['Topic']
text = new_df['Words']

# Split the data into training and test sets
xTrain, xTest, yTrain, yTest = train_test_split(text, labels, test_size=0.2, random_state=42, shuffle=True)

# Create individual pipelines for each classifier with SMOTE
nb_pipeline = ImbPipeline([
    ('tfidf', TfidfVectorizer()),
    ('smote', SMOTE(random_state=42)),  # Apply SMOTE
    ('clf', MultinomialNB()),
])

rf_pipeline = ImbPipeline([
    ('tfidf', TfidfVectorizer()),
    ('smote', SMOTE(random_state=42)),  # Apply SMOTE
    ('clf', RandomForestClassifier(random_state=42)),
])

mlp_pipeline = ImbPipeline([
    ('tfidf', TfidfVectorizer()),
    ('smote', SMOTE(random_state=42)),  # Apply SMOTE
    ('clf', MLPClassifier(random_state=42, max_iter=300)),
])

svm_pipeline = ImbPipeline([
    ('tfidf', TfidfVectorizer()),
    ('smote', SMOTE(random_state=42)),  # Apply SMOTE
    ('clf', SVC(kernel='linear', random_state=42, probability=True)),  # Using linear kernel for SVM and enabling probability estimates
])

# Define Gradient Boosting pipeline with SMOTE
gb_pipeline = ImbPipeline([
    ('tfidf', TfidfVectorizer()),
    ('smote', SMOTE(random_state=42)),  # Apply SMOTE
    ('clf', GradientBoostingClassifier(random_state=42)),
])

# Add additional pipelines
ada_pipeline = ImbPipeline([
    ('tfidf', TfidfVectorizer()),
    ('smote', SMOTE(random_state=42)),
    ('clf', AdaBoostClassifier(random_state=42)),
])

knn_pipeline = ImbPipeline([
    ('tfidf', TfidfVectorizer()),
    ('smote', SMOTE(random_state=42)),
    ('clf', KNeighborsClassifier()),
])

dt_pipeline = ImbPipeline([
    ('tfidf', TfidfVectorizer()),
    ('smote', SMOTE(random_state=42)),
    ('clf', DecisionTreeClassifier(random_state=42)),
])

qda_pipeline = ImbPipeline([
    ('tfidf', TfidfVectorizer()),
    ('smote', SMOTE(random_state=42)),
    ('to_dense', FunctionTransformer(lambda x: x.toarray(), accept_sparse=True)),
    ('clf', QuadraticDiscriminantAnalysis()),
])

gpc_pipeline = ImbPipeline([
    ('tfidf', TfidfVectorizer()),
    ('smote', SMOTE(random_state=42)),
    ('to_dense', FunctionTransformer(lambda x: x.toarray(), accept_sparse=True)),
    ('clf', GaussianProcessClassifier(kernel=RBF(), random_state=42)),
])

# Define the estimators for stacking
estimators = [
    ('rf', RandomForestClassifier(random_state=42)),
    ('svm', SVC(kernel='linear', random_state=42, probability=True)),
    ('mlp', MLPClassifier(random_state=42, max_iter=300))
]

stacking_pipeline = ImbPipeline([
    ('tfidf', TfidfVectorizer()),
    ('smote', SMOTE(random_state=42)),
    ('clf', StackingClassifier(estimators=estimators, final_estimator=GradientBoostingClassifier(random_state=42)))
])

# Define the voting classifier
voting_pipeline = ImbPipeline([
    ('tfidf', TfidfVectorizer()),
    ('smote', SMOTE(random_state=42)),
    ('clf', VotingClassifier(estimators=estimators, voting='soft'))
])

# Fit the models
nb_pipeline.fit(xTrain, yTrain)
rf_pipeline.fit(xTrain, yTrain)
mlp_pipeline.fit(xTrain, yTrain)
svm_pipeline.fit(xTrain, yTrain)
gb_pipeline.fit(xTrain, yTrain)
ada_pipeline.fit(xTrain, yTrain)
knn_pipeline.fit(xTrain, yTrain)
dt_pipeline.fit(xTrain, yTrain)
qda_pipeline.fit(xTrain, yTrain)
gpc_pipeline.fit(xTrain, yTrain)
stacking_pipeline.fit(xTrain, yTrain)
voting_pipeline.fit(xTrain, yTrain)

# Predict on the test set
nb_pred = nb_pipeline.predict(xTest)
rf_pred = rf_pipeline.predict(xTest)
mlp_pred = mlp_pipeline.predict(xTest)
svm_pred = svm_pipeline.predict(xTest)
gb_pred = gb_pipeline.predict(xTest)
ada_pred = ada_pipeline.predict(xTest)
knn_pred = knn_pipeline.predict(xTest)
dt_pred = dt_pipeline.predict(xTest)
qda_pred = qda_pipeline.predict(xTest)
gpc_pred = gpc_pipeline.predict(xTest)
stacking_pred = stacking_pipeline.predict(xTest)
voting_pred = voting_pipeline.predict(xTest)

# Predict probabilities for top_k accuracy score
nb_proba = nb_pipeline.predict_proba(xTest)
rf_proba = rf_pipeline.predict_proba(xTest)
mlp_proba = mlp_pipeline.predict_proba(xTest)
svm_proba = svm_pipeline.predict_proba(xTest)
gb_proba = gb_pipeline.predict_proba(xTest)
ada_proba = ada_pipeline.predict_proba(xTest)
knn_proba = knn_pipeline.predict_proba(xTest)
dt_proba = dt_pipeline.predict_proba(xTest)
qda_proba = qda_pipeline.predict_proba(xTest)
gpc_proba = gpc_pipeline.predict_proba(xTest)
stacking_proba = stacking_pipeline.predict_proba(xTest)
voting_proba = voting_pipeline.predict_proba(xTest)

# Print classification reports
print(f'Naive Bayes Report: \n{classification_report(yTest, nb_pred, zero_division=0)}')
print(f'Random Forest Report: \n{classification_report(yTest, rf_pred, zero_division=0)}')
print(f'MLP Report: \n{classification_report(yTest, mlp_pred, zero_division=0)}')
print(f'SVM Report: \n{classification_report(yTest, svm_pred, zero_division=0)}')
print(f'Gradient Boosting Report: \n{classification_report(yTest, gb_pred, zero_division=0)}')
print(f'AdaBoost Report: \n{classification_report(yTest, ada_pred, zero_division=0)}')
print(f'KNeighbors Report: \n{classification_report(yTest, knn_pred, zero_division=0)}')
print(f'Decision Tree Report: \n{classification_report(yTest, dt_pred, zero_division=0)}')
print(f'Quadratic Discriminant Analysis Report: \n{classification_report(yTest, qda_pred, zero_division=0)}')
print(f'Gaussian Process Report: \n{classification_report(yTest, gpc_pred, zero_division=0)}')
print(f'Stacking Classifier Report: \n{classification_report(yTest, stacking_pred, zero_division=0)}')
print(f'Voting Classifier Report: \n{classification_report(yTest, voting_pred, zero_division=0)}')

# Print top-1 and top-5 accuracy scores
print(f'Naive Bayes Top-1 Accuracy: {top_k_accuracy_score(yTest, nb_proba, k=1)}')
print(f'Naive Bayes Top-5 Accuracy: {top_k_accuracy_score(yTest, nb_proba, k=5)}')

print(f'Random Forest Top-1 Accuracy: {top_k_accuracy_score(yTest, rf_proba, k=1)}')
print(f'Random Forest Top-5 Accuracy: {top_k_accuracy_score(yTest, rf_proba, k=5)}')

print(f'MLP Top-1 Accuracy: {top_k_accuracy_score(yTest, mlp_proba, k=1)}')
print(f'MLP Top-5 Accuracy: {top_k_accuracy_score(yTest, mlp_proba, k=5)}')

print(f'SVM Top-1 Accuracy: {top_k_accuracy_score(yTest, svm_proba, k=1)}')
print(f'SVM Top-5 Accuracy: {top_k_accuracy_score(yTest, svm_proba, k=5)}')

print(f'Gradient Boosting Top-1 Accuracy: {top_k_accuracy_score(yTest, gb_proba, k=1)}')
print(f'Gradient Boosting Top-5 Accuracy: {top_k_accuracy_score(yTest, gb_proba, k=5)}')

print(f'AdaBoost Top-1 Accuracy: {top_k_accuracy_score(yTest, ada_proba, k=1)}')
print(f'AdaBoost Top-5 Accuracy: {top_k_accuracy_score(yTest, ada_proba, k=5)}')

print(f'KNeighbors Top-1 Accuracy: {top_k_accuracy_score(yTest, knn_proba, k=1)}')
print(f'KNeighbors Top-5 Accuracy: {top_k_accuracy_score(yTest, knn_proba, k=5)}')

print(f'Decision Tree Top-1 Accuracy: {top_k_accuracy_score(yTest, dt_proba, k=1)}')
print(f'Decision Tree Top-5 Accuracy: {top_k_accuracy_score(yTest, dt_proba, k=5)}')

print(f'Quadratic Discriminant Analysis Top-1 Accuracy: {top_k_accuracy_score(yTest, qda_proba, k=1)}')
print(f'Quadratic Discriminant Analysis Top-5 Accuracy: {top_k_accuracy_score(yTest, qda_proba, k=5)}')

print(f'Gaussian Process Top-1 Accuracy: {top_k_accuracy_score(yTest, gpc_proba, k=1)}')
print(f'Gaussian Process Top-5 Accuracy: {top_k_accuracy_score(yTest, gpc_proba, k=5)}')

print(f'Stacking Classifier Top-1 Accuracy: {top_k_accuracy_score(yTest, stacking_proba, k=1)}')
print(f'Stacking Classifier Top-5 Accuracy: {top_k_accuracy_score(yTest, stacking_proba, k=5)}')

print(f'Voting Classifier Top-1 Accuracy: {top_k_accuracy_score(yTest, voting_proba, k=1)}')
print(f'Voting Classifier Top-5 Accuracy: {top_k_accuracy_score(yTest, voting_proba, k=5)}')
