import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from cleanData import text
import gensim
from gensim import corpora

# Parameters
minWords = 10
numTopics = 20
passes = 200

# Ensure text_list is tokenized and cleaned
text_list = text
if isinstance(text_list[0], str):
    text_list = [doc.split() for doc in text_list]

# Create dictionary and corpus
textDic = corpora.Dictionary(text_list)
corpus = [textDic.doc2bow(doc) for doc in text_list]

# Train LDA model
lda = gensim.models.LdaModel(corpus, num_topics=numTopics, id2word=textDic, passes=passes)

# Convert topics to fixed-size feature vectors
def get_topic_vector(lda_model, bow, num_topics):
    topics = lda_model.get_document_topics(bow, minimum_probability=0.0)
    return [prob for _, prob in sorted(topics, key=lambda x: x[0])]

X = np.array([get_topic_vector(lda, doc, numTopics) for doc in corpus])

# Create dummy labels (replace with actual labels if available)
y = np.array([i % numTopics for i in range(len(X))])

# Ensure all classes are present in the labels
unique_labels = np.unique(y)
if len(unique_labels) < numTopics:
    raise ValueError(f"Not all classes are present in the labels. Found {len(unique_labels)} unique classes.")

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Convert to PyTorch tensors
X_train = torch.FloatTensor(X_train)
X_test = torch.FloatTensor(X_test)
y_train = torch.LongTensor(y_train)
y_test = torch.LongTensor(y_test)

# Create DataLoader
train_dataset = TensorDataset(X_train, y_train)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

# Define the neural network model
class SimpleClassifier(nn.Module):
    def __init__(self, input_size, hidden_size1, hidden_size2, num_classes):
        super(SimpleClassifier, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size1)
        self.dropout1 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(hidden_size1, hidden_size2)
        self.dropout2 = nn.Dropout(0.5)
        self.fc3 = nn.Linear(hidden_size2, num_classes)
        
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.dropout1(x)
        x = torch.relu(self.fc2(x))
        x = self.dropout2(x)
        x = self.fc3(x)
        return x

# Initialize the model
model = SimpleClassifier(numTopics, 64, 32, numTopics)

# Define loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
num_epochs = 500
for epoch in range(num_epochs):
    model.train()
    for batch_X, batch_y in train_loader:
        optimizer.zero_grad()
        outputs = model(batch_X)
        loss = criterion(outputs, batch_y)
        loss.backward()
        optimizer.step()
    
    if (epoch + 1) % 10 == 0:
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

# Evaluate the model
model.eval()
with torch.no_grad():
    y_pred = model(X_test)
    _, predicted = torch.max(y_pred, 1)

print("Classification Report:")
print(classification_report(y_test.numpy(), predicted.numpy()))

# Calculate top-k accuracy
def top_k_accuracy(output, target, k=5):
    _, pred = output.topk(k, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))
    correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
    return correct_k.mul_(100.0 / target.size(0))

top1_accuracy = top_k_accuracy(y_pred, y_test, k=1)
top5_accuracy = top_k_accuracy(y_pred, y_test, k=5)

print(f'Top 1 Accuracy: {top1_accuracy.item():.2f}%')
print(f'Top 5 Accuracy: {top5_accuracy.item():.2f}%')

# Function to classify new text
def classify_text(new_text):
    model.eval()
    with torch.no_grad():
        # Preprocess the new text (tokenize and clean)
        new_text_processed = new_text.split()  # Replace with your preprocessing steps
        
        # Convert to bag-of-words
        new_bow = textDic.doc2bow(new_text_processed)
        
        # Get topic vector
        new_vector = get_topic_vector(lda, new_bow, numTopics)
        
        # Convert to PyTorch tensor
        new_vector_tensor = torch.FloatTensor([new_vector])
        
        # Make prediction
        output = model(new_vector_tensor)
        _, predicted_class = torch.max(output, 1)
        
        return predicted_class.item()

# Example usage
new_text = "This is a sample text to classify."
predicted_class = classify_text(new_text)
print(f"Predicted class for '{new_text}': {predicted_class}")