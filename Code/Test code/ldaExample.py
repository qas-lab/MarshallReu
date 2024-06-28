import pandas as pd
import torch
from transformers import LongformerTokenizer, LongformerForSequenceClassification
from torch.utils.data import Dataset, DataLoader

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class BugReportDataset(Dataset):
    def __init__(self, dataframe, tokenizer, max_len):
        self.len = len(dataframe)
        self.data = dataframe
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __getitem__(self, index):
        text = str(self.data.Description[index])
        text = ' '.join(text.split())

        inputs = self.tokenizer(
            text,
            max_length=self.max_len,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        input_ids = inputs['input_ids'].squeeze()
        attention_mask = inputs['attention_mask'].squeeze()
        targets = self.data.Component[index]

        return {
            'ids': input_ids,
            'mask': attention_mask,
            'targets': torch.tensor(targets, dtype=torch.long)
        }

    def __len__(self):
        return self.len

# Load your dataset
df = pd.read_csv('eclipse_jdt.csv')
df = df[['Description', 'Component']]

# Encode components
encode_dict = {label: idx for idx, label in enumerate(df['Component'].unique())}
df['Component'] = df['Component'].apply(lambda x: encode_dict[x])

# Split the dataset
train_size = 0.8
train_dataset = df.sample(frac=train_size, random_state=200)
test_dataset = df.drop(train_dataset.index).reset_index(drop=True)
train_dataset = train_dataset.reset_index(drop=True)

# Parameters
MAX_LEN = 1024  # Longformer can handle longer sequences
TRAIN_BATCH_SIZE = 4
VALID_BATCH_SIZE = 2
EPOCHS = 3
LEARNING_RATE = 2e-5

# Tokenizer and Model
tokenizer = LongformerTokenizer.from_pretrained('allenai/longformer-base-4096')
model = LongformerForSequenceClassification.from_pretrained('allenai/longformer-base-4096', num_labels=len(encode_dict))
model.to(device)

# Datasets and Dataloaders
training_set = BugReportDataset(train_dataset, tokenizer, MAX_LEN)
testing_set = BugReportDataset(test_dataset, tokenizer, MAX_LEN)

train_params = {
    'batch_size': TRAIN_BATCH_SIZE,
    'shuffle': True,
    'num_workers': 0
}

test_params = {
    'batch_size': VALID_BATCH_SIZE,
    'shuffle': False,
    'num_workers': 0
}

training_loader = DataLoader(training_set, **train_params)
testing_loader = DataLoader(testing_set, **test_params)

# Optimizer and Loss Function
optimizer = torch.optim.AdamW(params=model.parameters(), lr=LEARNING_RATE)
loss_function = torch.nn.CrossEntropyLoss()

def calculate_accuracy(preds, labels):
    _, pred_ids = torch.max(preds, dim=1)
    correct = (pred_ids == labels).sum().item()
    return correct / len(labels)

def train_epoch(model, dataloader, optimizer, loss_function, device):
    model.train()
    total_loss = 0
    total_accuracy = 0
    for data in dataloader:
        input_ids = data['ids'].to(device)
        attention_mask = data['mask'].to(device)
        targets = data['targets'].to(device)

        optimizer.zero_grad()
        outputs = model(input_ids, attention_mask=attention_mask).logits
        loss = loss_function(outputs, targets)
        total_loss += loss.item()
        accuracy = calculate_accuracy(outputs, targets)
        total_accuracy += accuracy

        loss.backward()
        optimizer.step()

    avg_loss = total_loss / len(dataloader)
    avg_accuracy = total_accuracy / len(dataloader)
    return avg_loss, avg_accuracy

def validate_epoch(model, dataloader, loss_function, device):
    model.eval()
    total_loss = 0
    total_accuracy = 0
    with torch.no_grad():
        for data in dataloader:
            input_ids = data['ids'].to(device)
            attention_mask = data['mask'].to(device)
            targets = data['targets'].to(device)

            outputs = model(input_ids, attention_mask=attention_mask).logits
            loss = loss_function(outputs, targets)
            total_loss += loss.item()
            accuracy = calculate_accuracy(outputs, targets)
            total_accuracy += accuracy

    avg_loss = total_loss / len(dataloader)
    avg_accuracy = total_accuracy / len(dataloader)
    return avg_loss, avg_accuracy

for epoch in range(EPOCHS):
    train_loss, train_accuracy = train_epoch(model, training_loader, optimizer, loss_function, device)
    val_loss, val_accuracy = validate_epoch(model, testing_loader, loss_function, device)
    print(f'Epoch {epoch + 1}/{EPOCHS}')
    print(f'Train Loss: {train_loss:.3f}, Train Accuracy: {train_accuracy:.3f}')
    print(f'Validation Loss: {val_loss:.3f}, Validation Accuracy: {val_accuracy:.3f}')

print('Training complete')
