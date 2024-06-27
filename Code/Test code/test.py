#This file is for using the DistilBert classifier and topic modeling algorithm

import pandas as pd
import torch
import transformers
from transformers import DistilBertModel, DistilBertTokenizer
from torch.utils.data import Dataset, DataLoader

# The following code will use the GPU if the GPU is available
from torch import cuda
device = 'cuda' if cuda.is_available() else 'cpu'

# Class for triaging, setup from google colab review of how to use DistilBert
class Triage(Dataset):

    def __init__(self, dataframe, tokenizer, maxLen):
        
        self.len = len(dataframe)
        self.dataframe = dataframe
        self.tokenizer = tokenizer
        self.maxLen = maxLen
    
    def __getitem__(self, index):

        title = str(self.data.TITLE[index])
        title = ' '.join(title.split())
        inputs = self.tokenizer.encode_plus(
            title,
            None,
            add_special_tokens = True,
            maxLength = self.maxLen,
            padToMaxLength = True,
            returnTokenTypeId = True,
            truncation = True
        )
        ids = inputs['input_ids']
        mask = inputs['attention_mask']

        return {
            'ids': torch.tensor(ids, dtype=torch.long),
            'mask': torch.tensor(mask, dtype=torch.long),
            'targets': torch.tensor(self.data.ENCODE_CAT[index], dtype=torch.long)
        }
    
    def __len__(self):
        return self.len

# Class that creates a dense neural network and dropout
class DistillBertClass(torch.nn.Module):

    def __init__(self):
        super(DistillBertClass, self).__init__()
        self.l1 = DistilBertModel.from_pretrained('distilbert-base-cased')
        self.pre_classifier = torch.nn.Linear(768, 768)
        self.dropout = torch.nn.Dropout(0.3)
        self.classifier = torch.nn.Linear(768,4)

    def forward(self, inputs_ids, attention_mask):

        output_1 = self.l1(inputs_ids = inputs_ids, attention_mask = attention_mask)
        hidden_state = output_1[0]
        pooler = hidden_state[:, 0]
        pooler = self.pre_classifier(pooler)
        pooler = torch.nn.ReLU()(pooler)
        pooler = self.dropout(pooler)
        output = self.classifier(pooler)
        return output
    
# updating cat
def updateCat(x):
    return myDict[x]

# encoding cat
def encodeCat(x):
    if x not in encodeDict.keys():
        encodeDict[x] = len(encodeDict)
    return encodeDict[x]

# accuracy 
def calculate_acc(big_idx, targets):
    n_correct = (big_idx == targets).sum().item()
    return n_correct

# training model
def train(epoch):
    tr_loss = 0
    n_correct = 0
    nb_tr_step = 0
    nb_tr_example = 0

    for _,data in enumerate(training_loader, 0):
        ids    = data['ids'].to(device, dtype = torch.long)
        mask   = data['mask'].to(device, dtype = torch.long)
        target = data['target'].to(device, dtype = torch.long)

        output    = model(ids, mask)
        loss      = loss_function(output, target)
        tr_loss  += loss.item()
        big_val, big_idx = torch.max(output.data, dim=1) 
        n_correct += calculate_acc(big_idx, target)

        nb_tr_step += 1
        nb_tr_example += target.size(0)

        if _%5000 == 0:
            loss_step = tr_loss, nb_tr_step
            accu_step = (n_correct * 100) / nb_tr_example
            print(f'Training Loss per 5000 steps: {loss_step}')
            print(f'Training Accuracy per 5000 steps: {accu_step}')

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()    #only for GPU

    print(f'The total Accuracy for Epock {epoch} : {(n_correct*100)/nb_tr_example}')
    epoch_loss = tr_loss/nb_tr_step
    epoch_acc = (n_correct * 100) / nb_tr_example
    print(f'Training Loss Epoch: {epoch_loss}')
    print(f'Training Accuracy for Epoch: {epoch_acc}')

    return

# validate 
def valid(model, testing_loader):
    model.eval()
    n_correct = 0; n_wrong = 0; total = 0

    with torch.no_grad():
        for _, data in enumerate(testing_loader, 0):
            ids = data['ids'].to(device, dtype = torch.long)
            mask = data['mask'].to(device, dtype = torch.long)
            targets = data['targets'].to(device, dtype = torch.long)
            outputs = model(ids, mask).squeeze()
            loss = loss_function(outputs, targets)
            tr_loss += loss.item()
            big_val, big_idx = torch.max(outputs.data, dim=1)
            n_correct += calculate_acc(big_idx, targets)

            nb_tr_steps += 1
            nb_tr_examples += targets.size(0)

            if _%5000 == 0:
                loss_step = tr_loss, nb_tr_steps
                accu_step = (n_correct * 100) / nb_tr_examples
                print(f'Validation Loss per 100 steps: {loss_step}')
                print(f'Validation Accuracy per 100 steps: {accu_step}')
        
    epoch_loss = tr_loss/nb_tr_steps
    epoch_acc = (n_correct * 100) / nb_tr_examples
    print(f'Training Loss Epoch: {epoch_loss}')
    print(f'Training Accuracy for Epoch: {epoch_acc}')

    return epoch_acc

#defining varibles to use 
MAX_LEN = 512
TRAIN_BRANCH_SIZE = 4
VALID_BRANCH_SIZE = 2
EPOCHS = 1
LEARNING_RATE = 1e-05
tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-cased')

df = pd.read_csv('eclipse_jdt.csv')
df = df[['Description','Component']]

myDict = {
    'Debug' : 'Debug',
    'UI'    : 'UI',
    'Core'  : 'Core',
    'Text'  : 'Text',
    'Doc'   : 'Doc',
    'APT'   : 'Apt'
}

encodeDict = {}

df['Component'] = df['Component'].apply(lambda x: updateCat(x))
df['Description'] = df['Description'].apply(lambda x: encodeCat(x))

train_size = 0.8
train_dataset = df.sample(frac=train_size, random_state=200)
test_dataset = df.drop(train_dataset.index).reset_index(drop=True)
train_dataset = train_dataset.reset_index(drop=True)

print('Full Dataset: {}'.format(df.shape))
print('Train Dataset: {}'.format(train_dataset.shape))
print('Test Dataset: {}'.format(test_dataset.shape))
print()

training_set = Triage(train_dataset, tokenizer, MAX_LEN)
testing_set = Triage(test_dataset, tokenizer, MAX_LEN)

train_params = {
    'batch_size' : TRAIN_BRANCH_SIZE,
    'shuffle'    : True,
    'num_workers' : 0
}

test_params = {
    'batch_size' : VALID_BRANCH_SIZE,
    'shuffle'    : True,
    'num_workers' : 0
}

# Data Loader class
training_loader = DataLoader(training_set, **train_params)
testing_loader = DataLoader(testing_set, **test_params)

# Distil Bert class
model = DistillBertClass()
model.to(device)

# loss function for optimization
loss_function = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(params = model.parameters(), lr = LEARNING_RATE)

#training 
for epochs in range(EPOCHS):
    train(epochs)

#valadation
acc = valid(model, testing_loader)
print('This is the valadation section to print the accuracy of the model')
print('Accuracy on the test data = %0.2f%%', acc)