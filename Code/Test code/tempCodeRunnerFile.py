#This file is for using the DistilBert classifier and topic modeling algorithm

import pandas as pd
import torch
import transformers
from transformers import DistilBertModel, DistilBertTokenizer
from torch.utils.data import Dataset, DataLoader

# The following code will use the GPU if the GPU is available
from torch import cuda
device = torch.device('cuda' if cuda.is_available() else 'cpu')
print(torch.cuda.is_available())