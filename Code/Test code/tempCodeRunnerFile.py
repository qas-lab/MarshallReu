# Trying nmf as a topic model

import numpy as np
from sklearn.decomposition import NMF
from sklearn.metrics import classification_report, accuracy_score
from cleanData import text, devs

num_Topics = devs.count()
print(num_Topics)