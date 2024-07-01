# Temporily testing gensim lda
import pandas as pd
import re
import string
from gensim import corpora
from gensim.models import LdaModel
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

lem = WordNetLemmatizer()

# Function for text cleaning
def cleanText(text):
    text = text.lower()                                                    # Convert to lowercase
    text = re.sub(r'\d+', ' ', text)                                       # Remove numbers
    text = re.sub(r'\b[nan]', ' ', text)                                   # Remove nan (Hard Coded)
    text = text.translate(str.maketrans(' ', ' ', string.punctuation))    # Remove punctuation
    return text

#function for lemmztizing
def wordLem(text):

    tokens = word_tokenize(text)
    lemToken = [lem.lemmatize(tokens) for tokens in tokens]

    return ' '.join(lemToken)

#opening file and removing duplicate reports
df = pd.read_csv('eclipse_jdt.csv')
dupl = df.dropna(subset=['Duplicated_issue'])
newData = df.drop(index=dupl.index)
text = newData['Description'].astype(str).apply(cleanText).apply(wordLem) #+ ' ' + newData['Component'].astype(str)

text = text.to_numpy()
textDic = corpora.Dictionary(text)

print(text[0:5])