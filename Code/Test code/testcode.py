# # Temporily testing gensim lda
import pandas as pd
import re
import string
import gensim
from gensim import corpora
from gensim.models import LdaModel
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

# Initialize the lemmatizer
lem = WordNetLemmatizer()

# Function for text cleaning
def cleanText(text):
    text = text.lower()                                                    # Convert to lowercase
    text = re.sub(r'\d+', ' ', text)                                       # Remove numbers
    text = re.sub(r'\b[nan]', ' ', text)                                   # Remove 'nan'
    text = text.translate(str.maketrans('', '', string.punctuation))       # Remove punctuation
    return text

# Function for lemmatizing
def wordLem(text):
    tokens = word_tokenize(text)
    lemToken = [lem.lemmatize(token) for token in tokens]
    return lemToken

# Opening file and removing duplicate reports
df = pd.read_csv('eclipse_jdt.csv')
dupl = df.dropna(subset=['Duplicated_issue'])
newData = df.drop(index=dupl.index)

# Clean and lemmatize text
text = newData['Description'].astype(str).apply(cleanText).apply(wordLem)

# Create a dictionary and corpus for LDA
text_list = text.tolist()  # Convert to list of lists
textDic = corpora.Dictionary(text_list)
corpus = [textDic.doc2bow(doc) for doc in text_list]

# Train the LDA model
#lda = LdaModel(corpus=corpus, num_topics=10, id2word=textDic)

lda = gensim.models.ldamodel.LdaModel(corpus=corpus,
                                      num_topics=10,
                                      random_state=100,
                                      update_every=1,
                                      chunksize=100,
                                      passes=5,
                                      alpha="auto")

# Display the dominant topic for each document
for idx, doc in enumerate(corpus):
    topics = lda[doc]
    dominant_topic = max(topics, key=lambda x: x[1])
    print(f"Document {idx}: Topic {dominant_topic[0]} with probability {dominant_topic[1]:.4f}")
