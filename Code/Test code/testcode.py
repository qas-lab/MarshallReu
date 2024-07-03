# Testing gensim 

import pandas as pd
import re
import string
import gensim
from gensim import corpora
from gensim.models import LdaModel
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from gensim.models import CoherenceModel
from gensim.corpora import Dictionary
import time
import pyLDAvis.gensim_models as gensimvis
import torch
import torch.nn.functional as F
import pyLDAvis

# Load stop words
stop_words = set(stopwords.words('english'))

# Initialize the lemmatizer
lem = WordNetLemmatizer()

# Function for text cleaning
def cleanText(text):
    text = split_camel_case(text)
    text = text.lower()                                                      # Convert to lowercase
    text = re.sub(r'\d+', ' ', text)                                         # Remove numbers
    text = removeStopWords(text)                                             # Removing stop words
    text = re.sub(r'\b(nan)\b', ' ', text)                                   # Remove 'nan'
    text = text.translate(str.maketrans('', '', string.punctuation))         # Remove punctuation
    text = re.sub(r'\b[a-z]\b', ' ', text)                                   # removing single characters
    text = re.sub(r'\s+', ' ', text).strip()                                 # Remove extra whitespace
    return text

# Function for lemmatizing
def wordLem(text):
    tokens = word_tokenize(text)
    lemToken = [lem.lemmatize(token) for token in tokens]
    return lemToken

def removeStopWords(text):
    text = text.split()
    text = [word for word in text if word not in stop_words]
    text = ' '.join(text)
    return text

def split_camel_case(text):
    # Use regex to insert a space before each capital letter not preceded by another capital letter
    words = re.sub(r'([a-z])([A-Z])', r'\1 \2', text)
    # Insert a space before a sequence of capital letters followed by a lowercase letter
    words = re.sub(r'([A-Z]+)([A-Z][a-z])', r'\1 \2', words)
    return words

if __name__ == "__main__":
    # Measure the start time
    start_time = time.time()

    # Opening file and removing duplicate reports
    newData = pd.read_csv('eclipse_jdt.csv')

    # Clean and lemmatize text
    text = newData['Description'].astype(str).apply(cleanText).apply(wordLem)

    # Create a dictionary and corpus for LDA
    text_list = text.tolist()  # Convert to list of lists
    textDic = corpora.Dictionary(text_list)
    corpus = [textDic.doc2bow(doc) for doc in text_list]

    # Measure the time after preprocessing
    preprocessing_time = time.time()

    # Train the LDA model
    lda_start_time = time.time()

    topicNum = 10

    lda = LdaModel(corpus=corpus, num_topics=topicNum, id2word=textDic, passes=20, chunksize=100, random_state=42)

    # Measure the total time
    end_time = time.time()
    print(f"Total time: {end_time - start_time:.2f} seconds")

    # Example usage to print top words for each topic
    num_topics = lda.num_topics
    top_words_per_topic = []

    for idx, topic in lda.print_topics(num_topics=num_topics, num_words=10):
        print(f'Topic: {idx} \nWords: {topic}\n')

    # Measure the time after training the LDA model
    lda_end_time = time.time()
    print(f"LDA training time: {lda_end_time - lda_start_time:.2f} seconds")

    # Perplexity
    output = torch.rand(1, topicNum)
    target = torch.randint(topicNum, (1,))
    loss = F.cross_entropy(output, target)
    perp = torch.exp(loss)
    print(f'\n Torch Perplexity: {perp}\n')

    # PyLDAvis visualization
    vis = gensimvis.prepare(lda, corpus, textDic)

    # Save the visualization to an HTML file
    pyLDAvis.save_html(vis, 'lda_visualization.html')
    print("LDA visualization saved to lda_visualization.html")

    # Calculate coherence score
    coherence_model_lda = CoherenceModel(model=lda, texts=text_list, dictionary=textDic, coherence='c_v')
    coherence_score = coherence_model_lda.get_coherence()
    print(f'\nCoherence Score: {coherence_score}\n')

    # Measure the time after calculating the coherence score
    coherence_end_time = time.time()
    print(f"Coherence calculation time: {coherence_end_time - lda_end_time:.2f} seconds")

