
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
import pyLDAvis
import pyLDAvis.gensim_models as gensimvis
import time
import torch
import torch.nn.functional as F

# Load stop words
stop_words = set(stopwords.words('english'))

# Initialize the lemmatizer
lem = WordNetLemmatizer()

# Function for text cleaning
def cleanText(text):
    text = split_camel_case(text)                                            # Space in Camel Case
    text = re.sub(r'\.', ' ', text)                                          # Remove period and replace with space
    text = text.lower()                                                      # Convert to lowercase
    text = remove_letters_before_date(text)                                  # Remove Initials
    text = re.sub(r'\d+', ' ', text)                                         # Remove numbers
    text = text.translate(str.maketrans('', '', string.punctuation))         # Remove punctuation
    text = re.sub(r'\b[a-z]\b', ' ', text)                                   # removing single characters
    text = re.sub(r'\s+', ' ', text).strip()                                 # Remove extra whitespace
    text = re.sub(r'\b[nan]\b', ' ', text)                                   # Removing nan
    text = re.sub(r'\b(?:am|pm)\b', ' ', text)
    text = removeStopWords(text)                                             # Removing stop words
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

def remove_letters_before_date(text):
    # Define regex pattern for the date and time pattern
    date_pattern = r'.*?\(\d{1,2}/\d{1,2}/\d{2,4} \d{1,2}:\d{2}:\d{2} (?:AM|PM)\)'
    
    # Use re.sub to remove the letters before the date pattern
    result = re.sub(r'^[a-zA-Z\s]*', '', text, count=1)
    
    return result


if __name__ == "__main__":
    # Measure the start time
    start_time = time.time()

    # Opening file and removing duplicate reports
    newData = pd.read_csv('eclipse_jdt.csv')

    #print(newData['Description'])

    # Clean and lemmatize text
    text = newData['Description'].astype(str).apply(cleanText).apply(wordLem) +  newData['Title'].astype(str).apply(cleanText).apply(wordLem)

    #print(text)

    # Create a dictionary and corpus for LDA
    text_list = text.tolist()  # Convert to list of lists
    textDic = corpora.Dictionary(text_list)
    corpus = [textDic.doc2bow(doc) for doc in text_list]

    # Measure the time after preprocessing
    preprocessing_time = time.time()

    # Train the LDA model
    lda_start_time = time.time()

    topicNum = 10

    lda = LdaModel(corpus=corpus, 
               num_topics=topicNum, 
               id2word=textDic, 
               passes=10, 
               chunksize=2500, 
               random_state=42,
               alpha='auto',
               per_word_topics=True)
      
    # Measure the total time
    end_time = time.time()
    print(f"Total time: {end_time - start_time:.2f} seconds")

    # Example usage to print top 10 words for each topic
    num_preview_words = 10  # Number of words to print for each topic for preview
    num_save_words = 1000  # Number of words to save for each topic
    top_words_per_topic = []

    for idx, topic in lda.print_topics(num_topics=lda.num_topics, num_words=num_preview_words):
        print(f'Topic: {idx} \nWords: {topic}\n')

    for idx in range(topicNum):
        topic_terms = lda.get_topic_terms(idx, topn=num_save_words)
        top_words = [textDic[word_id] for word_id, prob in topic_terms]
        top_words_per_topic.append([f'Topic_{idx}', ', '.join(top_words)])

    # Save the topics to a CSV file with two columns (Topic and Words)
    topics_df = pd.DataFrame(top_words_per_topic, columns=['Topic', 'Words'])
    topics_df.to_csv('lda_topics.csv', index=False)
    print("LDA topics saved to lda_topics.csv")

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
