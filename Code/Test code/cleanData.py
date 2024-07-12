
import pandas as pd
import re
import string
import gensim
from gensim import corpora
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

# Load stop words
stop_words = set(stopwords.words('english'))
lem = WordNetLemmatizer()

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

def remove_urls(text):
    url_pattern = r'https?://(?:www\.)?\S+|www\.\S+'
    return re.sub(url_pattern, '', text)

# Function for text cleaning
def cleanText(text):
    text = split_camel_case(text)                                            # Space in Camel Case
    text = remove_urls(text)                                                 # Remove web addresses
    text = re.sub(r'\.', ' ', text)                                          # Remove period and replace with space
    text = text.lower()                                                      # Convert to lowercase
    text = remove_letters_before_date(text)                                  # Remove Initials
    text = re.sub(r'\d+', ' ', text)                                         # Remove numbers
    text = text.translate(str.maketrans('', '', string.punctuation))         # Remove punctuation
    text = re.sub(r'\b[a-z]\b', ' ', text)                                   # removing single characters
    text = re.sub(r'\s+', ' ', text).strip()                                 # Remove extra whitespace
    #text = re.sub(r'\b[nan]\b', ' ', text)                                   # Removing nan
    text = re.sub(r'\b(?:am|pm)\b', ' ', text)                               # Removing am and pm
    text = removeStopWords(text)                                             # Removing stop words
                                  
    return text

def cleanDevText(text):
    pattern = r'(.+?)@.*'
    return re.sub(pattern, r'\1', text)

data = pd.read_csv('eclipse_jdt.csv')
# data = pd.read_csv('mozilla_firefox.csv')

dupl = data.dropna(subset=['Duplicated_issue'])
newText = data.drop(index=dupl.index)
newData = newText[newText['Resolution'] == 'FIXED']
text = newData['Description'].astype(str).apply(cleanText).apply(wordLem) +  newData['Title'].astype(str).apply(cleanText).apply(wordLem)

# print(f'Text data \n {text}')  # Debugging 

# compNum = data['Component'].nunique()

# data = pd.read_csv('classifier_data_0.csv')

# text = data['description'].astype(str).apply(cleanText).apply(wordLem) + data['issue_title'].astype(str).apply(cleanText).apply(wordLem)

# devs = data['owner'].astype(str).apply(cleanDevText)
# devs = devs.value_counts()
# devs = devs[devs > 20]

#compNum = devs.nunique() 

# print(f'The size of the data set is {len(data)} \n {data}')
# print(f'The size of the text set is {len(text)} \n {text}')
# #print(f'The number of unique Component is {compNum}\n')

# print(f'The developer names is \n{devs}')
# #print(f'The total number of devs : {compNum}')
