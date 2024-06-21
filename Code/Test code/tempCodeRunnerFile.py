import nltk
#nltk.download('wordnet')
#nltk.download('punkt')
from nltk.stem.wordnet import WordNetLemmatizer     #lemmentizing words
from nltk.tokenize import word_tokenize             #tokenize words before applying lemmatization
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Print the top words for each topic (Fix the for loops to what I write)
def print_top_words(model, feature_names, n_top_words):
    for topic_idx, topic in enumerate(model.components_):
        message = f"Topic #{topic_idx}: "
        message += " ".join([feature_names[i] for i in topic.argsort()[:-n_top_words - 1:-1]])
        print(message)
    print()

#function for lemmatization (Fix the for loops to what I write)
def lemmatizeText(text):
    tokens = word_tokenize(text)
    lemmatized_tokens = [lemmatizer.lemmatize(token) for token in tokens]
    return ' '.join(lemmatized_tokens)

df = pd.read_csv("eclipse_jdt.csv")
df['combine'] = df['Description'] + '' + df['Title'] + '' + df['Component']

# Lemmatize all words in documents.
lemmatizer = WordNetLemmatizer()
df['lemmatizeText'] = df['combine'].apply(lemmatizeText)

summary = df['lemmatizeText'].values.astype('U')

print(summary)