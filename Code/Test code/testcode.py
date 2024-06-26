import pandas as pd
import re
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.tokenize import word_tokenize

#function for lemmztizing
def wordLem(text):

    tokens = word_tokenize(text)
    lemToken = [lem.lemmatize(tokens) for tokens in tokens]

    return ' '.join(lemToken)

#function for cleaning data
def cleanText(text):

    re.sub(r'\S([A-Za-z]*\s*)\d{1,2}:\d{2}\s*[APap][Mm]', '', text)
    text = re.sub(r'\d+', '', text)
    text = text.lower()
    text = text.strip()

    return text

#function for printing topics
def printTopics(model, featNames, topics):

    for topicIdx, topic in enumerate(model.components_):
        message = f'Topic #{topicIdx}'
        index = topic.argsort()[:-topics -1:-1]
        message += ' '.join([featNames[i] for i in index if i < len(featNames)])
        print(message)

    print()


#variables to use within code
stops = list(stopwords.words('english'))
numTopics = 6

#Creating instances of classes
lem = WordNetLemmatizer()
lda = LatentDirichletAllocation(n_components=numTopics, random_state=42)
vec = CountVectorizer(stop_words=stops)

#opening file and removing duplicate reports
df = pd.read_csv('eclipse_jdt.csv')
dupl = df.dropna(subset=['Duplicated_issue'])
newData = df.drop(index=dupl.index)
text = newData['Description'].astype(str).apply(cleanText).apply(wordLem)
trueLabel = newData['Component'].astype(str)

#splitting testing data
#xTrain, xTest, yTrain, yTest = train_test_split(trueLabel, text, shuffle=True, random_state=42, test_size=0.2)

train = vec.fit_transform(text)

model = lda.fit_transform(train)

names = list(vec.get_feature_names_out())

printTopics(lda, names, 10)
print()

