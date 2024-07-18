import gensim
from gensim import corpora
from gensim.models import LdaMulticore
from gensim.utils import simple_preprocess
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer

class ExtendedLDAModel:
    def __init__(self, num_topics=10):
        self.num_topics = num_topics
        self.lda_model = None
        self.dictionary = None
        self.feature_vectorizer = CountVectorizer()
        self.ps = PorterStemmer()
        self.stop_words = set(stopwords.words('english'))

    def preprocess_text(self, text):
        tokens = simple_preprocess(text)
        return [self.ps.stem(word) for word in tokens if word not in self.stop_words]

    def create_corpus(self, documents):
        preprocessed_docs = [self.preprocess_text(doc) for doc in documents]
        self.dictionary = corpora.Dictionary(preprocessed_docs)
        return [self.dictionary.doc2bow(doc) for doc in preprocessed_docs]

    def fit(self, bug_reports):
        # Extract text and features
        texts = [report['description'] + ' ' + report['summary'] for report in bug_reports]
        features = [[report['product'], report['component']] for report in bug_reports]

        # Create corpus and train LDA model
        corpus = self.create_corpus(texts)
        self.lda_model = LdaMulticore(corpus=corpus, id2word=self.dictionary, num_topics=self.num_topics)

        # Fit feature vectorizer
        self.feature_vectorizer.fit([' '.join(feature) for feature in features])

    def transform(self, bug_report):
        # Process text
        text = bug_report['description'] + ' ' + bug_report['summary']
        bow = self.dictionary.doc2bow(self.preprocess_text(text))
        topic_dist = self.lda_model.get_document_topics(bow)

        # Process features
        features = [bug_report['product'], bug_report['component']]
        feature_vec = self.feature_vectorizer.transform([' '.join(features)]).toarray()[0]

        # Combine topic distribution and feature vector
        topic_dist_dense = np.zeros(self.num_topics)
        for topic, prob in topic_dist:
            topic_dist_dense[topic] = prob

        return np.concatenate([topic_dist_dense, feature_vec])

class TopicMiner:
    def __init__(self, extended_lda_model):
        self.model = extended_lda_model
        self.developer_profiles = {}

    def update(self, bug_report, fixer):
        vector = self.model.transform(bug_report)
        if fixer not in self.developer_profiles:
            self.developer_profiles[fixer] = vector
        else:
            self.developer_profiles[fixer] = (self.developer_profiles[fixer] + vector) / 2

    def recommend(self, bug_report, top_k=5):
        vector = self.model.transform(bug_report)
        similarities = {dev: np.dot(vector, profile) for dev, profile in self.developer_profiles.items()}
        return sorted(similarities.items(), key=lambda x: x[1], reverse=True)[:top_k]

if __name__ == '__main__':
    # Usage example
    bug_reports = [
        {'description': 'App crashes on startup', 'summary': 'Startup crash', 'product': 'MobileApp', 'component': 'Startup'},
        {'description': 'Button color is wrong', 'summary': 'UI color issue', 'product': 'WebApp', 'component': 'UI'},
        # ... more bug reports ...
    ]

    model = ExtendedLDAModel(num_topics=20)
    model.fit(bug_reports)

    topic_miner = TopicMiner(model)

    # Update the model with some assigned bug reports
    topic_miner.update(bug_reports[0], 'developer1')
    topic_miner.update(bug_reports[1], 'developer2')

    # Recommend developers for a new bug report
    new_bug = {'description': 'App freezes when loading large file', 'summary': 'Performance issue', 'product': 'MobileApp', 'component': 'FileHandling'}
    recommendations = topic_miner.recommend(new_bug)
    print(recommendations)