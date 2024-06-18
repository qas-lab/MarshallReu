import pandas as pd
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from scipy.sparse import hstack

#Print function for displaying topics 
def print_top_words(model, feature_names, n_top_words):
    for topic_idx, topic in enumerate(model.components_):
        message = f"Topic #{topic_idx}: "
        top_indices = topic.argsort()[:-n_top_words - 1:-1]
        message += " ".join([feature_names[i] for i in top_indices if i < len(feature_names)])
        print(message)
    print()


df = pd.read_csv("dataset.csv")


summary = df['Summary']
assignee = df['Assignee']


xTrainSum, xTestSum, yTrainAs, yTestAs = train_test_split(
    summary, assignee, test_size=0.2, shuffle=True, random_state=42
)


summaryVec = CountVectorizer(max_df=0.95, stop_words='english', ngram_range=(1,1))
assigneeVec = CountVectorizer(max_df=0.95, stop_words='english', ngram_range=(1,1))


xSumVec = summaryVec.fit_transform(xTrainSum)
ySumVec = summaryVec.transform(xTestSum)
xAsVec = assigneeVec.fit_transform(yTrainAs)
yAsVec = assigneeVec.transform(yTestAs)


trainSet = hstack([xAsVec, xSumVec])
testSet = hstack([yAsVec, ySumVec])


numTopics = 10


lda = LatentDirichletAllocation(n_components=numTopics, random_state=42)
lda.fit(trainSet)


feature_names = list(assigneeVec.get_feature_names_out()) + list(summaryVec.get_feature_names_out())


print_top_words(lda, feature_names, 10)
