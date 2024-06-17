vec = CountVectorizer()
x = vec.fit_transform(dfTrain)
test = vec.get_feature_names_out()

print(x.toarray())