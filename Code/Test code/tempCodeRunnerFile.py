#categorical
catTrain = catVec.fit(countTrain.toarray(), xTrain)
catPredict = catVec.predict(countTest.toarray())
catReport = classification_report(xTest, catPredict)
print(f'Categorical Report: \n {catReport}')