import pandas as pd
from matplotlib import pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics

df = pd.read_csv("humansamples.csv")
testsample = pd.read_csv("sample2.csv")

df.drop(['Sample'], axis=1, inplace=True)

Y = df['Blood Detected'].values
Y=Y.astype('int')

X = df.drop(labels=['Blood Detected'], axis = 1)

#Change the test_size to change the amount of samples the model is tested on.
#Decrease test_size to increase the training size, and vice versa.
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2)
#Change n_estimators to change how many trees are in the random forest.
model = RandomForestClassifier(n_estimators = 20)
model.fit(X_train, Y_train)
#Change this from X_test to testsample to test sample 5 (false alarm) or any other specific samples.
prediction_test = model.predict(X_test)
#The below print statement gives the output of the random forest, how it classified the test data.
print(prediction_test)
#The below print statement will give the accuracy (correct classifications/total classifications)
#when testing the X_test data.
print("Accuracy = ", metrics.accuracy_score(Y_test, prediction_test))