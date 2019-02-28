import pandas

from sklearn import model_selection

from sklearn.linear_model import LogisticRegression

#from sklearn.tree import DecisionTreeClassifier

#from sklearn.neighbors import KNeighborsClassifier

#from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

#from sklearn.naive_bayes import GaussianNB

#from sklearn.svm import SVC
########################################################################

#two combinations for learning raw data
#1. complex algo simple data
#2. simple algo complex data

dataset = pandas.read_csv("iris.csv")

array = dataset.values


X = array [:,0:4]
Y = array[:,4]
#X: input
#X_train: 80%
#X_validtaion: 20%
#Y: output
#Y_train: 80%
#Y_validtaion: 20%

#DATA_DRIVEN APPROACH: 

#data Split
validation_size = 0.20 # give 20% data to test
seed = 7 

X_train, X_validation, Y_train, Y_validation = model_selection.train_test_split(X, Y, test_size = validation_size, random_state = seed)


clf = LogisticRegression().fit(X_train, Y_train)

print('IRIS dataset')
#printing traning accuracy
print('Accuracy of Logistic regression classifier on training set: {:.2f}'.format(clf.score(X_train, Y_train)))# score show result
#printing testing accuracy
print('Accuracy of Logistic regression classifier on test(validation) set: {:.2f}'.format(clf.score(X_validation, Y_validation)))#test score

r = clf.predict(X_validation[0:1, 0:4])#predict get the unknown answers
print('Predicted{}: Actual:{}'.format(r, Y_validation[0]))


#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import pandas
from sklearn import model_selection
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

#names = ['sepal-length','sepal-width','petal-length','petal-width','class']
#dataset = pandas.read_csv("data.csv",names=names)
#dataset = pandas.read_csv("data.csv")
#array = dataset.values

#X = array[:,0:4]
#Y = array[:,4]

#print(dataset.groupby('species').size())

#dataset.plot(kind='box',subplots=True,layout=(2,2),sharex=False,sharey=False)
#plt.show()
#plt.gray()

#print(dataset.shape)

#print(dataset.head(20))

#print(dataset.describe())

#dataset.hist()
#plt.show
#plt.gray()