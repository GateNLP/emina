#!/usr/bin/env python3

import numpy as np
import sys

from sklearn import metrics
from sklearn import svm
from sklearn.model_selection import cross_val_score
from sklearn.svm import LinearSVC
from sklearn.model_selection import cross_validate
from sklearn.model_selection import cross_val_predict
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_moons, make_circles, make_classification
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis

import sklearn.metrics

scoring = ['precision_micro', 'recall_micro', 'f1_micro']
scoring = {'prec_macro': 'precision_macro',
           'rec_micro': metrics.scorer.make_scorer(metrics.recall_score, average='micro')}


d = np.loadtxt(sys.argv[1])
X = d[:, 1:]
y = d[:, 0]

#print ('y', y)

names = ["Nearest Neighbors", 
    "Linear SVM", 
    "RBF SVM", 
    #"Gaussian Process",
    "Decision Tree", 
    "Random Forest", 
    "Neural Net", 
    "AdaBoost",
    "Naive Bayes", 
    "QDA"]
classifiers = [
    KNeighborsClassifier(3),
    SVC(kernel="linear", C=0.025),
    SVC(gamma=2, C=1),
    #GaussianProcessClassifier(1.0 * RBF(1.0)),
    DecisionTreeClassifier(max_depth=5),
    RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1),
    MLPClassifier(alpha=1),
    AdaBoostClassifier(),
    GaussianNB(),
    QuadraticDiscriminantAnalysis()]

#    clf = svm.SVC(kernel='poly', probability=True, class_weight={0:1, 1:10}, degree=7)

#names = ['RBF SVM']
#classifiers = [SVC(gamma=3, C=20, class_weight='balanced')]
#classifiers = [SVC(gamma=1, C=20, class_weight='balanced')]

print('%.20s %6s %6s %6s' % (sys.argv[1], 'acc', 'f1', 'r')) 

for name, clf in zip(names, classifiers):
    yhat = cross_val_predict(clf, X, y, cv=5)
    acc = metrics.accuracy_score(y, yhat)
    f1 = metrics.f1_score(y, yhat, pos_label=1)
    r = metrics.recall_score(y, yhat, average='binary', pos_label=1)
    print('%20s %0.4f %0.4f %0.4f' % (name, acc, f1, r)) 

