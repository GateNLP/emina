#!/usr/bin/env python3
# one param: featurefile

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
from sklearn.externals import joblib

import sklearn.metrics

scoring = ['precision_micro', 'recall_micro', 'f1_micro']
scoring = {'prec_macro': 'precision_macro',
           'rec_micro': metrics.scorer.make_scorer(metrics.recall_score, average='micro')}

featurefilename = sys.argv[1]

d = np.loadtxt(featurefilename)
X = d[:, 1:]
y = d[:, 0]

clf = SVC(gamma=3, C=20, class_weight='balanced')


print('Fitting')
print(clf)
print('Features from', sys.argv[1])
clf.fit(X, y)
outfilename = featurefilename.replace('features', 'model') + '.svm.rbf'
print('Saving to', outfilename)
joblib.dump(clf, outfilename)