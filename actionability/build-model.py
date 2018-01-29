#!/usr/bin/env python3
# one param: featurefile

import argparse
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

from sklearn.exceptions import ConvergenceWarning
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=ConvergenceWarning)
warnings.filterwarnings("ignore", category=UserWarning)


parser = argparse.ArgumentParser()
parser.add_argument('--featurefilename', help='Path to the sklearn model file used', required=True)
parser.add_argument('--autosearch', help='Pick best classifier automatically', action='store_true')
parser.add_argument('--metric', help='Metric to use for model selection (f1, p, r, acc)', default='f1', const='f1', nargs='?')
opts = parser.parse_args()


scoring = ['precision_micro', 'recall_micro', 'f1_micro']
scoring = {'prec_macro': 'precision_macro',
           'rec_micro': metrics.scorer.make_scorer(metrics.recall_score, average='micro')}

print('Features from', opts.featurefilename)
d = np.loadtxt(opts.featurefilename)
X = d[:, 1:]
y = d[:, 0]


if opts.autosearch:
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
        SVC(gamma=3, C=20, class_weight='balanced'),
        #GaussianProcessClassifier(1.0 * RBF(1.0)),
        DecisionTreeClassifier(max_depth=5),
        RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1),
        MLPClassifier(alpha=1),
        AdaBoostClassifier(),
        GaussianNB(),
        QuadraticDiscriminantAnalysis()]

    best_name = None
    best_score = -1.0
    for name, clf in zip(names, classifiers):
        yhat = cross_val_predict(clf, X, y, cv=10)

        m = {}
        m['acc'] = metrics.accuracy_score(y, yhat)
        m['f1'] = metrics.f1_score(y, yhat, pos_label=1)
        m['r'] = metrics.recall_score(y, yhat, average='binary', pos_label=1)
        m['p'] = metrics.precision_score(y, yhat, average='binary', pos_label=1)
        print('%20s %3s %0.5f' % (name, opts.metric, m[opts.metric]))
        if m[opts.metric] > best_score:
        	best_clf = clf
        	best_score = m[opts.metric]

else:
    best_clf = SVC(gamma=3, C=20, class_weight='balanced')

print('Fitting', best_clf)
best_clf.fit(X, y)

outfilename = opts.featurefilename.replace('features', 'model') + '.classifier'
print('Saving to', outfilename)
joblib.dump(clf, outfilename)