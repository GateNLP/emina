#!/usr/bin/env python3
# returns lines of a text file that classify as actionable
# usage: predict <model> <textfile>

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

from embprox import Embprox

modelfilename = sys.argv[1]
textfilename = sys.argv[2]


clf = joblib.load(modelfilename)

letter = modelfilename.replace('actionability.', '')[0]

featuregen = Embprox()
featuregen.load_keywords('keywords')
featuregen.load_embeddings('/home/leon/gate-extras/gate-twitter/docs/drift-is-a-thing/representations/socmed/2014.20M.tok.vectors.25.txt')

for line in open(textfilename):
    line = line.strip()
    t = featuregen.process_text(line)
    f = np.array(featuregen.wordlist2weights(t, letter))
    pred = clf.predict(f.reshape(1,-1))
    if pred > 0.5:
        print(line)