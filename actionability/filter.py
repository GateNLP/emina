#!/usr/bin/env python3
# returns lines of a text file that classify as actionable
# usage: predict <model> <textfile>

import argparse
import numpy as np
import os
import sys

from sklearn import metrics
from sklearn import svm
from sklearn.model_selection import cross_val_score
from sklearn.svm import LinearSVC
from sklearn.model_selection import cross_validate
from sklearn.model_selection import cross_val_predict
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
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

from embprox import Embprox, default_embeddings_file

parser = argparse.ArgumentParser()
parser.add_argument('--modelfilename', help='Path to the sklearn model file used', required=True)
parser.add_argument('--inputfilename', help='Path to the text file of instances to predict actionability of (one message per line)', required=True)
parser.add_argument('--embeddings', help='Embeddings file name (in text format)', default=default_embeddings_file, action='store')
parser.add_argument('--prefix', help='The filename prefix for reading from', default='actionability')
parser.add_argument('--keywords', help='Keywords file', default='keywords')
parser.add_argument('--threshold', help='Decision threshold', default=0.5, type=float)
opts = parser.parse_args()


clf = joblib.load(opts.modelfilename)

letter = opts.modelfilename.split('/')[-1].replace(opts.prefix+'.', '')[0]

featuregen = Embprox()
featuregen.load_keywords(opts.keywords)
if not os.path.isfile(opts.embeddings) and opts.embeddings == default_embeddings_file:
    featuregen.download_embeddings()
featuregen.load_embeddings(opts.embeddings)

for line in open(opts.inputfilename):
    line = line.strip()
    t = featuregen.process_text(line)
    f = np.array(featuregen.wordlist2weights(t, letter))
    pred = clf.predict(f.reshape(1,-1))
    if pred > opts.threshold:
        print(line)