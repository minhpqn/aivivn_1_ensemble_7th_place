import os
import sys
sys.path.append("./")
import pandas as pd
import csv
import numpy as np
np.random.seed(1337)  # for reproducibility

import logging
from optparse import OptionParser
import matplotlib.pyplot as plt

from sklearn.model_selection import cross_val_score, cross_val_predict
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.linear_model import RidgeClassifier
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC
from sklearn.linear_model import SGDClassifier
from sklearn.linear_model import Perceptron
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.naive_bayes import BernoulliNB, MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neighbors import NearestCentroid
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn import metrics
from collections import Counter


def load_csv_data(filepath, textcol="text"):
    """Load data from csv file

    Parameters
    -----------
    filepath: String
        Path to CSV data

    Return
    -----------
    samples: List
        list of samples
    labels:  List
        list of labels
    """
    df = pd.read_csv(filepath)
    samples = [ str(text) for text in df[textcol] ]
    labels  = [ str(intent) for intent in df["label"] ]

    return samples, labels


def remove_less_than_n_samples(samples, labels, n_splits):
    # remove labels with less than n_splits samples
    counter = Counter(labels)
    less_then_n_samples = [lb for lb in counter.keys() if counter[lb] < n_splits]

    _samples = []
    _labels = []
    for exmp, lb in zip(samples, labels):
        if not lb in less_then_n_samples:
            _samples.append(exmp)
            _labels.append(lb)

    return _samples, _labels


def get_categories(samples, labels, categories):
    category_hsh = {c: 1 for c in categories}
    _samples = []
    _labels = []
    for i in range(len(samples)):
        if category_hsh.__contains__(labels[i]):
            _samples.append(samples[i])
            _labels.append(labels[i])
    return _samples, _labels


# Display progress logs on stdout
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(levelname)s %(message)s')

op = OptionParser('Usage: %prog [Options] input_file')
op.add_option("--report",
                            action="store_true",
                            help="Print a detailed classification report.")
op.add_option("--save_result",
                            action="store_true",
                            help="Save the results to a file")
op.add_option("--chi2_select",
                            action="store", type="int", dest="select_chi2",
                            help="Select some number of features using a chi-squared test")
op.add_option("--confusion_matrix",
                            action="store_true",
                            help="Print the confusion matrix.")
op.add_option("--save_plot",
                            action="store", type=str, dest="fig_name",
                            help="Save the plot to file")
op.add_option("--top_words",
                            action="store", dest="top_words",
                            help="Print ten most discriminative terms per class"
                                     " for every classifier.")
op.add_option("--use_hashing",
                            action="store_true",
                            help="Use a hashing vectorizer.")
op.add_option("--binary_feature",
                            action="store_true",
                            help="Use binary features")
op.add_option("--n_features",
                            action="store", type=int, default=2 ** 16,
                            help="n_features when using the hashing vectorizer.")

(opts, args) = op.parse_args()
print(__doc__)
op.print_help()
print()

if len(args) < 1:
    print('Input file is missing')
    sys.exit(1)

input_file = args[0]
print("Data file: %s" % input_file)
print()

ext = os.path.splitext(input_file)[1]

samples, labels = load_csv_data(input_file, textcol="text_ws")

m = {}
for i in range(len(samples)):
    m[samples[i]] = labels[i]

samples = sorted(m.keys())
labels  = [m[s] for s in samples]

samples, labels = remove_less_than_n_samples(samples, labels, 5)

target_names = np.unique(labels)
print("Target classes: %s" % target_names)
print()

categories = target_names

samples, labels = get_categories(samples, labels, categories)

print(Counter(labels))
print('Total samples:', len(samples))
print()

print("First sample in the data: ")
print(samples[0])
print()


###############################################################################
# Benchmark classifiers
def benchmark(clf):
        N = 5
        print('_' * 80)
        print("%d-fold cross validation" % N)
        print(clf)

        if opts.use_hashing:
            vectorizer = HashingVectorizer(
                                        # stop_words='english',
                                        non_negative=True,
                                        n_features=opts.n_features,
                                          )
        elif opts.binary_feature:
            vectorizer = CountVectorizer(
                                        # stop_words="english",
                                        binary=True,
                                        )
        else:
            vectorizer = TfidfVectorizer(
                                        sublinear_tf=True,
                                        max_df=0.5,
                                        # stop_words='english',
                                        )

        if opts.select_chi2:
            print("Extracting %d best features by a chi-squared test" %
                                opts.select_chi2)
            ch2 = SelectKBest(chi2, k=opts.select_chi2)
            pip = Pipeline([
                        ('vect', vectorizer),
                        ("feature_selection", ch2),
                        ('classification', clf)
                        ])
        else:
            pip = Pipeline([
                        ('vect', vectorizer),
                        ('classification', clf)
                        ])

        scores = cross_val_score(pip, samples, labels, cv=N,
                                 scoring = 'accuracy')

        print(scores)
        score = scores.mean()
        print("average accuracy: %0.3f" % scores.mean())

        print()

        y_pred = cross_val_predict(pip, samples, labels, cv=N)
        if opts.report:
            print("classification report:")
            print(metrics.classification_report(labels, y_pred,))

        clf_descr = str(clf).split('(')[0]
        return clf_descr, score, y_pred


results = []
for clf, name in (
                (RidgeClassifier(tol=1e-2, solver="sag", random_state=42), "Ridge Classifier"),
                (Perceptron(n_iter=500), "Perceptron"),
                (PassiveAggressiveClassifier(n_iter=50), "Passive-Aggressive"),
                (KNeighborsClassifier(n_neighbors=10), "kNN"),
                (RandomForestClassifier(n_estimators=100, max_depth=None,
                                                            random_state=100), "Random forest"),
                # (MLPClassifier(solver="adam"), "MLPClassifier"),
                ):
        print('=' * 80)
        print(name)
        results.append(benchmark(clf))


for penalty in ["l2", "l1"]:
        print('=' * 80)
        print("%s penalty" % penalty.upper())
        # Train Liblinear model
        results.append(benchmark(LinearSVC(loss='squared_hinge', penalty=penalty,
                                                dual=False, tol=1e-3)))
        # Train SGD model
        results.append(benchmark(SGDClassifier(alpha=.0001, n_iter=50, penalty=penalty)))

# Train SGD with Elastic Net penalty
print('=' * 80)
print("Elastic-Net penalty")
results.append(benchmark(SGDClassifier(alpha=.0001, n_iter=50,
                                       penalty="elasticnet")))

# Train NearestCentroid without threshold
print('=' * 80)
print("NearestCentroid (aka Rocchio classifier)")
results.append(benchmark(NearestCentroid()))

# Train sparse Naive Bayes classifiers
print('=' * 80)
print("Naive Bayes")
results.append(benchmark(MultinomialNB(alpha=.01)))
results.append(benchmark(BernoulliNB(alpha=.01)))

print('=' * 80)
print("LinearSVC with L1-based feature selection")
# The smaller C, the stronger the regularization.
# The more regularization, the more sparsity.
l1svc = LinearSVC(penalty="l1", dual=False, tol=1e-3)
results.append(benchmark(Pipeline([
    ('feature_selection', SelectFromModel(l1svc)),
    ('classification', LinearSVC())
])))

# make some plots

indices = np.arange(len(results))

results = [[x[i] for x in results] for i in range(3)]

clf_names, score, predictions = results
i = np.argmax(score)

print('Best classifier: %s, score: %0.3f' % (clf_names[i], score[i]))
best_y_pred = predictions[i]

if opts.confusion_matrix:
    print('Confusion matrix:')
    print(metrics.confusion_matrix(labels, best_y_pred))
    print()

if opts.report:
    print("classification report:")
    print(metrics.classification_report(labels, best_y_pred))

# Print output
if opts.save_result:
    filename = os.path.join('.', clf_names[i] + '-out.csv')
    is_correct = [ x==y for x,y in zip(labels,best_y_pred) ]

    df = pd.DataFrame(data={'intent':labels, 'predicted_intent':best_y_pred,
                            'text':samples, 'is_correct':is_correct},
                      columns=['intent','predicted_intent','text','is_correct'])
    df.to_csv(filename, quoting=csv.QUOTE_ALL, index=False)

fig = plt.figure(figsize=(16, 8))
plt.title("Score")
plt.barh(indices, score, .3, label="Score", color='navy')
plt.yticks(())
plt.subplots_adjust(left=.25)
plt.subplots_adjust(top=.95)
plt.subplots_adjust(bottom=.05)

for i, c in zip(indices, clf_names):
    plt.text(-0.2, i, c)

if opts.fig_name:
    print("Save plot to file %s" % opts.fig_name)
    fig.savefig(opts.fig_name)
else:
    plt.show()
