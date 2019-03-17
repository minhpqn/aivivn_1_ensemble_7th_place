"""Using ensemble method with sklearn for sentiment analysis task
"""
import numpy as np
import pandas as pd
import csv
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.ensemble import VotingClassifier
from argparse import ArgumentParser
from utils import load_csv_data
from sklearn import metrics


def train(train_sens, y_train, tune_sgd=False):
    C_OPTIONS = np.logspace(-9, 3, 15)
    text_clf = Pipeline([('vect', CountVectorizer(ngram_range=(1,3))),
                         ('tfidf', TfidfTransformer(sublinear_tf=False)),
                         ('clf', LinearSVC(dual=False, tol=1e-3, penalty="l2", loss='squared_hinge')),])
    parameters = {
        'clf__C': C_OPTIONS,
    }

    score = 'accuracy'

    print("\nTuning parameters for Linear SVM\n")
    gs_clf = GridSearchCV(text_clf, parameters, cv=5, scoring=score,
                          n_jobs=-1)
    gs_clf.fit(train_sens, y_train)
    for param_name in sorted(parameters.keys()):
        print("%s: %r" % (param_name, gs_clf.best_params_[param_name]))
    print("Best score (Grid search): %f" % gs_clf.best_score_)

    clf1 = LinearSVC(
        C=gs_clf.best_params_['clf__C'],
        loss='squared_hinge',
        penalty="l2",
        dual=False, tol=1e-3)

    if tune_sgd:

        print("\nTunning parameters for SGDClassifier\n")

        text_clf = Pipeline([('vect', CountVectorizer(ngram_range=(1,3))),
                             ('tfidf', TfidfTransformer(sublinear_tf=False)),
                             ('clf', SGDClassifier(alpha=0.0001, n_jobs=-1))])

        parameters = {
            'clf__n_iter': [50, 100, 200],
            'clf__l1_ratio': [0.01, 0.1, 0.15, 0.25, 0.3],
            'clf__loss': ['hinge', 'log', 'squared_hinge'],
            'clf__penalty': ['l1', 'l2', 'elasticnet'],
        }
        gs_clf = GridSearchCV(text_clf, parameters, cv=5, scoring=score,
                              n_jobs=-1)
        gs_clf.fit(train_sens, y_train)
        for param_name in sorted(parameters.keys()):
            print("%s: %r" % (param_name, gs_clf.best_params_[param_name]))
        print("Best score (Grid search): %f" % gs_clf.best_score_)

        clf2 = SGDClassifier(alpha=0.0001, n_iter=gs_clf.best_params_['clf__n_iter'],
                             l1_ratio=gs_clf.best_params_['clf__l1_ratio'],
                             loss=gs_clf.best_params_['clf__loss'],
                             penalty=gs_clf.best_params_['clf__penalty'])
    else:
        clf2 = SGDClassifier(alpha=0.0001, l1_ratio=0.1, loss='hinge', penalty="l2")

    print("\nEnsemble learning\n")

    clf3 = RandomForestClassifier(n_estimators=200, max_depth=None,
                                  random_state=100)

    ensemble_clf = VotingClassifier(
        estimators=[('linearSVM', clf1), ('sgd', clf2),
                    ('rdf', clf3),
                    ],
        voting='hard')

    text_clf = Pipeline([('vect', CountVectorizer(ngram_range=(1,3))),
                         ('tfidf', TfidfTransformer(sublinear_tf=False)),
                         ('clf', ensemble_clf),])

    text_clf.fit(train_sens, y_train)
    return text_clf


def predict(text_clf, test_sens):
    return text_clf.predict(test_sens)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("-eval", action="store_true", help="Evaluate on gold data (in development stage)")
    parser.add_argument("-tune_sgd", action="store_true", help="Tune parameters for SGDClassifier")
    parser.add_argument("train_file", help="Path to training data")
    parser.add_argument("test_file", help="Path to test data")
    parser.add_argument("output_file", help="Path to output")
    args = parser.parse_args()

    train_samples, train_labels = load_csv_data(args.train_file, textcol="text_ws")
    test_df = pd.read_csv(args.test_file)
    test_indices = [str(text) for text in test_df["id"]]
    test_samples = [str(text) for text in test_df["text_ws"]]
    test_labels = [str(intent) for intent in test_df["label"]]

    text_clf = train(train_samples, train_labels, tune_sgd=args.tune_sgd)
    preds = predict(text_clf, test_samples)
    output = []

    for id, label in zip(test_indices, preds):
        output.append([id, label])
    df = pd.DataFrame(data=output, columns=["id", "label"])
    df.to_csv(args.output_file, index=False, quoting=csv.QUOTE_NONE)

    if args.eval:
        acc = metrics.accuracy_score(test_labels, preds)
        print("Accuracy: {}".format(acc))
        print(metrics.classification_report(test_labels, preds))



