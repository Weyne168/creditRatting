# -*- coding: utf-8 -*-
import sys

sys.path.append('../')
import pickle
from datasets.datasets import load_data
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_curve
from sklearn.metrics import auc


"""
acc=0.7446

acc(0.3)=0.7013
acc(0.2)=0.7046
AUC=0.7397
fpr=0.6928
fnr=0.1285
"""

if __name__ == '__main__':
    train_X, train_y, _,_ = load_data('../data/train.npz')
    test_X, test_y, _,_ = load_data('../data/test.npz')
    print(train_X.shape)
    clf = RandomForestClassifier(criterion='gini', n_estimators=20, max_depth=5)
    clf.fit(train_X[:, :-1], train_y)


    p_label = clf.predict(test_X[:, :-1])
    fp, fn = 0, 0
    for i, p in enumerate(p_label):
        if p == 1 and test_y[i] == 0:
            fp += 1
        if p == 0 and test_y[i] == 1:
            fn += 1
    print(len(test_y[test_y == 0]))
    fp_rate = 1.0 * fp / (len(test_y[test_y == 0]))
    fn_rate = 1.0 * fn / (len(test_y[test_y == 1]))
    print(fp_rate, fn_rate)

    probs = clf.predict_proba(test_X[:, :-1])
    acc = clf.score(test_X[:, :-1], test_y)
    fpr, tpr, thresholds = roc_curve(test_y, probs[:, 1], pos_label=1, sample_weight=None, drop_intermediate=True)
    AUC = auc(fpr, tpr)
    print(acc, AUC)

    exit(12)
    with open('forest_model.pickle', 'wb') as file:
        save = {
            'model': clf,
            'acc': acc
        }
        pickle.dump(save, file)
