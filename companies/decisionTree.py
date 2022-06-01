# -*- coding: utf-8 -*-
import sys

# import graphviz

sys.path.append('../')
import pickle
import numpy as np
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn import tree
from sklearn.metrics import roc_curve
from sklearn.metrics import auc

"""
CHINA:
acc(testing 0.3)=0.7483
auc=0.7962
fpr=0.21623419826998255 
fnr=0.3048523206747839

acc=0.9364368255306813
auc=0.9693337479727326
fpr=0.05513816846030209
fnr=0.08457642725598137

US2:
acc(testing 0.3)=0.7342704149933066
auc=0.8035470258743673
fpr=0.2855939342878807 
fnr=0.18892508143260936
"""


def ch2():
    from datasets.preprocFinReports import format2Vec, dataset
    format2Vec('../data/train.dat')
    format2Vec('../data/test.dat')
    train_dat, ind_num = dataset('../data/train.dat.vec', span=5, stride=1)
    test_dat, ind_num = dataset('../data/test.dat.vec', span=5, stride=1)

    train_X = train_dat[:, :-1]
    print(train_X.shape)
    train_y = train_dat[:, -1]
    print(len(train_y[train_y == 1]) / len(train_y))
    scaler = StandardScaler()
    scaler.fit(train_X[:, ind_num:])
    train_X[:, ind_num:] = scaler.transform(train_X[:, ind_num:])

    test_X = test_dat[:, :-1]
    print(test_X.shape)
    test_y = test_dat[:, -1]
    print(len(test_y[test_y == 1]) / len(test_y))
    # exit(1)
    test_X[:, ind_num:] = scaler.transform(test_X[:, ind_num:])

    print('train_num', train_X.shape[0])
    print('test_num', test_X.shape[0])

    m = tree.DecisionTreeClassifier(criterion='gini', max_depth=6)
    clf = make_pipeline(StandardScaler(), m)

    clf.fit(train_X[:, :-1], train_y)

    probs = clf.predict_proba(test_X[:, :-1])

    acc = clf.score(test_X[:, :-1], test_y)
    fpr, tpr, thresholds = roc_curve(test_y, probs[:, 1], pos_label=1, sample_weight=None, drop_intermediate=True)
    AUC = auc(fpr, tpr)
    print(acc, AUC)

    p_label = clf.predict(test_X[:, :-1])
    fp, fn, pp, nn = 0, 0, 0, 0
    for i, p in enumerate(p_label):
        if p == 1 and test_y[i] == 0:
            fp += 1
        if p == 0 and test_y[i] == 1:
            fn += 1

        if p == 1:
            pp += 1
        else:
            nn += 1
    fp_rate = 1.0 * fp / (pp + 1e-9)
    fn_rate = 1.0 * fn / (nn + 1e-9)
    print(fp_rate, fn_rate)

    # exit(12)

    with open('LR_model.pickle', 'wb') as file:
        save = {
            'model': clf,
            'acc': acc
        }
        pickle.dump(save, file)


def us2():
    from datasets.preprocFinUS import format2Vec, dataset
    format2Vec('../data/train.us.dat')
    format2Vec('../data/test.us.dat')
    train_dat, ind_num = dataset('../data/train.us.dat.vec', span=5, stride=1)
    test_dat, ind_num = dataset('../data/test.us.dat.vec', span=5, stride=1)

    train_X = train_dat[:, :-1]
    print(train_X.shape)
    train_y = train_dat[:, -1]
    print(len(train_y[train_y == 1]) / len(train_y))
    scaler = StandardScaler()
    scaler.fit(train_X[:, ind_num:])
    train_X[:, ind_num:] = scaler.transform(train_X[:, ind_num:])

    test_X = test_dat[:, :-1]
    print(test_X.shape)
    test_y = test_dat[:, -1]
    print(len(test_y[test_y == 1]) / len(test_y))
    # exit(1)
    test_X[:, ind_num:] = scaler.transform(test_X[:, ind_num:])

    print('train_num', train_X.shape[0])
    print('test_num', test_X.shape[0])

    m = tree.DecisionTreeClassifier(criterion='gini', max_depth=5)
    clf = make_pipeline(StandardScaler(), m)

    clf.fit(train_X[:, :-1], train_y)
    probs = clf.predict_proba(test_X[:, :-1])

    acc = clf.score(test_X[:, :-1], test_y)
    fpr, tpr, thresholds = roc_curve(test_y, probs[:, 1], pos_label=1, sample_weight=None, drop_intermediate=True)
    AUC = auc(fpr, tpr)
    print(acc, AUC)

    p_label = clf.predict(test_X[:, :-1])
    fp, fn, pp, nn = 0, 0, 0, 0
    for i, p in enumerate(p_label):
        if p == 1 and test_y[i] == 0:
            fp += 1
        if p == 0 and test_y[i] == 1:
            fn += 1

        if p == 1:
            pp += 1
        else:
            nn += 1
    fp_rate = 1.0 * fp / (pp + 1e-9)
    fn_rate = 1.0 * fn / (nn + 1e-9)
    print(fp_rate, fn_rate)

    # dot_data = tree.export_graphviz(clf, out_file=None, filled=True, rounded=True, special_characters=True)
    # graph = graphviz.Source(dot_data)
    # graph.render("tree_decision_rule")


if __name__ == '__main__':
    print('CHINA:')
    ch2()
    print('US:')
    #us2()
