# -*- coding: utf-8 -*-
import sys

sys.path.append('../')
import pickle
from datasets.datasets import load_data, load_data_us
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve
from sklearn.metrics import auc
import numpy as np
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


"""
CH:
acc(0.3)=0.6532027743778049
AUC=0.6624571882221889
fpr=0.3099199014169378
fnr=0.4190821256033586

acc=0.846441700815622
auc=0.9252892308550198
fpr=0.16782026369461261 
fnr=0.08837345860245975

US2:
acc(0.3)=0.7255689424364123
AUC=0.7813480800708804
fpr=0.26822429906516987
fnr=0.29009433962195735
"""


def ch():
    train_X, train_y, cols_name, n_corp_cls = load_data('../data/train.npz')
    test_X, test_y, cols_name, n_corp_cls = load_data('../data/test.npz')
    cls_t_n = np.load('../data/cls_time_norm.npz', allow_pickle=True)['dict'][()]
    cls_t_n = cls_t_n['cls_t_n']

    n_const_feats = len(cols_name)

    train_X = train_X[:, n_const_feats - 1:-1]
    test_X = test_X[:, n_const_feats - 1:-1]

    m = LogisticRegression(max_iter=10000, penalty='l2', solver='liblinear', tol=1e-4)
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

    m = LogisticRegression(max_iter=10000, penalty='l2', solver='liblinear', tol=1e-4)
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


def us():
    train_X, train_y = load_data_us('../data/train.us.npz')
    test_X, test_y = load_data_us('../data/test.us.npz')

    cls_t_n = np.load('../data/cls_time_norm.us.npz', allow_pickle=True)['dict'][()]
    cls_t_n = cls_t_n['cls_t_n']

    train_X = train_X[:, 2:]  # remove id
    test_X = test_X[:, 2:]
    '''
    
    nf = cls_t_n.shape[-1]
    print(train_X.shape, cls_t_n.shape)
    
    for i in range(train_X.shape[0]):
        t = 0
        for k in range(train_X.shape[1], nf):
            cls = int(train_X[i][k + nf - 3])
            cln = cls_t_n[cls, t] + 1e-9

            train_X[i][k:k + nf - 3] /= cln[nf - 3]
            if k + nf < train_X.shape[1]:
                train_X[i][k + nf - 2] /= cln[nf - 2]
            t += 1

    for i in range(test_X.shape[0]):
        t = 0
        for k in range(test_X.shape[1], nf):
            # print(test_X[i][k + nf - 3])
            cls = int(test_X[i][k + nf - 3])
            cln = cls_t_n[cls, t] + 1e-9
            # print(test_X[i][k:k + nf])

            test_X[i][k:k + nf - 3] /= cln[nf - 3]
            if k + nf < test_X.shape[1]:
                test_X[i][k + nf - 2] /= cln[nf - 2]
            t += 1
    '''
    m = LogisticRegression(max_iter=10000, penalty='l1', tol=1e-4, solver='liblinear')
    clf = make_pipeline(StandardScaler(), m)

    clf.fit(train_X, train_y)

    # graph = graphviz.Source(dot_data)
    # graph.render("tree_decision_rule")
    probs = clf.predict_proba(test_X)

    acc = clf.score(test_X, test_y)
    fpr, tpr, thresholds = roc_curve(test_y, probs[:, 1], pos_label=1, sample_weight=None, drop_intermediate=True)
    AUC = auc(fpr, tpr)
    print(acc, AUC)

    p_label = clf.predict(test_X)
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

    with open('gbdt.pickle', 'wb') as file:
        save = {
            'model': clf,
            'acc': acc,
            'auc': AUC
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

    m = LogisticRegression(max_iter=10000, penalty='l2', solver='liblinear', tol=1e-4)
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


if __name__ == '__main__':
    print('CHINA:')
    ch2()
    print('US:')
    #us2()
