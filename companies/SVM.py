# -*- coding: utf-8 -*-
import sys

# import graphviz

sys.path.append('../')
import pickle
from datasets.datasets import load_data, load_data_us
from sklearn import svm
from sklearn.metrics import roc_curve
from sklearn.metrics import auc
import numpy as np
import time
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

np.random.randn()
"""
CH:
acc(testing 0.3)=0.6119951040391677
auc=0.6626303750256317
fpr=0.36477987421364527 
fnr=0.46961325966764345

acc=0.7926027433360125
auc=0.8458315830485688
fpr=0.22234468788632206
fnr=0.09275759652729121

US2:rbf
acc(testing 0.3)=0.7269076305220884
auc=0.7791105404634259
fpr=0.28790459965903925
fnr=0.2187499999993164
"""


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

    m = svm.NuSVC(probability=True, kernel='rbf', max_iter=3000, class_weight='balanced', tol=1e-4,
                  random_state=int(time.time()))

    clf = make_pipeline(StandardScaler(), m)

    # train_y[train_y[:] == 0] = -1
    # test_y[test_y[:] == 0] = -1

    clf.fit(train_X, train_y)

    # graph = graphviz.Source(dot_data)
    # graph.render("tree_decision_rule")
    acc = clf.score(test_X, test_y)
    probs = clf.predict_proba(test_X)
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
    print(pp, nn)
    print(fp_rate, fn_rate)


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

    m = svm.NuSVC(probability=True, kernel='rbf', max_iter=3000, class_weight='balanced', tol=1e-4, verbose=True,
                  cache_size=2000,
                  random_state=int(time.time()))

    clf = make_pipeline(StandardScaler(), m)

    # train_y[train_y[:] == 0] = -1
    # test_y[test_y[:] == 0] = -1

    clf.fit(train_X[:5000], train_y[:5000])

    # graph = graphviz.Source(dot_data)
    # graph.render("tree_decision_rule")
    acc = clf.score(test_X, test_y)
    probs = clf.predict_proba(test_X)
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
    print(pp, nn)
    print(fp_rate, fn_rate)


def ch():
    train_X, train_y, cols_name, n_corp_cls = load_data('../data/train.npz')
    test_X, test_y, cols_name, n_corp_cls = load_data('../data/test.npz')
    cls_t_n = np.load('../data/cls_time_norm.npz', allow_pickle=True)['dict'][()]
    cls_t_n = cls_t_n['cls_t_n']
    print(test_X.shape, train_X.shape, len(train_y[train_y == 1]))
    # exit(1)

    n_const_feats = len(cols_name)
    '''
    nf = 24 + 1
    f_index = 8
    for i in range(train_X.shape[0]):
        cls = int(train_X[i][n_const_feats - 1])
        t = 0
        for k in range(n_const_feats, train_X.shape[1], nf):
            cln = cls_t_n[cls, t] + 1e-9
            train_X[i][k:k + f_index] /= cln

    for i in range(test_X.shape[0]):
        cls = int(test_X[i][n_const_feats - 1])
        t = 0
        for k in range(n_const_feats, test_X.shape[1], nf):
            cln = cls_t_n[cls, t] + 1e-9
            test_X[i][k:k + f_index] /= cln
    '''
    train_X = train_X[:, n_const_feats - 1:-1]
    test_X = test_X[:, n_const_feats - 1:-1]

    # clf = svm.SVC(probability=True, kernel='rbf', max_iter=1000, tol=1e-4,decision_function_shape='ovr', verbose=True, shrinking=False,random_state=int(time.time()))

    m = svm.NuSVC(probability=True, kernel='rbf', max_iter=3000, class_weight='balanced', tol=1e-4,
                  random_state=int(time.time()))

    clf = make_pipeline(StandardScaler(), m)

    # train_y[train_y[:] == 0] = -1
    # test_y[test_y[:] == 0] = -1

    clf.fit(train_X, train_y)

    # graph = graphviz.Source(dot_data)
    # graph.render("tree_decision_rule")
    acc = clf.score(test_X, test_y)
    probs = clf.predict_proba(test_X)
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
    print(pp, nn)
    print(fp_rate, fn_rate)

    # exit(12)


if __name__ == '__main__':
    print('CHINA:')
    ch2()
    print('US:')
    # us2()
