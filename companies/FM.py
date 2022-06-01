# -*- coding: utf-8 -*-
import sys

sys.path.append('../')
from datasets.datasets import load_data, load_data_us
import numpy as np
from sklearn.preprocessing import StandardScaler

"""
CH:
acc(testing 0.3)=0.6099551203590371
auc=--
fpr=0.32269257460074763
fnr=0.48613861386090484

US:
acc(testing 0.3)=0.6646586345381527
auc=--
fpr=0.24506749740369152
fnr=0.4990583804133728
"""


def ch2_data():
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

    train_f = open('train.txt', 'w')
    for i in range(train_X.shape[0]):
        line = str(int(train_y[i]))
        for j in range(train_X.shape[1]):
            if train_X[i, j] != 0:
                line += '\t' + str(j) + ':' + str(train_X[i, j])
        train_f.write(line + '\n')
    train_f.close()

    test_f = open('test.txt', 'w')
    for i in range(test_X.shape[0]):
        line = str(int(test_y[i]))
        for j in range(test_X.shape[1]):
            if test_X[i, j] != 0:
                line += '\t' + str(j) + ':' + str(test_X[i, j])
        test_f.write(line + '\n')
    test_f.close()


def ch(preds):
    ps = []
    with open(preds, 'r') as f:
        for line in f.readlines():
            if float(line.strip()) > 0.5:
                ps.append(1)
            else:
                ps.append(0)

    _, test_y, _, _ = load_data('../data/test.npz')

    fp, fn, pp, nn, r = 0, 0, 0, 0, 0
    for i, p in enumerate(ps):
        if p == 1 and test_y[i] == 0:
            fp += 1
        if p == 0 and test_y[i] == 1:
            fn += 1
        if p == test_y[i]:
            r += 1

        if p == 1:
            pp += 1
        else:
            nn += 1
    fp_rate = 1.0 * fp / (pp + 1e-9)
    fn_rate = 1.0 * fn / (nn + 1e-9)

    acc = r / (len(ps))
    print(acc, fp_rate, fn_rate)


def us2_data():
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

    train_f = open('us.train.txt', 'w')
    for i in range(train_X.shape[0]):
        line = str(int(train_y[i]))
        for j in range(train_X.shape[1]):
            if train_X[i, j] != 0:
                line += '\t' + str(j) + ':' + str(train_X[i, j])
        train_f.write(line + '\n')
    train_f.close()

    test_f = open('us.test.txt', 'w')
    for i in range(test_X.shape[0]):
        line = str(int(test_y[i]))
        for j in range(test_X.shape[1]):
            if test_X[i, j] != 0:
                line += '\t' + str(j) + ':' + str(test_X[i, j])
        test_f.write(line + '\n')
    test_f.close()


def us2(preds, gf):
    ps = []
    with open(preds, 'r') as f:
        for line in f.readlines():
            if float(line.strip()) > 0.5:
                ps.append(1)
            else:
                ps.append(0)
    test_y = []
    with open(gf, 'r') as f:
        for line in f.readlines():
            lab = line.split('\t')[0]
            lab = int(lab)
            test_y.append(lab)

    fp, fn, pp, nn, r = 0, 0, 0, 0, 0
    for i, p in enumerate(ps):
        if p == 1 and test_y[i] == 0:
            fp += 1
        if p == 0 and test_y[i] == 1:
            fn += 1
        if p == test_y[i]:
            r += 1

        if p == 1:
            pp += 1
        else:
            nn += 1
    fp_rate = 1.0 * fp / (pp + 1e-9)
    fn_rate = 1.0 * fn / (nn + 1e-9)

    acc = r / (len(ps))
    print(acc, fp_rate, fn_rate)


if __name__ == '__main__':
    #ch2_data()
    #us2_data()
    # ch('libfm-1.42.src/bin/res.txt')
    #us2('libfm-1.42.src/bin/res.txt', 'us.test.txt')
    us2('libfm-1.42.src/bin/ch.res.txt', 'test.txt')
