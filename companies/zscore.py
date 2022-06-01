import sys

sys.path.append('../')
from datasets.datasets import load_data, load_data_us
import numpy as np

'''
CH:
acc=0.5487556099551204
fpr=0.316934720907931 
fnr=0.553084648493147


US:
acc=0.6987951807228916
fpr=0.29153846153823726
fnr=0.3659793814414125
'''


class Z_score_model_ch():
    """
    https://www.kancloud.cn/wizardforcel/python-quant-uqer/186240
    x1: 负债合计/资产总计                        (资产负债率)
    x2: 净利润/0.5*(资产总计 + 资产总计[上期])    (净资产收益率)
    x3: 营运资本/资产总计                        ((总营收-净利润)/净资产)
    x4: 未分配利润/资产总计                      (每股未分配利润/每股净资产)
    净资产=净利润/净资产收益率
    Z-score < 0.5: 已经违约
    0.5 < Z-score < 0.9: 有违约的可能性
    Z-score > 0.9: 财务健康，短期内不会出现违约情况
    """

    def __init__(self):
        self.coef = [0.517, -0.460, 18.640, 0.388, 1.158]
        self.thredhold = 0.9

        from datasets.preprocFinReports import format2Vec, dataset
        format2Vec('../data/test.dat')
        test_dat, ind_num = dataset('../data/test.dat.vec', span=5, stride=1)
        test_X = test_dat[:, ind_num:-1]
        print(test_X.shape)
        test_y = test_dat[:, -1]

        nf = test_X.shape[1] // 5
        X = []
        Y = []
        for i in range(test_X.shape[0]):
            for j in range(0, test_X.shape[0] - nf, nf):
                d = test_X[i, j:j + nf]
                if np.sum(d) != 0 and d[16] != 0 and d[4] != 0 and d[0] != 0 and d[-1] != 0 and d[2] != 0 and d[6] != 0:
                    X.append(d.reshape(1, -1))
                    Y.append(test_y[i])
                    break

        self.Y = np.array(Y)
        self.X = np.concatenate(X)

        print(len(self.Y[self.Y == 1]) / len(Y))
        # exit(1)
        print('test_num', self.X.shape[0])

        self.total_assets = self.X[:, 0] / (self.X[:, 16] + 1e-6)
        print(self.total_assets)

        self.x1 = self.X[:, -1]
        self.x2 = self.X[:, 0] / (self.total_assets + 1e-6)
        self.x3 = (self.X[:, 2] - self.X[:, 0]) / (self.total_assets + 1e-6)
        self.x4 = self.X[:, 6] / (self.X[:, 4] + 1e-6)

    def predict_scores(self):
        scores = self.coef[0] - self.coef[1] * self.x1 + self.coef[2] * self.x2 + self.coef[3] * self.x3 + self.coef[
            4] * self.x4
        return scores

    def predict_cls(self):
        scores = self.predict_scores()
        scores[scores > self.thredhold] = 1
        scores[scores <= self.thredhold] = 0
        return scores

    def acc(self, predicts):
        t = predicts + self.Y
        pn = t[t == 0]
        pp = t[t == 2]
        acc = (len(pn) + len(pp)) / len(predicts)
        print(acc)

    def type_errs(self, predicts):
        fp, fn, pp, nn = 0, 0, 0, 0
        for i, p in enumerate(predicts):
            if p == 1 and self.Y[i] == 0:
                fp += 1
            if p == 0 and self.Y[i] == 1:
                fn += 1
            if p == 1:
                pp += 1
            else:
                nn += 1

        fp_rate = 1.0 * fp / (pp + 1e-9)
        fn_rate = 1.0 * fn / (nn + 1e-9)
        print(fp_rate, fn_rate)
        return fp_rate, fn_rate

    def get_total_assets(self):
        """
        :return: 净利润/净资产收益率
        """
        pass


class Z_score_model_us():
    """
    x1: (流动资产-流动负债)/总资产  (f36-f45)/f42
    x2: 总利润/总资产 f3/f42
    x3: (总利润+利息支出)/总资产 (f3+f8)/f42
    x4: Book Value per Share每股股东权益/每股账面价值Shareholders Equity per Share f136/f138
    x5: 收入/资产总计 f0/f42

    Z-score < 0.5: 已经违约
    0.5 < Z-score < 0.9: 有违约的可能性
    Z-score > 0.9: 财务健康，短期内不会出现违约情况
    """

    def __init__(self, dat_file):
        self.coef = [1.2, 1.4, 3.3, 0.6, 0.999]
        self.thredhold = 1.81
        X, self.Y = load_data_us(dat_file)
        idx2corp = np.load('../data/idx2corp.us.npz', allow_pickle=True)['dict'][()]['idx2corp']
        nf = np.load('../data/cls_time_norm.us.npz', allow_pickle=True)['dict'][()]['cls_t_n'].shape[-1]
        print(idx2corp[int(X[2, 0])])

        self.X = X[:, 4 * nf + 2:5 * nf + 2]
        # print()
        # print(self.X[2, 0], self.X[2, 3], self.X[2, 8], self.X[2, 42], self.X[2, 45])
        self.x1 = (self.X[:, 36] - self.X[:, 45]) / (self.X[:, 42] + 1e-6)
        self.x2 = self.X[:, 3] / (self.X[:, 42] + 1e-6)
        self.x3 = (self.X[:, 3] + self.X[:, 8]) / (self.X[:, 42] + 1e-6)

        self.x4 = self.X[:, 136] / (self.X[:, 138] + 1e-6)
        self.x5 = self.X[:, 0] / (self.X[:, 42] + 1e-6)

    def predict_scores(self):
        scores = self.coef[0] * self.x1 + self.coef[1] * self.x2 + self.coef[2] * self.x3 + self.coef[
            3] * self.x4 + self.coef[4] * self.x5
        return scores

    def predict_cls(self):
        scores = self.predict_scores()
        # print(scores[2])
        scores[scores > self.thredhold] = 1
        scores[scores <= self.thredhold] = 0
        return scores

    def acc(self, predicts):
        t = predicts + self.Y
        pn = t[t == 0]
        pp = t[t == 2]
        acc = (len(pn) + len(pp)) / len(predicts)
        print(acc)

    def type_errs(self, predicts):
        fp, fn, pp, nn = 0, 0, 0, 0
        for i, p in enumerate(predicts):
            if p == 1 and self.Y[i] == 0:
                fp += 1
            if p == 0 and self.Y[i] == 1:
                fn += 1
            if p == 1:
                pp += 1
            else:
                nn += 1

        fp_rate = 1.0 * fp / (pp + 1e-9)
        fn_rate = 1.0 * fn / (nn + 1e-9)
        print(fp_rate, fn_rate)
        return fp_rate, fn_rate

    def get_total_assets(self):
        """
        :return: 净利润/净资产收益率
        """
        pass


class Z_score_model_us2():
    """
    x1: 总负债Total debt/总资产 f47/f42
    x2: 净利润/0.5*资产总计  f3/(0.5*f42)
    x3: 营运资本Operating Expenses/资产总计 f6/f42
    x4: Book Value per Share每股股东权益/每股账面价值Shareholders Equity per Share f136/f138

    Z-score < 0.5: 已经违约
    0.5 < Z-score < 0.9: 有违约的可能性
    Z-score > 0.9: 财务健康，短期内不会出现违约情况

    """

    def __init__(self, dat_file):
        self.coef = [0.517, -0.460, 18.640, 0.388, 1.158]
        self.thredhold = 0.9
        X, self.Y = load_data_us(dat_file)
        idx2corp = np.load('../data/idx2corp.us.npz', allow_pickle=True)['dict'][()]['idx2corp']
        nf = np.load('../data/cls_time_norm.us.npz', allow_pickle=True)['dict'][()]['cls_t_n'].shape[-1]
        print(idx2corp[int(X[2, 0])])

        self.X = X[:, 4 * nf + 2:5 * nf + 2]
        # print()
        # print(self.X[2, 0], self.X[2, 3], self.X[2, 8], self.X[2, 42], self.X[2, 45])
        self.x1 = self.X[:, 47] / (self.X[:, 42] + 1e-6)
        self.x2 = 2 * self.X[:, 3] / (self.X[:, 42] + 1e-6)
        self.x3 = self.X[:, 6] / (self.X[:, 42] + 1e-6)
        self.x4 = self.X[:, 136] / (self.X[:, 138] + 1e-6)

    def predict_scores(self):
        scores = self.coef[0] - self.coef[1] * self.x1 + self.coef[2] * self.x2 + self.coef[3] * self.x3 + self.coef[
            4] * self.x4
        return scores

    def predict_cls(self):
        scores = self.predict_scores()
        # print(scores[2])
        scores[scores > self.thredhold] = 1
        scores[scores <= self.thredhold] = 0
        return scores

    def acc(self, predicts):
        t = predicts + self.Y
        pn = t[t == 0]
        pp = t[t == 2]
        acc = (len(pn) + len(pp)) / len(predicts)
        print(acc)

    def type_errs(self, predicts):
        fp, fn, pp, nn = 0, 0, 0, 0
        for i, p in enumerate(predicts):
            if p == 1 and self.Y[i] == 0:
                fp += 1
            if p == 0 and self.Y[i] == 1:
                fn += 1
            if p == 1:
                pp += 1
            else:
                nn += 1

        fp_rate = 1.0 * fp / (pp + 1e-9)
        fn_rate = 1.0 * fn / (nn + 1e-9)
        print(fp_rate, fn_rate)
        return fp_rate, fn_rate

    def get_total_assets(self):
        """
        :return: 净利润/净资产收益率
        """
        pass


if __name__ == '__main__':
    zs = Z_score_model_ch()
    ss = zs.predict_cls()
    print(ss)
    zs.acc(ss)
    zs.type_errs(ss)
    exit(1)

    zs = Z_score_model_us2('../data/test.us.npz')
    ss = zs.predict_cls()
    print(ss)
    zs.acc(ss)
    zs.type_errs(ss)
