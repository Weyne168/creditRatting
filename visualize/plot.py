# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
from matplotlib.pyplot import MultipleLocator

ch_stock_y_acc = [0.8558, 0.9521, 0.9652, 0.9802, 0.8972, 0.8752]
ch_stock_y_auc = [0.9205, 0.9717, 0.9837, 0.9952, 0.9402, 0.9189]

us_stock_y_acc = [0.7073, 0.7186, 0.7259, 0.7489, 0.7723]
us_stock_y_auc = [0.7216, 0.7497, 0.7750, 0.7997, 0.8341]


def ch_stock():
    time = [2, 5, 10, 15, 20, 25]
    plt.plot(time, ch_stock_y_acc, marker='o', mec='r', mfc='w', label='Accuracy')
    plt.plot(time, ch_stock_y_auc, marker='*', ms=10, label='AUC')

    x_major_locator = MultipleLocator(5)
    y_major_locator = MultipleLocator(0.1)
    # plt.xticks(x, names, rotation=1)

    # 让图例生效
    # plt.xticks(range(0, 300, 50))
    ax = plt.gca()

    ax.xaxis.set_major_locator(x_major_locator)
    # 把x轴的主刻度设置为1的倍数
    ax.yaxis.set_major_locator(y_major_locator)
    # 把y轴的主刻度设置为10的倍数
    plt.xlim(0, 26)
    # 把x轴的刻度范围设置为-0.5到11，因为0.5不满一个刻度间隔，所以数字不会显示出来，但是能看到一点空白
    plt.ylim(0.7, 1.0)

    plt.margins(0)
    plt.subplots_adjust(bottom=0.10)
    plt.xlabel('Time spans', fontsize=18)  # X轴标签
    plt.ylabel("Accuracy/AUC", fontsize=18)  # Y轴标签
    plt.title('CH-Stock', fontsize=20)

    plt.legend(loc=3, fontsize=15)
    plt.grid()
    plt.savefig('ch_acc_auc.eps', dpi=900)
    plt.close()


def us_stock():
    time = [1, 2, 3, 4, 5]
    plt.plot(time, us_stock_y_acc, marker='o', mec='r', mfc='w', label='Accuracy')
    plt.plot(time, us_stock_y_auc, marker='*', ms=10, label='AUC')

    x_major_locator = MultipleLocator(1)
    y_major_locator = MultipleLocator(0.1)
    # plt.xticks(x, names, rotation=1)

    # 让图例生效
    # plt.xticks(range(0, 300, 50))
    ax = plt.gca()

    ax.xaxis.set_major_locator(x_major_locator)
    # 把x轴的主刻度设置为1的倍数
    ax.yaxis.set_major_locator(y_major_locator)
    # 把y轴的主刻度设置为10的倍数
    plt.xlim(0, 6)
    # 把x轴的刻度范围设置为-0.5到11，因为0.5不满一个刻度间隔，所以数字不会显示出来，但是能看到一点空白
    plt.ylim(0.5, 1.0)

    plt.margins(0)
    plt.subplots_adjust(bottom=0.10)
    plt.xlabel('Time spans', fontsize=18)  # X轴标签
    plt.ylabel("Accuracy/AUC", fontsize=18)  # Y轴标签
    plt.title('US-Stock', fontsize=20)
    plt.legend(loc=3, fontsize=15)
    plt.grid()
    plt.savefig('us_acc_auc.eps', dpi=900)
    plt.close()


def ch_stock_training(log_files, labels):
    colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k', 'w']
    for i in range(0, len(log_files)):
        X, Y = [], []
        with open(log_files[i], "r") as training_log:
            for line in training_log.readlines():
                x, y = line.strip().split(',')
                X.append(int(x))
                Y.append(float(y))
        plt.plot(X, Y, color=colors[i % len(colors)], label=labels[i])

    x_major_locator = MultipleLocator(5)
    y_major_locator = MultipleLocator(0.1)
    # plt.xticks(x, names, rotation=1)

    # 让图例生效
    # plt.xticks(range(0, 300, 50))
    ax = plt.gca()

    ax.xaxis.set_major_locator(x_major_locator)
    # 把x轴的主刻度设置为1的倍数
    ax.yaxis.set_major_locator(y_major_locator)
    # 把y轴的主刻度设置为10的倍数
    plt.xlim(0, 16)
    # 把x轴的刻度范围设置为-0.5到11，因为0.5不满一个刻度间隔，所以数字不会显示出来，但是能看到一点空白
    plt.ylim(0.5, 1.0)

    plt.margins(0)
    plt.subplots_adjust(bottom=0.10)
    plt.xlabel('Iterations')  # X轴标签
    plt.ylabel("Accuracy")  # Y轴标签
    plt.title('CH-Stock')

    plt.legend()
    plt.grid()
    plt.savefig('ch_iters.eps', dpi=900)
    plt.close()


def us_stock_training(log_files, labels):
    colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k', 'w']
    for i in range(0, len(log_files)):
        X, Y = [], []
        with open(log_files[i], "r") as training_log:
            for line in training_log.readlines():
                x, y = line.strip().split(',')
                X.append(int(x))
                Y.append(float(y))
        plt.plot(X, Y, color=colors[i % len(colors)], label=labels[i])

    x_major_locator = MultipleLocator(5)
    y_major_locator = MultipleLocator(0.1)
    # plt.xticks(x, names, rotation=1)

    # 让图例生效
    # plt.xticks(range(0, 300, 50))
    ax = plt.gca()

    ax.xaxis.set_major_locator(x_major_locator)
    # 把x轴的主刻度设置为1的倍数
    ax.yaxis.set_major_locator(y_major_locator)
    # 把y轴的主刻度设置为10的倍数
    plt.xlim(0, 16)
    # 把x轴的刻度范围设置为-0.5到11，因为0.5不满一个刻度间隔，所以数字不会显示出来，但是能看到一点空白
    plt.ylim(0.5, 1.0)

    plt.margins(0)
    plt.subplots_adjust(bottom=0.10)
    plt.xlabel('Iterations')  # X轴标签
    plt.ylabel("Accuracy")  # Y轴标签
    plt.title('US-Stock')

    plt.legend()
    plt.grid()
    plt.savefig('us_iters.eps', dpi=900)
    plt.close()


if __name__ == '__main__':
    ch_stock()
    us_stock()
