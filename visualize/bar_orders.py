import numpy as np
import matplotlib.pyplot as plt

xdim = np.array([1, 2, 3, 4, 5])
ch_stock_y_acc = [0.8612, 0.9364, 0.9572, 0.9802, 0.9676]
ch_stock_y_auc = [0.9273, 0.9631, 0.9746, 0.9955, 0.9853]

us_stock_y_acc = [0.7273, 0.7486, 0.7653, 0.7723, 0.7724]
us_stock_y_auc = [0.7893, 0.8021, 0.8303, 0.8341, 0.8321]

str1 = ("1", "2", "3", "4", "5")


def acc_bar():
    plt.bar(xdim, height=ch_stock_y_acc, width=0.35, tick_label=str1, label="CH-Stock")
    plt.bar(xdim + 0.4, height=us_stock_y_acc, width=0.35, align="center", label="US-Stock")
    plt.ylim(0.6, 1)
    plt.legend(loc=0, fontsize=15)
    plt.savefig('order_acc.eps', dpi=900)
    plt.close()


def auc_bar():
    # str1 = ("3", "5", "10", "15", "20")
    plt.bar(xdim, height=ch_stock_y_auc, width=0.35, tick_label=str1, label="CH-Stock", color="red")
    plt.bar(xdim + 0.4, height=us_stock_y_auc, width=0.35, align="center", label="US-Stock", color="gray")
    plt.ylim(0.7, 1)
    plt.legend(loc=2, fontsize=15)
    plt.savefig('order_auc.eps', dpi=900)
    plt.close()


def count_ch_bar():
    xdim = range(1, 11)
    print(len(xdim))
    str1 = ("1", "2", "3", "4", "5", "8", "10", "12", "15", "20")
    num = [0, 0, 0, 27, 3084, 1928, 256, 892, 429, 1170]
    plt.bar(xdim, height=num, width=0.45, tick_label=str1, label="CH-Stock", color="red")

    plt.ylim(0, 4000)
    plt.xlabel('Years', fontsize=18)
    plt.ylabel("Number of companies", fontsize=18)
    # plt.legend(loc=1)
    plt.grid()
    plt.savefig('count_ch.eps', dpi=900)
    plt.close()


def count_us_bar():
    xdim = range(1, 6)
    print(len(xdim))
    str1 = ("1", "2", "3", "4", "5")
    num = [130, 469, 241, 414, 3725]
    plt.bar(xdim, height=num, width=0.35, tick_label=str1, label="CH-Stock", color="blue")

    plt.ylim(0, 4000)
    # plt.legend(loc=1)
    plt.grid()
    plt.xlabel('Years', fontsize=18)
    plt.ylabel("Number of companies", fontsize=18)
    plt.savefig('count_us.eps', dpi=900)
    plt.close()


if __name__ == '__main__':
    #count_ch_bar()
    #count_us_bar()
    acc_bar()
    auc_bar()
