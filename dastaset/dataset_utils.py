import os
import pickle
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.impute import SimpleImputer

imputer = SimpleImputer(strategy="median")
scaler = StandardScaler()


# scaler = MinMaxScaler()


def read_data(dataset):
    data_list = []
    for root, dirs, files in os.walk(dataset):
        for file in files:
            print(file)
            dat = pd.read_csv(os.path.join(root, file))
            data_list.append(dat)

    return data_list


def train_test_split(dataset, test=0.2, savePath='data'):
    dat_len = len(dataset)
    train_begin = int(test * dat_len)
    index = [i for i in range(dat_len)]
    np.random.shuffle(index)
    test_set_idx = index[0:train_begin]
    train_set_idx = index[train_begin:]

    test_set = []
    train_set = []
    for i in test_set_idx:
        test_set.append(dataset[i])
    for i in train_set_idx:
        train_set.append(dataset[i])

    print(
        'total number of samples is %d,\n the number of training samples is %d \n the number of test samples is %d' % (
            dat_len, len(train_set_idx), len(test_set_idx))
    )

    with open(os.path.join(savePath, 'test.dat'), 'wb') as f:
        pickle.dump(test_set, f)
    with open(os.path.join(savePath, 'train.dat'), 'wb') as f:
        pickle.dump(train_set, f)


def get_data_ch(data_list):
    dataset = []
    sectors = []
    Nlabel = []
    k = 0
    for sample in data_list:
        sample = pd.DataFrame(sample[sample['营业总收入(元)'] != 0])
        sample.dropna(subset=['营业总收入(元)'], inplace=True, )
        sector = sample['Sector'].values[1:]
        sectors.extend(list(sector))
        sector = np.expand_dims(sector, axis=1)
        # times = sample['time'].values
        labels = sample['营业总收入(元)']  # .values
        # labels = imputer.fit_transform(labels.reshape((len(labels),1)))

        labels_shift = labels.shift(1)
        labels = (labels - labels_shift) / labels_shift
        labels = labels[1:].values

        # print(labels)
        # labels_shift = labels[1:,0]
        # labels = (labels_shift - labels[:-1,0]) / labels[:-1,0]

        labels[labels <= 0] = 0
        labels[labels > 0] = 1

        Nlabel.extend(list(labels))
        labels = np.expand_dims(labels, axis=1)

        sample.drop(['Sector', 'time', '营业总收入(元)', '营业总收入同比增长率'], axis=1, inplace=True)
        for column in list(sample.columns[sample.isnull().sum() > 0]):
            sample[column].fillna(0, inplace=True)
        print(sample.shape)

        sample = scaler.fit_transform(sample.values)
        if sample.shape[1] != 22:
            k += 1
            print('cccc_%d' % k)
            continue
        # sample = scaler.fit_transform(sample)
        d = np.concatenate([sample[1:], sector, labels], axis=1)
        dataset.append(d)
    print('cccc_%d' % k)
    print('company sectors are %d, label num is %d' % (len(set(sectors)), len(set(Nlabel))))
    print('numberical features are %d' % (dataset[0].shape[1] - 2))
    print('company sectors', set(sectors))
    print('Rattings', set(Nlabel))
    return dataset


def get_data_us(data_list):
    dataset = []
    sectors = []
    Nlabel = []
    for sample in data_list:
        sector = sample['Sector'].values
        sectors.extend(list(sector))
        sector = np.expand_dims(sector, axis=1)
        # times = sample['time'].values
        labels = sample['Class'].values
        Nlabel.extend(list(labels))
        labels = np.expand_dims(labels, axis=1)

        sample.drop(['Sector', 'time', 'company'], axis=1, inplace=True)
        # sample = imputer.fit_transform(sample)
        for column in list(sample.columns[sample.isnull().sum() > 0]):
            sample[column].fillna(0, inplace=True)
        sample = scaler.fit_transform(sample.values)
        d = np.concatenate([sample, sector, labels], axis=1)
        dataset.append(d)
    print('company sectors are %d, label num is %d' % (len(set(sectors)), len(set(Nlabel))))
    print('numberical features are %d' % (dataset[0].shape[1] - 2))
    print('company sectors', set(sectors))
    print('Rattings', set(Nlabel))
    return dataset


def get_data_chr(data_list):
    dataset = []
    ctyps = []
    Nlabel = []
    for sample in data_list:
        sample = pd.DataFrame(sample[sample['ratting'] != 0])
        sample.dropna(subset=['ratting'], inplace=True)
        print(len(sample))
        if len(sample) < 2:
            print('less 2 samples')
            continue
        company_type = sample['company_type'].values
        # print(company_type)
        ctyps.extend(list(company_type))
        company_type = np.expand_dims(company_type, axis=1)
        # times = sample['time'].values
        labels = sample['ratting'].values[1:]  # the next time ratting is treated as the current time label
        Nlabel.extend(list(labels))
        labels = np.expand_dims(labels, axis=1)

        sample.drop(['company_type', 'time', 'ratting', '证券代码', '证券简称', '流动负债/销售'], axis=1, inplace=True)
        # print(list(sample))
        # print(len(list(sample)))
        # print(sample.values.shape)

        for column in list(sample.columns[sample.isnull().sum() > 0]):
            sample[column].fillna(0, inplace=True)

        # sample = imputer.fit_transform(sample.values)
        sample = scaler.fit_transform(sample.values)

        d = np.concatenate([sample[:-1], company_type[:-1], labels], axis=1)
        print(d.shape)
        dataset.append(d)
    print('company types are %d, label num is %d' % (len(set(ctyps)), len(set(Nlabel))))
    print('numberical features are %d' % (dataset[0].shape[1] - 2))
    print('company types', set(ctyps))
    print('Rattings', set(Nlabel))
    return dataset


if __name__ == '__main__':
    ch_dataset_path = 'C:\\workspace\\creditRating\\data\\CH'
    us_dataset_path = 'C:\\workspace\\creditRating\\data\\US'
    chr_dataset_path = 'C:\\workspace\\creditRating\\data\\dataCHR'
    savePath = 'C:\\workspace\\creditRating\\data'
    data = read_data(ch_dataset_path)
    dataset = get_data_ch(data)

    # data = read_data(us_dataset_path)
    # dataset = get_data_us(data)

    # data = read_data(chr_dataset_path)
    # dataset = get_data_chr(data)
    train_test_split(dataset, 0.2, savePath)
