import os
import pandas as pd
from sklearn.preprocessing import LabelEncoder


def read_cat_data(dataset, savePath):
    data_list = []
    for root, dirs, files in os.walk(dataset):
        for file in files:
            t = file.split('_')[0]
            print(t)
            dat = pd.read_csv(os.path.join(root, file))
            dat['time'] = t
            data_list.append(dat)
    data = pd.concat(data_list, axis=0, ignore_index=True)

    sector = data['Sector']
    labelEncoder = LabelEncoder()
    sector_encode = labelEncoder.fit_transform(sector)
    data['Sector'] = sector_encode
    data.to_csv(os.path.join(savePath, 'dat_all.csv'))


def save_company(dataset, savePath):
    data = pd.read_csv(dataset)
    corps = data['company'].values
    corps = set(corps.tolist())

    colnames = list(data)
    colnames.remove('2015 PRICE VAR [%]')
    colnames.remove('2016 PRICE VAR [%]')
    colnames.remove('2017 PRICE VAR [%]')
    colnames.remove('2018 PRICE VAR [%]')
    colnames.remove('2019 PRICE VAR [%]')

    price_vars = {2014: '2015 PRICE VAR [%]', 2015: '2016 PRICE VAR [%]', 2016: '2017 PRICE VAR [%]',
                  2017: '2018 PRICE VAR [%]', 2018: '2019 PRICE VAR [%]'}

    for company in corps:
        comInfos = data.loc[data['company'] == company]
        years = comInfos['time'].values
        years = sorted(set(years.tolist()))
        dfNew = pd.DataFrame(data=None, columns=colnames)
        dfNew['PRICE VAR [%]'] = 0
        for i, y in enumerate(years):
            ydat = comInfos[comInfos['time'] == y]
            for col in colnames:
                dfNew.loc[y, col] = ydat[col].values[0]
            dfNew.loc[y, 'PRICE VAR [%]'] = ydat[price_vars[y]].values[0]
        dfNew.set_index(["time"], inplace=True)
        dfNew.to_csv(os.path.join(savePath, 'US', company + '.csv'))


if __name__ == '__main__':
    dataset = 'C:\\Users\\guowe\\Desktop\\CVPR\\ssi-template\\eswa\\data\\USstocks'
    savePath = 'C:\\workspace\\creditRating\\data'
    #read_cat_data(dataset, savePath)
    save_company('C:\\workspace\\creditRating\\data\\dat_all.csv', savePath)
