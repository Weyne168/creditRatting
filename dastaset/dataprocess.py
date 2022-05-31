import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder

companyIDs = pd.read_excel('C:\\Users\\guowe\\Desktop\\CVPR\\ssi-template\\eswa\\data\\process.xlsx', usecols=['证券代码'])
# companyIDs = pd.read_excel('C:\\Users\\guowe\\Desktop\\CVPR\\ssi-template\\eswa\\data\\A股公司-主体信用评级(评级转化数值).xlsx', usecols=['证券代码'])
companyIDs = np.array(companyIDs.stack())
companyIDs = companyIDs.tolist()


def createDf(dataPath, savePath):
    data = pd.read_excel(dataPath)
    colName = list(data)

    colName.remove('证券代码')
    colName.remove('证券简称')
    colName.remove('company_type')
    colList = []
    rowList = []

    for col in colName:
        tmp = col.split('\n')
        colList.append(tmp[0])
        rowList.append(tmp[1].split(' ')[1])

    colNew = list(set(colList))
    colNew.append('证券代码')
    colNew.append('证券简称')

    rowNew = sorted(list(set(rowList)))
    # writer = pd.ExcelWriter(os.path.join(savePath, newFileName))
    for cid in companyIDs:
        dfNew = pd.DataFrame(data=None, index=rowNew, columns=colNew)
        dfNew.index.name = 'time'
        dfNew['证券代码'] = cid
        dfNew['证券简称'] = data['证券简称'][data['证券代码'] == cid].values[0]
        print(cid)
        dfNew.to_csv(os.path.join(savePath, cid + '.csv'), encoding='utf_8_sig')


def fillNewForm(oriFormPath, newFormDir):
    oriform = pd.read_excel(oriFormPath)
    companyType = oriform['company_type']
    labelEncoder = LabelEncoder()
    company_type_encode = labelEncoder.fit_transform(companyType)
    oriform['company_type'] = company_type_encode

    colName = list(oriform)
    colName.remove('证券代码')
    colName.remove('证券简称')
    colName.remove('company_type')

    for cid in companyIDs:
        oriComInfo = oriform.loc[oriform['证券代码'] == cid]
        print(cid)
        newForm = pd.read_csv(os.path.join(savePath, cid + '.csv'), encoding='utf_8_sig')
        newForm.set_index(["time"], inplace=True)
        for index, row in oriComInfo.iterrows():
            for col in colName:
                time = col.split('\n')[1].split(' ')[1]
                index = col.split('\n')[0]
                newForm.loc[time, index] = row[col]
        newForm['company_type'] = row['company_type']
        newForm.to_csv(os.path.join(savePath + '2', cid + '.csv'), encoding='utf_8_sig')


def fillRatings(oriFormPath, savePath):
    oriform = pd.read_excel(oriFormPath)
    colName = list(oriform)
    colName.remove('证券代码')
    colName.remove('证券简称')
    for cid in companyIDs:
        oriComInfo = oriform.loc[oriform['证券代码'] == cid]

        newForm = pd.read_csv(os.path.join(savePath + '2', cid + '.csv'))
        newForm.set_index(["time"], inplace=True)
        newForm['ratting'] = 0
        for time in colName:
            newForm.loc[time, 'ratting'] = int(oriComInfo[time].values[0])
            print(newForm.loc[time]['ratting'], oriComInfo[time].values[0])
        newForm.to_csv(os.path.join(savePath + '3', cid + '.csv'), encoding='utf_8_sig')
        # exit(1)


if __name__ == '__main__':
    dataPath_2 = 'C:\\Users\\guowe\\Desktop\\CVPR\\ssi-template\\eswa\\data\\process.xlsx'
    dataPath = 'C:\\Users\\guowe\\Desktop\\CVPR\\ssi-template\\eswa\\data\\A股公司-主体信用评级(评级转化数值).xlsx'
    savePath = 'C:\\workspace\\creditRating\\data\\dataCHR'
    createDf(dataPath_2, savePath)
    fillNewForm(dataPath_2, savePath)
    fillRatings(dataPath, savePath)
