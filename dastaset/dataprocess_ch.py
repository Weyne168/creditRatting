import os
import pandas as pd


def read_sector4company(sector_pt):
    companyID2sector = {}
    sectors = {}
    for root, dirs, files in os.walk(sector_pt):
        for file in files[1:]:
            sector = file.split('(')[0]
            if sector not in sectors.keys():
                sectors[sector] = len(sectors)
            frame = pd.read_excel(os.path.join(root, file))
            corps = list(frame['代码'].values)
            for c in corps[1:]:
                companyID2sector[c] = sectors[sector]
    print('sector types==', len(sectors))
    return companyID2sector


def read_cat_data(dataset, savePath, company_sector):
    for root, dirs, files in os.walk(dataset):
        for file in files:
            frame = pd.read_excel(os.path.join(root, file))
            colnames = list(frame.iloc[1:, 0].values)
            print(file)

            tms = list(frame.iloc[0, 1:].values)
            tms = sorted(tms, reverse=True)
            dfNew = pd.DataFrame(data=None, index=tms, columns=colnames)

            for i, t in enumerate(tms):
                for j, col in enumerate(colnames):
                    v = frame.iloc[j + 1, i + 1]
                    if type(v) is str:
                        continue
                    dfNew.loc[t, col] = v
            dfNew.index.name = 'time'

            companyID = file.split('.')[0]
            fps = [companyID + '.SZ', companyID + '.SH', companyID + '.BJ']
            keep = False
            for fp in fps:
                if fp in company_sector:
                    dfNew['Sector'] = company_sector[fp]
                    keep = True
                    break
            if keep == False:
                continue
            dfNew.sort_index(ascending=True, inplace=True)
            dfNew.to_csv(os.path.join(savePath, companyID + '.csv'), encoding='utf_8_sig')
            print(companyID)
            # exit(1)


if __name__ == '__main__':
    dataset = 'C:\\Users\\guowe\\Desktop\\CVPR\\ssi-template\\eswa\\data\\financial_reports'
    savePath = 'C:\\workspace\\creditRating\\data\\CH'
    sector_pt = 'C:\\Users\\guowe\\Desktop\\CVPR\\ssi-template\\eswa\\data\\Chinese Listed Companies\\all_corps\\values'

    company_sector = read_sector4company(sector_pt)
    read_cat_data(dataset, savePath, company_sector)
