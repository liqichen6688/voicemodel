import os
import pandas as pd
import numpy as np
from python_speech_features import mfcc

data_path = 'data_backup/1minbar'
frame_length = 40
store_path = 'data_backup/feed_data'


def get_company_name() -> set:
    file_list = os.listdir(data_path)
    file_list = sorted(file_list)
    company_name = set()
    for name in file_list:
        index = name.find('_')
        company_name.add(name[:index])

    company_name -= set(['.DS'])
    return company_name

def get_dataframe() -> dict:
    print('loading data.....')
    file_list = os.listdir(data_path)
    file_list = sorted(file_list)
    if '.DS_Store' in file_list:
        file_list.pop(0)
    company_name = get_company_name()
    data_dict = {}
    for name in company_name:
        for file_name in file_list:
            if file_name.startswith(name):
                if name not in data_dict.keys():
                    data_dict[name] = []
                new_data = pd.read_csv(data_path + '/' + file_name)
                new_data = new_data.drop('Unnamed: 0', axis=1)
                new_data = new_data.drop('symbol', axis=1)
                if not new_data.empty:
                    data_dict[name].append(new_data)
    print('done')

    return data_dict

def get_mfcc_feature(all_data, past_day_num=5, fre=16000):
    company_list = get_company_name()
    for company_name in company_list:
        folder = os.path.exists(store_path + '/' + 'x_train')
        num_min = all_data[company_name][0].shape[0]



        close_price = all_data[company_name][0].iloc[:,-2].values
        volume = all_data[company_name][0].iloc[:,-1].values

        mfcc_price = mfcc(close_price - np.mean(close_price), fre, close_price.shape[0]/fre, close_price.shape[0]/fre, nfft = close_price.shape[0],numcep=14)
        mfcc_volume = mfcc(volume - np.mean(volume), fre, volume.shape[0]/fre, volume.shape[0]/fre, nfft = volume.shape[0],numcep=14)
        feature = np.append(mfcc_price, mfcc_volume, axis=0)

        feature = feature[np.newaxis, :]
        num_days = len(all_data[company_name])
        for i in range(1, num_days):
            close_price = all_data[company_name][0].iloc[:,-2].values
            volume = all_data[company_name][0].iloc[:,-1].values
            mfcc_price = mfcc(close_price - np.mean(close_price), fre, close_price.shape[0]/fre, close_price.shape[0]/fre, nfft = close_price.shape[0],numcep=14)
            mfcc_volume = mfcc(volume - np.mean(volume), fre, volume.shape[0]/fre, volume.shape[0]/fre, nfft = volume.shape[0],numcep=14)
            new_feature = np.append(mfcc_price, mfcc_volume, axis=0)
            new_feature = new_feature[np.newaxis,:]

            feature = np.append(feature, new_feature, axis=0)

        recur_feature = feature[np.newaxis, 0:past_day_num, :]
        for i in range(1, num_days-past_day_num):
            recur_feature = np.append(recur_feature, feature[np.newaxis, i:i+past_day_num,:], axis = 0)

        num_days = num_days - past_day_num
        np.save(store_path + '/' + 'x_train' + '/' + company_name +'_feature_train', recur_feature[: int(0.8*num_days)])
        np.save(store_path + '/' + 'x_test' + '/' + company_name + '_feature_test',
                recur_feature[int(0.8 * num_days):])
        get_prediction(all_data, past_day_num)

def get_prediction(all_data, past_day_num=5):
    company_list = get_company_name()
    company_returns = {}
    all_return = []
    for company_name in company_list:

        num_days = len(all_data[company_name])
        close_price = []
        for i in range(num_days):
            close_price.append(all_data[company_name][i].iloc[-1, -2])

        company_returns[company_name] = [0]
        for i in range(1, num_days):
            company_returns[company_name].append((close_price[i]-close_price[i-1])/close_price[i-1])
            all_return.append((close_price[i]-close_price[i-1])/close_price[i-1])

    low_thres = np.percentile(all_return, 33.33)
    high_thres = np.percentile(all_return, 66.66)
    for company_name in company_list:
        num_days = len(company_returns[company_name])
        for i in range(num_days):
            if company_returns[company_name][i] < low_thres:
                company_returns[company_name][i] = -1
            elif company_returns[company_name][i] < high_thres:
                company_returns[company_name][i] = 0
            else:
                company_returns[company_name][i] = 1

        num_days = num_days - past_day_num
        company_returns[company_name] = np.array(company_returns[company_name])
        save = company_returns[company_name][past_day_num:]
        np.save(store_path + '/' + 'y_train' + '/' + company_name +'_y_train', save[: int(0.8*num_days)])
        np.save(store_path + '/' + 'y_test' + '/' + company_name + '_y_test', save[int(0.8 * num_days):])




if __name__ == '__main__':
    all_data = get_dataframe()
    get_mfcc_feature(all_data)
    get_prediction(all_data)