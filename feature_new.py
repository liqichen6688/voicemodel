# -*- coding: utf-8 -*-
"""
Created on Thu Jul  4 15:30:16 2019

@author: Luofeng
"""
import pandas as pd
import numpy as np
import librosa
import os

LEN_DATA = 241
NUM_FEATURE = 20
NUM_PERIOD = 3
NUM_DAYS = 20
MAX_PERIOD = 20
data_path = 'data_backup/1minbar_new'
frame_length = 40
store_path = 'data_backup/feed_data'
BEGIN_ID = 155
#LABEL_CLOSE = "0"
#LABEL_VOLUME = "1"

LABEL_CLOSE = "ClosePrice"
LABEL_VOLUME = "BargainAmount"

def get_features(series):
    '''
    return wave-related features
    '''
    series = np.array(series).astype("float32")
    mfcc = librosa.feature.mfcc(series)
#    zrate_price = librosa.feature.zero_crossing_rate(series - np.mean(series))
#    features = np.append(mfcc, zrate_price, axis=0)
#    if len(mfcc) != 20: print(len(mfcc))
    return mfcc

def get_company_name():
    file_list = os.listdir(data_path)
    file_list = sorted(file_list)
    company_name = set()
    for name in file_list:
        index = name.find('_')
        company_name.add(name[:index])
    company_name -= set(['.DS'])
    return file_list, company_name

def get_company_features(df, scales = [1], m = "f"):
    '''
    return all scales of wave-related features
    '''
    try:
        init = True
        if m == "f":
            for i in scales:
                feature = np.array([[np.nan]*(i-1)] * NUM_FEATURE)
                dlen = LEN_DATA * i
                for j in range(int(len(df) / LEN_DATA) - i):
                    feature = np.append(feature,  np.array([get_features(df[j * LEN_DATA: j * LEN_DATA +dlen]).T[0]]).T, axis = 1)
                if init:
                    features = feature.T
                    init = False
                else:
                    if features.shape != feature.T.shape:
                        print(features.shape)
                        print(feature.shape)
                        print(feature.T.shape)
    #                    continue
                    features = np.append(features, feature.T, axis = 1)
        elif m == "l":
            for i in scales:
                dlen = LEN_DATA * i
                features = np.array([])
                for j in range(int(len(df) / LEN_DATA) - i):
                    features = np.append(features,  df[j * LEN_DATA +dlen] / df[j * LEN_DATA] - 1)
        return features
    except:
        return []



def main():
#if True:
    file_list, company_name = get_company_name()
    scales = [1,5,20]
    company_name = list(company_name)
    company_name.sort()
    for company in company_name:
        if int(company) < BEGIN_ID:
            continue
        print(company)
        company_file_list = [_ for _ in file_list if _[0:6] == company]
        company_file_list.sort()
        init = True
        for item in company_file_list:
            reader = pd.read_csv(data_path+"/" + item)
            if reader.shape[0] != LEN_DATA:
                continue
            if init:
                company_ts = reader
                init = False
            else:
                company_ts = pd.concat([company_ts, reader], axis = 0, ignore_index = True)
        clsfeature = get_company_features(company_ts[LABEL_CLOSE], scales)
        volfeature = get_company_features(company_ts[LABEL_VOLUME], scales)
        label = get_company_features(company_ts[LABEL_CLOSE], [1], m = "l")
        if len(clsfeature) == 0 or len(volfeature) == 0 or len(label) == 0:
            continue
#        clsfeature5 = get_company_features(company_ts["ClosePrice"], [5])
#        volfeature5 = get_company_features(company_ts["BargainAmount"], [5])
        num_days, num_features = clsfeature.shape
        assert  num_features == NUM_FEATURE * NUM_PERIOD
#        names = np.array([[int(company)] * num_days])
#        r = np.append(names, clsfeature, axis = 1)
#        r = np.append(r, volfeature, axis = 1).T
        r = np.array([clsfeature, volfeature])
        init = True
        for i in range(MAX_PERIOD + 1, num_days - NUM_DAYS):
            if init:
                fe = np.array([r[:,i:i+NUM_DAYS,:]])
                init = False
            else:
                fe = np.append(fe, np.array([r[:,i:i+NUM_DAYS,:]]), axis = 0)
        fe = fe.transpose((0,2,1,3))
        np.save(store_path + '/' + 'x_feature_train' + '/' + company +'_feature_train', fe[: int(0.8*num_days)])
        np.save(store_path + '/' + 'x_feature_test' + '/' + company + '_feature_test', fe[int(0.8 * num_days):])
        np.save(store_path + '/' + 'y_pretrain' + '/' + company +'_y_train', label)



def process_label():
    file_list, company_name = get_company_name()
    labels = np.array([])
    for company in company_name:
        try:
            prelabel = np.load(store_path + '/' + 'y_pretrain' + '/' + company +'_y_train.npy')
            labels = np.append(labels, prelabel)
        except:
            continue
    low_thres = np.percentile(labels, 33.33)
    high_thres = np.percentile(labels, 66.66)
    for company in company_name:
        try:
            prelabel = np.load(store_path + '/' + 'y_pretrain' + '/' + company +'_y_train.npy')
            label = np.array([])
            for k in range(MAX_PERIOD + NUM_DAYS + 1, len(prelabel)):
                if prelabel[k] < low_thres:
                    label = np.append(label, -1)
                elif prelabel[k] > high_thres:
                    label = np.append(label, 1)
                else:
                    label = np.append(label, 0)
            np.save(store_path + '/' + 'y_train' + '/' + company +'_y_train', label[: int(0.8*len(prelabel))])
            np.save(store_path + '/' + 'y_test' + '/' + company +"_y_test", label[int(0.8*len(prelabel)):])
        except:
            continue



if __name__ == '__main__':
    main()
    process_label()