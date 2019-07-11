# -*- coding: utf-8 -*-
"""
Created on Thu Jul  4 15:30:16 2019

@author: Luofeng
"""
import pandas as pd
import numpy as np
import librosa
import os

scales = [1, 5]
#LEN_DATA = 240
NUM_FEATURE = 20
NUM_PERIOD = len(scales)
NUM_DAYS = 20
MAX_PERIOD = max(scales)
NORM_WINDOW = 25
#data_path = 'data_backup/1minbar_new'
data_path = 'data_backup/1minbar'

frame_length = 40
store_path = 'data_backup/feed_data'
BEGIN_ID = 0
#LABEL_CLOSE = "0"
#LABEL_VOLUME = "1"

LABEL_CLOSE = "ClosePrice"
LABEL_VOLUME = "BargainAmount"


ALPHA_LIST = ["MACD", "DIF"]
alpha_path = 'data_backup/alpha'
def get_features(series):
    '''
    return wave-related features
    '''
    series = np.array(series).astype("float32")
    mfcc = librosa.feature.mfcc(series)
    if mfcc.shape[1] > 1:
        mfcc = np.array([mfcc[:,0]]).T
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

def get_company_features(df, len_ts, m = "f"):
    '''
    return all scales of wave-related features
    '''
#    try:
    if True:
        init = True
        if m == "f":
            for i in scales:
                feature = np.array([[np.nan]*(i-1)] * NUM_FEATURE)
                for j in range(len(len_ts) - i):
                    feature = np.append(feature, get_features(df[len_ts[j]: len_ts[j+i] ]), axis = 1)
                print(feature.shape)
                print(len(len_ts))
                assert feature.shape[1] == len(len_ts) - 1
#
#
#
#
#                dlen = LEN_DATA * i
#                for j in range(int(len(df) / LEN_DATA) - i + 1):
#                    feature = np.append(feature,  np.array([get_features(df[j * LEN_DATA: j * LEN_DATA +dlen]).T[0]]).T, axis = 1)
#
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
            print("Before")
            print(features.shape)
            features = normalization(features)
            print(features.shape)
            print("After")
        elif m == "l":
            for i in scales:
#                dlen = LEN_DATA * i
                features = np.array([np.nan])
                for j in range(1, len(len_ts) - 1):
                    features = np.append(features,  df[len_ts[j+1] -1] / df[len_ts[j]] - 1)
                assert len(features) == len(len_ts) - 1
        return features
#    except:
#        return []



def main():
#if True:
    file_list, company_name = get_company_name()
    company_name = list(company_name)
    company_name.sort()
#    for alpha in ALPHA_LIST:
##        assert os.path.exist(alpha_path + alpha + '.csv')
#        exec(alpha + "_ts = pd.read_csv(alpha_path + alpha + .csv")
#        print("Alpha: " + alpha + " is loaded!")
    for company in company_name:
        if int(company) < BEGIN_ID:
            continue
        print(company)
        company_file_list = [_ for _ in file_list if _[0:6] == company]
        company_file_list.sort()
        init = True
        for item in company_file_list:
            reader = pd.read_csv(data_path+"/" + item)
#            if reader.shape[0] != LEN_DATA:
#                continue
            if init:
                company_ts = reader
                len_ts = [0, reader.shape[0]]
                init = False
            else:
                company_ts = pd.concat([company_ts, reader], axis = 0, ignore_index = True)
                len_ts.append(company_ts.shape[0])
        clsfeature = get_company_features(company_ts[LABEL_CLOSE], len_ts)
        print(clsfeature.shape)
        print("aaaaa")
        volfeature = get_company_features(company_ts[LABEL_VOLUME], len_ts)
        print(volfeature.shape)
        print("Market")
        marfeature, timeline = get_company_markets(company, file_list)
        print(marfeature.shape)
        print(len(timeline))
        '''
        '''
#        var_col = []
#        for alpha in ALPHA_LIST:
#            exec(alpha + '_feature = np.array(get_alphas(timeline, ' + alpha + '_ts[str(int(company))]))[1:], ' + alpha + '_ts[\"date\"][1:]')
#            exec("var_col.append(" + alpha + "_feature)")
#        init = True
#        for alp in var_col:
#            if init:
#                alpfeature = np.array([alp])
#                init = False
#            else:
#                alpfeature = np.append(alpfeature, [alp])
#        print(alpfeature.shape)
        '''
        '''
        print("Label processing...")
        label = get_company_features(company_ts[LABEL_CLOSE], len_ts, m = "l")
#        if len(clsfeature) == 0 or len(volfeature) == 0 or len(label) == 0:
#            continue
#        clsfeature5 = get_company_features(company_ts["ClosePrice"], [5])
#        volfeature5 = get_company_features(company_ts["BargainAmount"], [5])
        num_days, num_features = clsfeature.shape
        assert  num_features == NUM_FEATURE * NUM_PERIOD
#        names = np.array([[int(company)] * num_days])
#        r = np.append(names, clsfeature, axis = 1)
#        r = np.append(r, volfeature, axis = 1).T
        r = np.array([clsfeature, volfeature, marfeature])
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
#        print("LABEL")
#        print(label)
        np.save(store_path + '/' + 'y_pretrain' + '/' + company +'_y_train', label)



def process_label():
    file_list, company_name = get_company_name()
    labels = np.array([])
    company_name = list(company_name)
    company_name.sort()
    print(company_name)
    for company in company_name:
        try:
            prelabel = np.load(store_path + '/' + 'y_pretrain' + '/' + company +'_y_train.npy')
            labels = np.append(labels, prelabel)
        except:
            continue
    for la in range(len(labels)):
        if str(labels[la]) == 'nan':
            labels[la] = 0
    low_thres = np.percentile(labels, 33.33)
    print("Low_thres: " + str(low_thres))
    high_thres = np.percentile(labels, 66.66)
    for company in company_name:
        try:
            prelabel = np.load(store_path + '/' + 'y_pretrain' + '/' + company +'_y_train.npy')
        except:
            continue
#        print("Prelabel")
#        print(prelabel)
        label = np.array([])
        for k in range(MAX_PERIOD + NUM_DAYS + 1, len(prelabel)):
            if prelabel[k] < low_thres:
                label = np.append(label, -1)
            elif prelabel[k] > high_thres:
                label = np.append(label, 1)
            else:
                label = np.append(label, 0)
        print(label)
        np.save(store_path + '/' + 'y_train' + '/' + company +'_y_train', label[NORM_WINDOW-1: int(0.8*(len(prelabel)-NORM_WINDOW+1))])
        np.save(store_path + '/' + 'y_test' + '/' + company +"_y_test", label[int(0.8*(len(prelabel)-NORM_WINDOW+1)):])
#        except:
#            continue

def normalization(feature: np.ndarray, window=NORM_WINDOW) -> np.ndarray:
    num_clomn = feature.shape[1]
    num_row = feature.shape[0]
    nor_feature = np.zeros((num_row - window + 1, num_clomn))
    for i in range(num_row - window + 1):
        past_day = feature[i:i+window]
        max = np.max(past_day, axis=0)
        min = np.min(past_day, axis=0)
        divided = max - min
        nor_feature[i, divided != 0] = (feature[i + window - 1, divided != 0]-min[divided != 0])/divided[divided != 0]
        nor_feature[i, divided == 0] = 0.5

    return nor_feature


def get_company_markets(company, file_list):
    '''
    usage: mktfeature = get_company_markets(company, scales)
    '''
    assert 1 in scales
    # Calculate daily alphas
    print(company)
    init = True
    company_file_list = [_ for _ in file_list if _[0:6] == company]
    company_file_list.sort()
    for item in company_file_list:
        reader = pd.read_csv(data_path+"/" + item)
        t_close = np.array(reader["ClosePrice"])[-1]
        t_open = reader["OpenPrice"][0]
        t_high = np.array(reader["AccuHighPrice"])[-1]
        t_low = np.array(reader["AccuLowPrice"])[-1]
        t_volume = np.array(reader["AccuBargainAmount"])[-1]
        t_amount = np.array(reader["AccuBargainSum"])[-1]
        t_turnover = np.array(reader["AccuTurnoverDeals"])[-1]
        if t_volume > 0:
            t_vwap = t_amount / t_volume
        else:
            t_vwap = 1 / 2 * (t_close + t_open)
        t_risk_close = np.std(reader["ClosePrice"])
        t_risk_high = np.std(reader["HighPrice"])
        t_risk_low = np.std(reader["LowPrice"])
        t_risk_open = np.std(reader["OpenPrice"])
        t_risk_vol = np.std(reader["BargainAmount"])
        t_risk_amount = np.std(reader["BargainSum"])
        t_risk_turnover = np.std(reader["TurnoverDeals"])
        t_date = reader["BargainDate"][0]
        if init:
            market_features = np.array([[ t_close, t_open, t_high,
                                        t_low, t_volume, t_amount, t_turnover,
                                        t_vwap, t_risk_close, t_risk_low, t_risk_high,
                                        t_risk_open, t_risk_vol, t_risk_amount,
                                        t_risk_turnover]]).T
            print(market_features.shape)
            timeline = [t_date]
            init = False
        else:
#            print("Before")
#            print(market_features.shape)
            market_features = np.append(market_features, np.array([[t_close, t_open, t_high,
                                        t_low, t_volume, t_amount, t_turnover,
                                        t_vwap, t_risk_close, t_risk_low, t_risk_high,
                                        t_risk_open, t_risk_vol, t_risk_amount,
                                        t_risk_turnover]]).T, axis = 1)
#            print("After")
#            print(market_features.shape)
            timeline.append(t_date)
    df = pd.DataFrame(market_features).T
    returns = np.array(df[0].diff())
    print("bbbb")
    print(returns)
    d_volume = np.array(df[4].diff())
    d_amount = np.array(df[5].diff())
    d_turnover = np.array(df[6].diff())
    d_vwap = np.array(df[7].diff())
    d_features = np.array([returns,d_volume, d_amount, d_turnover, d_vwap]).T
    print(d_features.shape)
    print("Before1")
    print(market_features.shape)
    market_features = np.append(market_features, d_features.T, axis = 0)
    print(market_features.shape)
    print("After1")
    market_features_p = market_features.copy()
    for scale in scales:
        if scale == 1:
            continue
        t_df = np.array(pd.DataFrame(market_features_p).T.rolling(scale).mean()).T
#        print(t_df)
        print(t_df.shape)
        print("Before2")
        print(market_features.shape)
        market_features = np.append(market_features, t_df, axis = 0)
        print(market_features.shape)
        print("After2")
#    market_features = get_alphas(company, market_features, timeline, ALPHA_LIST)
    market_features = normalization(market_features.T)
    print("market_features size: " + str(market_features.shape))
    print(market_features[0,:])
    print("++++++++++++++++++++++")
    print(market_features[-1,:])
    return market_features, timeline

#def get_alphas(timeline, ts, time):
#
#    return alp_features

if __name__ == '__main__':
    main()
    process_label()