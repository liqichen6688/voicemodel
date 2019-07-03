import os
import pandas as pd
import numpy as np
from python_speech_features import mfcc
import librosa
from scipy.stats import rankdata

data_path = 'data_backup/1minbar_new'
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
                if not new_data.empty:
                    data_dict[name].append(new_data)
    print('done')

    return data_dict

def get_feature(all_data, past_day_num=10, fre=16000):
    company_list = get_company_name()
    for company_name in company_list:
        folder = os.path.exists(store_path + '/' + 'x_train')
        num_min = all_data[company_name][0].shape[0]



        close_price = all_data[company_name][0][['ClosePrice']].to_numpy()[:,0]
        volume = all_data[company_name][0][['BargainAmount']].to_numpy()[:,0].astype(np.float)

        feature = get_all_feature(close_price, volume)

        feature = feature[np.newaxis, :]
        num_days = len(all_data[company_name])
        for i in range(1, num_days):
            close_price = all_data[company_name][i][['ClosePrice']].to_numpy()[:,0]
            volume = all_data[company_name][i][['BargainAmount']].to_numpy()[:,0].astype(np.float)
            new_feature = get_all_feature(close_price, volume)
            new_feature = new_feature[np.newaxis,:]
            feature = np.append(feature, new_feature, axis=0)

        price_feature = feature[:,0,:]
        volume_feature =  feature[:,1,:]
        lenth = price_feature.shape[0]
        for i in range(price_feature.shape[1]):
            price_feature[:,i] = rankdata(price_feature[:,i], method='ordinal')/lenth
            volume_feature[:,i] = rankdata(volume_feature[:,i], method='ordinal')/lenth

        feature[:,0,:] = price_feature
        feature[:,1,:] = volume_feature

        recur_feature = feature[np.newaxis, 0:past_day_num, :]
        for i in range(1, num_days-past_day_num):
            recur_feature = np.append(recur_feature, feature[np.newaxis, i:i+past_day_num,:], axis = 0)

        num_days = num_days - past_day_num
        np.save(store_path + '/' + 'x_feature_train' + '/' + company_name +'_feature_train', recur_feature[: int(0.8*num_days)])
        np.save(store_path + '/' + 'x_feature_test' + '/' + company_name + '_feature_test',
                recur_feature[int(0.8 * num_days):])
        get_prediction(all_data, past_day_num)

def get_prediction(all_data, past_day_num=10):
    company_list = get_company_name()
    company_returns = {}
    all_return = []
    for company_name in company_list:

        num_days = len(all_data[company_name])
        close_price = []
        for i in range(num_days):
            close_price.append(all_data[company_name][i][['ClosePrice']].to_numpy()[-1][0])

        company_returns[company_name] = [0]
        for i in range(1, num_days):
            company_returns[company_name].append((close_price[i]-close_price[i-1])/close_price[i-1])
            all_return.append((close_price[i]-close_price[i-1])/close_price[i-1])

    low_thres = np.percentile(all_return, 33.33)
    high_thres = np.percentile(all_return, 66.66)
    print(len(all_return), all_return)
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

def get_all_feature(close_price: np.ndarray, volume:np.ndarray):
    price_feature = librosa.feature.mfcc(close_price)
    volume_feature = librosa.feature.mfcc(volume)

    zrate_price = librosa.feature.zero_crossing_rate(close_price - np.mean(close_price))
    zrate_volume = librosa.feature.zero_crossing_rate(volume - np.mean(volume))
    price_feature = np.append(price_feature, zrate_price, axis=0)
    volume_feature = np.append(volume_feature, zrate_volume, axis=0)

    chromagram_price = librosa.feature.chroma_stft(close_price, n_chroma=12)
    chromagram_volume = librosa.feature.chroma_stft(volume, n_chroma=12)
    price_feature = np.append(price_feature, chromagram_price, axis=0)
    volume_feature = np.append(volume_feature, chromagram_volume, axis=0)

    cqt_price = librosa.feature.chroma_cqt(close_price, n_chroma=12)
    cqt_volume = librosa.feature.chroma_cqt(volume, n_chroma=12)
    price_feature = np.append(price_feature, cqt_price, axis=0)
    volume_feature = np.append(volume_feature, cqt_volume, axis=0)

    cens_price = librosa.feature.chroma_cens(close_price, n_chroma=12)
    cens_volume = librosa.feature.chroma_cens(volume, n_chroma=12)
    price_feature = np.append(price_feature, cens_price, axis=0)
    volume_feature = np.append(volume_feature, cens_volume, axis=0)

    mel_price = librosa.feature.melspectrogram(close_price - np.mean(close_price), n_mels=12)
    mel_volume = librosa.feature.melspectrogram(volume - np.mean(volume), n_mels=12)
    price_feature = np.append(price_feature, mel_price, axis=0)
    volume_feature = np.append(volume_feature, mel_volume, axis=0)

    rms_price = librosa.feature.rms(close_price)
    rms_volume = librosa.feature.rms(volume)
    price_feature = np.append(price_feature, rms_price, axis=0)
    volume_feature = np.append(volume_feature, rms_volume, axis=0)

    sc_price = librosa.feature.spectral_centroid(close_price)
    sc_volume = librosa.feature.spectral_centroid(volume)
    price_feature = np.append(price_feature, sc_price, axis=0)
    volume_feature = np.append(volume_feature, sc_volume, axis=0)

    bw1_price = librosa.feature.spectral_bandwidth(close_price, p=1)
    bw1_volume = librosa.feature.spectral_bandwidth(volume, p=1)
    price_feature = np.append(price_feature, bw1_price, axis=0)
    volume_feature = np.append(volume_feature, bw1_volume, axis=0)

    scon_price = librosa.feature.spectral_contrast(close_price)
    scon_volume = librosa.feature.spectral_contrast(volume)
    price_feature = np.append(price_feature, scon_price, axis=0)
    volume_feature = np.append(volume_feature, scon_volume, axis=0)

    sf_price = librosa.feature.spectral_flatness(close_price)
    sf_volume = librosa.feature.spectral_flatness(volume)
    price_feature = np.append(price_feature, sf_price, axis=0)
    volume_feature = np.append(volume_feature, sf_volume, axis=0)

    ro_price = librosa.feature.spectral_rolloff(close_price - np.mean(close_price))
    ro_volume = librosa.feature.spectral_rolloff(volume - np.mean(volume))
    price_feature = np.append(price_feature, ro_price, axis=0)
    volume_feature = np.append(volume_feature, ro_volume, axis=0)

    pf1_price = librosa.feature.poly_features(close_price - np.mean(close_price), order=2)
    pf1_volume = librosa.feature.poly_features(volume - np.mean(volume), order=2)
    price_feature = np.append(price_feature, pf1_price, axis=0)
    volume_feature = np.append(volume_feature, pf1_volume, axis=0)

    ton_price = librosa.feature.tonnetz(close_price)
    ton_volume = librosa.feature.tonnetz(volume)
    price_feature = np.append(price_feature, ton_price, axis=0)
    volume_feature = np.append(volume_feature, ton_volume, axis=0)

    price_feature = price_feature.T
    volume_feature = volume_feature.T
    feature = np.append(price_feature, volume_feature, axis=0)

    return feature

def get_market_feature(all_data: pd.DataFrame, past_day_num =10):
    company_list = get_company_name()
    for company_name in company_list:

        num_days = len(all_data[company_name])
        close_price = []
        all_close_price = []
        all_return = []
        for i in range(num_days):
            close_price.append(all_data[company_name][i][['ClosePrice']].to_numpy()[-1][0])
            close = all_data[company_name][i][['ClosePrice']].to_numpy()[:,0]
            high = all_data[company_name][i][['HighPrice']].to_numpy()[:,0]
            low = all_data[company_name][i][['LowPrice']].to_numpy()[:,0]
            all_return.append((high-low)/close)
            all_close_price.append(close)


        company_returns = 0
        market_feature = []
        company_spr = (all_close_price[0].max() - all_close_price[0].min())/close_price[0]
        r2 = (all_return[0] * all_return[0]).sum() / all_return[0].shape[0]
        r2_down = (all_return[0][all_return[0]>0] ** 2).sum() / all_return[0].shape[0]
        s = max(min(company_returns/(r2)**(1/2),10),-10)
        s_down = max(min(company_returns/(r2_down)**(1/2),10),-10)
        market_feature = np.array([company_returns, company_spr, r2, r2_down, s, s_down])
        market_feature = market_feature[np.newaxis, :]


        for i in range(1, num_days):
            company_returns= (close_price[i]-close_price[i-1])/close_price[i-1]
            company_spr = (all_close_price[i].max() - all_close_price[i].min())/close_price[i]
            r2 = (all_return[i] * all_return[i]).sum() / all_return[i].shape[0]
            r2_down = (all_return[i][all_return[i] > 0] ** 2).sum() / all_return[i].shape[0]
            s = company_returns/(r2)**(1/2)
            s_down = company_returns/(r2_down)**(1/2)
            new_feature = np.array([company_returns, company_spr, r2, r2_down, s, s_down])
            new_feature = new_feature[np.newaxis, :]
            market_feature = np.append(market_feature, new_feature, axis=0)

        recur_feature = market_feature[np.newaxis, 0:past_day_num, :]
        for i in range(1, num_days - past_day_num):
            recur_feature = np.append(recur_feature, market_feature[np.newaxis, i:i + past_day_num, :], axis=0)

        num_days = num_days - past_day_num
        np.save(store_path + '/' + 'x_market_train' + '/' + company_name + '_market_train',
                recur_feature[: int(0.8 * num_days)])
        np.save(store_path + '/' + 'x_market_test' + '/' + company_name + '_market_test',
                recur_feature[int(0.8 * num_days):])










if __name__ == '__main__':
    all_data = get_dataframe()
    get_market_feature(all_data)
    get_feature(all_data)
    get_prediction(all_data)
