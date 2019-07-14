# -*- coding: utf-8 -*-
store_path = 'data_backup/feed_data/0714'
import numpy as np
import pandas as pd
import glob
import os
for feature_path in [store_path + '/' + 'x_feature_train' + '/', store_path + '/' + 'x_feature_test' + '/']:
    filelist = [_ for _ in glob.glob(feature_path + "*.npy")]
    init = True
    for i in filelist:
        temp = np.load(i)
        print(i, temp.shape)
        temp = temp.reshape([temp.shape[0], 20,temp.shape[2] * temp.shape[3]])[:,-2,:]
        if init:
            templist = temp
            init = False
        else:
            print(temp.shape)
            templist = np.append(templist, temp, axis = 0)
            print(templist.shape)
    if not os.path.exists(feature_path[:-1] +"_all/"):
        os.makedirs(feature_path[:-1] +"_all/")
    np.save(feature_path[:-1] +"_all/all_data", templist )
for feature_path in [store_path + '/' + 'y_train' + '/', store_path + '/' + 'y_test' + '/']:
    filelist = [_ for _ in glob.glob(feature_path + "*.npy")]
    init = True
    for i in filelist:
        temp = np.load(i)
        if init:
            templist = temp
            init = False
        else:
            templist = np.append(templist, temp, axis = 0)
    if not os.path.exists(feature_path[:-1] +"_all/"):
        os.makedirs(feature_path[:-1] +"_all/")
    np.save(feature_path[:-1] +"_all/all_data", templist )
    
    
training_set = pd.DataFrame(np.append(np.load(store_path + '/x_feature_train_all/all_data.npy'), np.array([np.load(store_path +  '/y_train_all/all_data.npy')]).T, axis = 1))
test_set = pd.DataFrame(np.append(np.load(store_path + '/' + "x_feature_test_all"+ '/all_data.npy'), np.array([np.load(store_path + '/' + 'y_test_all' + '/all_data.npy')]).T, axis = 1))
training_set.to_csv("training_set_all.tsv", sep = "\t")
test_set.to_csv("test_set_all.tsv", sep = "\t")