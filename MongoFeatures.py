'''newmongo'''
import numpy as np
import pandas as pd
import pymongo
import re
import datetime
from collections import defaultdict
#import zhongzheng800
import baostock as bs
import os
import xlrd
lg = bs.login()
rs = bs.query_sz50_stocks()
sz50_stocks = []
while (rs.error_code == '0') & rs.next():
    # 获取一条记录，将记录合并在一起
    sz50_stocks.append(rs.get_row_data())
a = list(pd.DataFrame(sz50_stocks)[1])
symbol_list =  [ _[3:] for _ in a]
bs.logout()
# symbol_list = zhongzheng800.symbol_list
#symbol_list = None

# default_config = {
#     'user'   : 'xtech',
#     'passwd' : 'Xtech2018',
#     'host'   : '175.25.50.123',
#     'port'   : '32789',
#     'db'     : 'admin',
#     }

#default_config = {
#    'user': 'xtech_read',
#    'passwd': 'factor_z0t0p0',
#    'host': '175.25.50.121',
#    'port': '32788',
#    'db': 'factor_ztp',
#}

my_config = {
     'user': 'xtech_guest',
     'passwd': 'Xtech123',
     'host': '175.25.50.119',
     'port': '27017',
     'db': 'L1_STOCK_MIN',
 }


class BTMongo(object):
    def __init__(self, config, name):
        self.config = config
        self.url = 'mongodb://{user}:{passwd}@{host}:{port}/{db}'.format(
            user=config['user'],
            passwd=config['passwd'],
            host=config['host'],
            port=config['port'],
            db=config['db'])
        print('url', self.url)
        self.client = pymongo.MongoClient(self.url)
        self._db = self.client[name]
        self.collist = self._db.collection_names()

    @property
    def db(self):
        return self._db

    def set_db(self, db, user=None, passwd=None):
        self._db = self.client[db]
        if user != None and passwd != None:
            self._db.authenticate(user, passwd)
        return self._db

    def get_collections(self):
        self.collist = self._db.collection_names()
        return self.collist

    def get_data(self, name, symbol):
        collection = self._db.get_collection(name)
        document = collection.find({"StockCode" : symbol})
        fac_df = pd.DataFrame(list(document))
        if fac_df.empty:
            print(name + "=[]");
            return fac_df
        del fac_df['_id']
        return fac_df
        # print(document)


####################################

# if __name__ == '__main__':
#     #factor_loader.get_factor_latest('event_factor')
#     '''
#     factor_loader = FactordfMongo('demo', 'EMA5', default_config)
#     col = factor_loader.get_collections()
#     description = factor_loader.factor_description(col[0])
#     latest = factor_loader.get_factor_all(col[0])
#     #print(factor_loader.summary)
#     latest = latest.melt(id_vars=['date'],var_name='ticker',value_name='proba')
#     print(latest)
#     '''
#     data_loader = BTMongo(config=default_config)
#     table_names = data_loader.get_collections()
#     data=data_loader.get_data(table_names[0])
#     print(data)

def get_feature(data_loader, feature_name, symbol):
    # ret_data=pd.DataFrame(columns=['date']+symbol_list)
#    ret_data = pd.DataFrame()
    tmp_data = data_loader.get_data(feature_name, symbol)
    if tmp_data.empty: return tmp_data
    if tmp_data.shape[0] == 1: return tmp_data
    # print(tmp_data)
#    print(tmp_data.name[0], tmp_data.formula[0])
#    st = 0;
#    ed = len(tmp_data)
#    form = tmp_data['formula']
#    useless = []
#    for i in range(len(form)):
#        if isinstance(form[i], str) == True:
#            useless.append(i)
#
#    tmp_data = tmp_data.drop(tmp_data.index[useless])
#    tmp_data.drop_duplicates('date', 'first', inplace=True)
#    tmp_data.set_index('date')
#    # for i in range(len(form)):
#    # 	if st==0 and isinstance(form[i],str)==False: st=i
#    # 	if st!=0 and isinstance(form[i],str): ed=i
#    # print(st,ed)
#    # print(tmp_data)
#    # # tmp_data=tmp_data.iloc[st:ed]
#    # ret_data.date=tmp_data.index.shiftime("%Y%m%d")
#    ret_data['date'] = [int(s.replace('-', '')) for s in tmp_data['date']]
#    tp = 1
#    if '000001' in tmp_data.columns:
#        tp = 0
#
#    if symbol_list is None:
#        for st in tmp_data.keys():
#            if st[:6].isdigit():
#                # ret_data[st]=tmp_data['%06d'%st]
#                ret_data[st] = np.array(tmp_data[st])
#    else:
#        for st in symbol_list:
#            if tp == 0:
#                # ret_data[st]=tmp_data['%06d'%st]
#                ret_data[st] = np.array(tmp_data['%06d' % st])
#            else:
#                # ret_data[st]=tmp_data['%d'%st]
#                ret_data[st] = np.array(tmp_data['%d' % st])
    return tmp_data


def save_feature(data, path):
#    if not os.path.exists(path):
#     	os.makedirs(path)
    if not os.path.exists(path):
        data.to_csv(path, index=False)


def get_features(data_loader, feature_names):
    for x in feature_names:
        for z in symbol_list:
            st_data = get_feature(data_loader, x, z)
            if st_data.empty or st_data.shape[0] == 1: print("none ++++++++++++++++++++++++");continue
    #        train_data=st_data[(st_data.date>=20150901) & (st_data.date<=20180831)]
    #        print(train_data.date)
            # test_data = st_data[(st_data.date >= 20150901)]
            # train_path='./MongoFeatures_Train/'+x+'.csv'
            test_path = './data_backup/1minbar_new/' + z +"_" + x  + '.csv'
            # save_feature(train_data,train_path)
            save_feature(st_data, test_path)


def get_list():
    workbook = xlrd.open_workbook('feature.xlsx')
    booksheet = workbook.sheet_by_name("allfactors")
    return booksheet


if __name__ == '__main__':
    # f_list = get_list()
    # nrows = f_list.nrows
    # for i in range(nrows):
    #     x = f_list.row_values(i)[0];
    #     y = f_list.row_values(i)[1]
    #     print(x, y)
    x = 'L1_STOCK_MIN'
    data_loader = BTMongo(config=my_config, name=x)
    y_list = [ _ for _ in data_loader.collist if _[-2:] == "M1" and _[:4] in ['2018','2019']]
    y_list.sort()
    for y in y_list:
        # if x!='jinyifei': continue
        if True:
            table_names = data_loader.get_collections()
            get_features(data_loader, [y])
            print("OVER " + str(y))
#        except:
#            print("ERROR!")
#            continue
'''

import fetchMongo
import imp
imp.reload(fetchMongo)
data_loader = fetchMongo.BTMongo(config=fetchMongo.default_config,name='zhangchuheng')
table_names = data_loader.get_collections()
feature_names=[]
for x in table_names:
    if x[0]=='g': feature_names.append(x)

fetchMongo.get_features(data_loader,feature_names)
st_data=fetchMongo.get_feature(data_loader,feature_names[0])
print(data)


imp.reload(fetchMongo)
data_loader = fetchMongo.BTMongo(config=fetchMongo.default_config,name='zhangchuheng')
tmp_data=data_loader.get_data('galpha191')
st_data=fetchMongo.get_feature(data_loader,'galpha191')
tdata=pd.read_csv('/home/user/stock/data/TrainDataMG/galpha191.csv')
tmp_data[(tmp_data.date>='2017-10-25') & (tmp_data.date<='2017-11-05')]['600000']
st_data[(st_data.date>=20171025) & (st_data.date<=20171105)][600000]
tdata[(tdata.date>=20171025)&(tdata.date<=20171105)]['600000']



import fetchMongo
import imp
imp.reload(fetchMongo)
list=fetchMongo.get_list()
nrows=list.nrows
for i in range(nrows):
	x=list.row_values(i)[0];y=list.row_values(i)[1]
	print(x,y)
	# if x!='jinyifei': continue
	data_loader = fetchMongo.BTMongo(config=fetchMongo.default_config,name=x)
	table_names = data_loader.get_collections()
	fetchMongo.get_features(data_loader,[y])

imp.reload(fetchMongo)
data_loader = fetchMongo.BTMongo(config=fetchMongo.default_config,name='gaotianyun')
tmp_data=data_loader.get_data('vwapClose_10')
st_data=fetchMongo.get_feature(data_loader,'vwapClose_10')
tmp_data[(tmp_data.date>='2017-10-25') & (tmp_data.date<='2017-11-05')]['600000']
st_data[(st_data.date>=20171025) & (st_data.date<=20171105)][600000]


imp.reload(fetchMongo)
data_loader = fetchMongo.BTMongo(config=fetchMongo.default_config,name='wangyue')
tmp_data=data_loader.get_data('RSST')
st_data=fetchMongo.get_feature(data_loader,'RSST')
tmp_data[(tmp_data.date>='2017-10-25') & (tmp_data.date<='2017-11-05')]['600000']
# ret_data[ret_data.date==20171030][600000]
st_data[(st_data.date>=20171025) & (st_data.date<=20171105)][600000]


imp.reload(fetchMongo)
data_loader = fetchMongo.BTMongo(config=fetchMongo.default_config,name='tangyuxuan')
tmp_data=data_loader.get_data('Barra_size')
st_data=fetchMongo.get_feature(data_loader,'Barra_size')
tmp_data[(tmp_data.date>='2017-10-25') & (tmp_data.date<='2017-11-05')]['600000']
st_data[(st_data.date>=20171025) & (st_data.date<=20171105)][600000]

'''
