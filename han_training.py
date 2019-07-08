from keras.models import Model
from keras.layers import Dense, Input, Activation, multiply, Lambda
from keras.layers import TimeDistributed, GRU, Bidirectional
from keras import backend as K
import keras
import os
import numpy as np
import tensorflow as tf
from keras.utils.np_utils import to_categorical




def han():
    # refer to 4.2 in the paper whil reading the following code
    #window_size = 5
    market_input = Input(shape=(20, 6), dtype='float32' )
    market_out = Dense(90, activation='sigmoid')(market_input)
    market_out1 = Lambda(lambda x: K.expand_dims(x, axis=2))(market_out)
    market_input = Input(shape=(10, 6), dtype='float32' )
    market_1 = Dense(90, activation='tanh')(market_input)
    market_2 = Activation('softmax')(market_1)
    market_out1 = Lambda(lambda x: K.expand_dims(x, axis=2))(market_2)

    # Input for one day : max article per day =40, dim_vec=200
    input1 = Input(shape=(3, 90), dtype='float32')

    # Attention Layer
    dense_layer = Dense(90, activation='sigmoid')(input1)
    softmax_layer = Activation('sigmoid')(dense_layer)
    dense_layer = Dense(90, activation='tanh')(input1)
    softmax_layer = Activation('softmax')(dense_layer)
    attention_mul = multiply([softmax_layer,input1])
    #end attention layer


    vec_sum = Lambda(lambda x: K.sum(x, axis=1))(attention_mul)
    pre_model1 = Model(input1, vec_sum)
    pre_model2 = Model(input1, vec_sum)

    # Input of the HAN shape (None,11,40,200)
    # 5 = Window size = N in the paper 40 = max articles per day, dim_vec = 200
    input2 = Input(shape=(20, 2, 90), dtype='float32')

    # TimeDistributed is used to apply a layer to every temporal slice of an input
    # So we use it here to apply our attention layer ( pre_model ) to every article in one day
    # to focus on the most critical article
    combine = keras.layers.concatenate([input2, market_out1], axis=2)
    pre_gru = TimeDistributed(pre_model1)(combine)

# bidirectional gru
    l_gru = Bidirectional(GRU(45, return_sequences=True))(pre_gru)

    # We apply attention layer to every day to focus on the most critical day
    post_gru = TimeDistributed(pre_model2)(l_gru)

# MLP to perform classification
    dense1 = Dense(400, activation='sigmoid')(post_gru)
    dense2 = Dense(400, activation='sigmoid')(dense1)
    dense3 = Dense(3, activation='sigmoid')(dense2)
    final = Activation('softmax')(dense3)
    final_model = Model(inputs=[input2, market_input], outputs=[final])
    final_model.summary()

    return final_model

'''
def load_data(model, x_train_file, x_test_file, y_train_file, y_test_file):
    x_train = np.load(x_train_file)
    y_train = np.load(y_train_file)

    x_test = np.load(x_test_file)
    y_test = np.load(y_test_file)


    encoder = LabelEncoder()
    encoder.fit(y_train)
    encoded_Y = encoder.transform(y_train)
    y_train_end = to_categorical(encoded_Y)


    encoder2 = LabelEncoder()
    encoder.fit(y_test)
    encoded_Y2 = encoder.transform(y_test)
    y_test_end = to_categorical(encoded_Y2)
    print(y_test_end.shape)



    print("model compiling - Hierachical attention network")

    model.compile(optimizer='adam',loss='categorical_crossentropy', metrics=['accuracy'])


    print("model fitting - Hierachical attention network")

    print(x_train.shape, y_test_end.shape)

    model.fit(x_train, y_train_end, epochs=200)


    print("validation_test")

    final_x_test_predict = model.predict(x_train)



    print("Prediction de Y ", final_x_test_predict)
    print("vrai valeur Y ", y_train)

    return

'''



def twin_creation(x_feature_folder, x_market_folder, y_train_folder) :
    ''' Here we create a list of twins ( duo_list)
	Twin = [CompanyA_x_train_filepath, CompanyA_y_train_filepath]'''
    x_feature_list= os.listdir(x_feature_folder)
    x_feature_list=sorted(x_feature_list)
    if x_feature_list[0] == '.DS_Store':
        x_feature_list.pop(0)

    x_market_folder = os.listdir(x_market_folder)
    x_market_list = sorted(x_market_folder)


    y_train_list=os.listdir(y_train_folder)
    y_train_list=sorted(y_train_list)
    if y_train_list[0] == '.DS_Store':
        y_train_list.pop(0)


    duo_list=[]
    for i in range (len(y_train_list)):
        duo=[x_feature_list[i], x_market_list[i], y_train_list[i]]
        duo_list.append(duo)

    duo_list=[duo for duo in duo_list if duo[0][-4:]=='.npy']

    return duo_list

def training(x_feature_name,x_market_name,y_name,model):
    x_feature = np.load('./data_backup/feed_data/x_feature_train/'+x_feature_name)
    y_train = np.load('./data_backup/feed_data/y_train/'+y_name)
    x_market = np.load('./data_backup/feed_data/x_market_train/'+x_market_name)


    y_oh_list=[] # y one hot for one hot encoding
    # Transforming Y so that it has 3 dim for a 3 class classification
    for trend in y_train:
        new_value=trend+1
        code = [0 for _ in range(3)]
        code[new_value]=1
        y_oh_list.append(code)

        
    y_train_end=np.asarray(y_oh_list)

    # Encoding y
    #encoder = LabelEncoder()
    #encoder.fit(y_train)
    #encoded_Y = encoder.transform(y_train)
    print("model fitting on " + x_feature_name)
    batch_size = 50
    data_length = y_train_end.shape[0]
    for i in range(100):
            index = np.random.randint(data_length-batch_size)
            x1 = x_feature[index:index+batch_size,:]
            #x1 = x1/np.linalg.norm(x1, axis=2)[:,:,np.newaxis]
            
            x2 = x_market[index:index+batch_size,:]
            while(False in np.isfinite(x1) or False in np.isfinite(x2)):
                index = np.random.randint(data_length-batch_size)
                x1 = x_feature[index:index+batch_size,:]
                x2 = x_market[index:index+batch_size,:]
#x2 = x2/np.linalg.norm(x2, axis=2)[:,:,np.newaxis]
            #print(x2)
            train = model.train_on_batch([x1, x2], y_train_end[index:index+batch_size,:])
            print(model.metrics_names[0] , ':' , train[0])
            print(model.metrics_names[1] , ':' , train[1])



def testing(x_feature_test_folder, x_market_test_folder, y_test_folder,model):
    duo_test_list = []
    x_feature_test_list= os.listdir(x_feature_test_folder)
    x_feature_test_list=sorted(x_feature_test_list)
    if x_feature_test_list[0] == '.DS_Store':
        x_feature_test_list.pop(0)
    x_market_test_list= os.listdir(x_market_test_folder)
    x_market_test_list=sorted(x_market_test_list)
    if x_market_test_list[0] == '.DS_Store':
        x_market_test_list.pop(0)
    y_test_list=os.listdir(y_test_folder)
    y_test_list=sorted(y_test_list)
    if y_test_list[0] == '.DS_Store':
        y_test_list.pop(0)
    for i in range (len(y_test_list)):
        duo=[x_feature_test_list[i], x_market_test_list[i],y_test_list[i]]
        duo_test_list.append(duo)
    duo_test_list=[duo for duo in duo_test_list if duo[0][-4:]=='.npy']

    num_sample = 0
    num_right_sample = 0
    for duo in duo_test_list:
        x_feature_test = np.load('./data_backup/feed_data/x_feature_test/'+duo[0])
        x_market_test = np.load('./data_backup/feed_data/x_market_test/'+duo[1])
        y_test = np.load('./data_backup/feed_data/y_test/'+duo[2])
        num_sample += x_feature_test.shape[0]
        y_test_list = []
        for trend in y_test:
            new_value = trend + 1
            code = [0 for _ in range(3)]
            code[new_value] = 1
            y_test_list.append(code)
        y_test_end = np.asarray(y_test_list)
        print('-------test--------')
        x1 = x_feature_test
        x2 = x_market_test
        test = model.test_on_batch([x1, x2], y_test_end)
        print(model.metrics_names[0], ':', test[0])
        print(model.metrics_names[1], ':', test[1])
        num_right_sample += test[1] * x_feature_test.shape[0]

    print('total performance{}/{} = {}'.format(int(num_right_sample), num_sample, num_right_sample/num_sample))




if __name__ == "__main__":
    model = han()
    opt = keras.optimizers.Adagrad(lr=0.0001, epsilon=None, decay=0.04, clipnorm=1.)
    model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])
    # Put your training data folder path
    x_feature_train_folder='./data_backup/feed_data/x_feature_train'
    y_train_folder='./data_backup/feed_data/y_train'
    x_market_train_folder = './data_backup/feed_data/x_market_train'

    x_market_test_folder = './data_backup/feed_data/x_market_test'
    x_feature_test_folder = './data_backup/feed_data/x_feature_test'
    y_test_folder = './data_backup/feed_data/y_test'
    epochs=40
	
    duo_list= twin_creation(x_feature_train_folder, x_market_train_folder,y_train_folder)
    for epoch in range(epochs):
        stock_index = np.random.randint(len(duo_list))
        print('fitting on firm nb {} out of 494 epoch {}'.format(stock_index,epoch))
        duo = duo_list[stock_index]
        training(duo[0],duo[1], duo[2],model)
        epoch += 1

    testing(x_feature_test_folder, x_market_test_folder,y_test_folder, model)

    model.save('your_model_{}epochs.hdf5'.format(epochs))
    
#load_data(model, '', '', '', '')
