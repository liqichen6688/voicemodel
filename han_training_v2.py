from keras.models import Model
from keras.layers import Dense, Input, Activation, multiply, Lambda
from keras.layers import TimeDistributed, GRU, Bidirectional
from keras import backend as K
import keras
import os
import numpy as np
from keras.utils.np_utils import to_categorical




def han(features = 60):
    # refer to 4.2 in the paper whil reading the following code

    # Input for one day : max article per day =40, dim_vec=200
    input1 = Input(shape=(2, features), dtype='float32')

    # Attention Layer
    dense_layer = Dense(features, activation='relu')(input1)
    softmax_layer = Activation('softmax')(dense_layer)
    attention_mul = multiply([softmax_layer,input1])
    #end attention layer


    vec_sum = Lambda(lambda x: K.sum(x, axis=1))(attention_mul)
    pre_model1 = Model(input1, vec_sum)
    pre_model2 = Model(input1, vec_sum)

    # Input of the HAN shape (None,11,40,200)
    # 11 = Window size = N in the paper 40 = max articles per day, dim_vec = 200
    input2 = Input(shape=(20, 2, features), dtype='float32')

    # TimeDistributed is used to apply a layer to every temporal slice of an input
    # So we use it here to apply our attention layer ( pre_model ) to every article in one day
    # to focus on the most critical article
    pre_gru = TimeDistributed(pre_model1)(input2)

# bidirectional gru
    l_gru = Bidirectional(GRU(int(features / 2), return_sequences=True))(pre_gru)

    # We apply attention layer to every day to focus on the most critical day
    post_gru = TimeDistributed(pre_model2)(l_gru)

# MLP to perform classification
    dense1 = Dense(400, activation='relu')(post_gru)
    dense2 = Dense(400, activation='relu')(dense1)
    dense3 = Dense(3, activation='tanh')(dense2)
    final = Activation('softmax')(dense3)
    final_model = Model(input2, final)
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



def twin_creation( x_train_folder, y_train_folder) :
    ''' Here we create a list of twins ( duo_list)
	Twin = [CompanyA_x_train_filepath, CompanyA_y_train_filepath]'''
    x_train_list= os.listdir(x_train_folder)
    x_train_list=sorted(x_train_list)


    y_train_list=os.listdir(y_train_folder)
    y_train_list=sorted(y_train_list)


    duo_list=[]
    for i in range (len(y_train_list)):
        duo=[x_train_list[i],y_train_list[i]]
        duo_list.append(duo)

    duo_list=[duo for duo in duo_list if duo[0][-4:]=='.npy']

	
    return duo_list
def training(x_name,y_name,model):
    x_train = np.load('data_backup/feed_data/x_feature_train/'+x_name)
    y_train = np.load('data_backup/feed_data/y_train/'+y_name)


    y_oh_list=[] # y one hot for one hot encoding
    # Transforming Y so that it has 3 dim for a 3 class classification
    for trend in y_train:
        new_value=int(trend+1)
        code = [0 for _ in range(3)]
        code[new_value]=1
        y_oh_list.append(code)

        
    y_train_end=np.asarray(y_oh_list)

    # Encoding y
    #encoder = LabelEncoder()
    #encoder.fit(y_train)
    #encoded_Y = encoder.transform(y_train)
    print("model fitting on " + x_name)
    for i in range(10):
        train = model.train_on_batch(x_train, y_train_end)
        print(model.metrics_names[0] , ':' , train[0])
        print(model.metrics_names[1] , ':' , train[1])



def testing(x_test_folder, y_test_foloder,model):
    duo_test_list = []
    x_test_list= os.listdir(x_test_folder)
    x_test_list=sorted(x_test_list)
    y_test_list=os.listdir(y_test_folder)
    y_test_list=sorted(y_test_list)
    for i in range (len(y_test_list)):
        duo=[x_test_list[i],y_test_list[i]]
        duo_test_list.append(duo)
    duo_test_list=[duo for duo in duo_test_list if duo[0][-4:]=='.npy']

    num_sample = 0
    num_right_sample = 0
    for duo in duo_test_list:
        x_test = np.load('data_backup/feed_data/x_feature_test/'+duo[0])
        y_test = np.load('data_backup/feed_data/y_test/'+duo[1])
        num_sample += x_test.shape[0]
        y_test_list = []
        for trend in y_test:
            new_value = trend + 1
            code = [0 for _ in range(3)]
            code[new_value] = 1
            y_test_list.append(code)
        y_test_end = np.asarray(y_test_list)
        print('-------test--------')
        test = model.test_on_batch(x_test, y_test_end)
        print(model.metrics_names[0], ':', test[0])
        print(model.metrics_names[1], ':', test[1])
        num_right_sample += test[1] * x_test.shape[0]

    print('total performance{}/{} = {}'.format(int(num_right_sample), num_sample, num_right_sample/num_sample))


if __name__ == "__main__":
    model = han()
    optimizer = keras.optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
    model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
    # Put your training data folder path
    x_train_folder='data_backup/feed_data/x_feature_train'
    y_train_folder='data_backup/feed_data/y_train'
    x_test_folder = 'data_backup/feed_data/x_feature_test'
    y_test_folder = 'data_backup/feed_data/y_test'
    epochs=200
	
    duo_list= twin_creation(x_train_folder, y_train_folder)
    for epoch in range(epochs):
        for k,duo in enumerate(duo_list):
            print('fitting on firm nb {} out of 494 epoch {}'.format(k,epoch))
            training(duo[0],duo[1],model)
        epoch += 1

    testing(x_test_folder, y_test_folder, model)

    model.save('your_model_{}epochs.hdf5'.format(epochs))
    
#load_data(model, '', '', '', '')
