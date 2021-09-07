import os

from keras.losses import mean_squared_error

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import numpy as np
from keras.callbacks import EarlyStopping
from keras.layers import LSTM, Dropout, Flatten, Dense
from keras.models import Sequential
import random
from tensorflow import keras


class Config(object):
    '''
    模型参数配置。预先定义模型参数和加载语料以及模型保存名称
    '''
    poetry_file = "file_data.txt"
    model_output = "final_lstm_model"
    ModelDir = "./ipynb_garbage_files/"

    VEC_SIZE = 160
    T_train = 10  # 最小样本数

def processing_data():
    with open(Config.ModelDir + Config.poetry_file, 'r') as f:
        county = 0
        x=[]
        y=[]
        for line in f:
            # 处理后数据
            process_data = [0.0 for i in range(160)]

            temp_Data = line.split()
            # 制作答题情况
            flag = random.randint(0, 1)
            if flag == 0:
                for i in range(80):
                    process_data[i] = 0.0
                    process_data[i + 80] = float(temp_Data[i])
            else:
                for i in range(80):
                    process_data[i] = float(temp_Data[i])
                    process_data[i + 80] = 0.0

            '''
            #查看答题情况
            if temp_Data[80]==0:
                for i in range(80):
                    process_data[i]=0
                    process_data[i+80]=temp_Data[i]
            else :
                for i in range(80):
                    process_data[i]=temp_Data[i]
                    process_data[i+80]=0
            '''
            x.append(process_data)
            y.append(flag)
            county += 1
        x = np.array(x)
        y = np.array(y)
    f.close()
    x_train=[]
    y_train=[]
    for i in range(0, x.shape[0] - Config.T_train,Config.T_train):

        given = x[i:i + Config.T_train]
        predict = y[i:i + Config.T_train]
        x_train.append(given)
        y_train.append(predict)
    x_train = np.array(x_train)
    y_train = np.array(y_train)
    return x_train,y_train


def train_lstm(x_train=None,y_train=None):

    model = Sequential()
    model.add(LSTM(1, input_shape=(Config.T_train, Config.VEC_SIZE)))
    model.add(Dropout(0.5))
    model.add(Flatten())
    model.add(Dense(1, activation='sigmoid'))

    # 训练模型
    #es = EarlyStopping(monitor='val_acc', patience=5)
    model.compile(loss="mse", optimizer="adam", metrics=['accuracy'])

    batch_size = 2
    epochs = 20
    try:
        model.fit(x_train, y_train,
                  validation_split=0.1,
                  batch_size=batch_size,
                  epochs=epochs,
                  #callbacks=[es],
                  shuffle=True)
    except Exception as e:
        with open('./exception.txt', 'w') as file:
            file.write(str(e))
            raise Exception('sb')

    model.save(Config.ModelDir + Config.model_output)


def start():

    x_train,y_train=processing_data()
    train_lstm(x_train,y_train)
    model = keras.models.load_model(Config.ModelDir + Config.model_output)
    feature = model.predict(np.array(x_train))
    print(feature)
    #mean_squared_error(np.array(y_train), feature)


start()
