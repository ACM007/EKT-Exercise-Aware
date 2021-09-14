import os

from keras.applications.densenet import layers

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import numpy as np
from keras.layers import LSTM, Dropout, Flatten, Dense, Lambda
from keras.models import Model, Sequential
import random
# from tensorflow import keras
import keras
import tensorflow as tf

class Config(object):
    '''
    模型参数配置。预先定义模型参数和加载语料以及模型保存名称
    '''
    poetry_file = "file_data.txt"
    model_output = "final_lstm_model"
    ModelDir = "./ipynb_garbage_files/"

    time_step=1
    dh = 80
    dv = 100
    VEC_SIZE = 160
    T_train = 0  # 最小样本数
    dy = 50



input_x = []#有问题，应该是每个状态都对应一个输入
input_y = []#有问题，需要加进去


def processing_data():
    with open(Config.ModelDir + Config.poetry_file, 'r') as f:
        zero_array = np.array([0.0 for i in range(Config.dh)])
        x=[]
        y=[]
        for line in f:

            temp_Data=np.array(line.split())

            temp_Data=temp_Data.astype(np.float32)

            input_x.append(temp_Data)

            # 制作答题情况
            flag = random.randint(0, 1)

            if flag == 0:
                temp_Data = np.concatenate((zero_array,temp_Data))
            else:
                temp_Data=np.concatenate((temp_Data,zero_array))

            x.append(temp_Data)
            y.append(flag*1.0)
            Config.T_train+=1
    f.close()
    x=np.array(x)
    y=np.array(y)
    x = x[None, :, :]
    y = y[None, :]
    return x,y


## intersection over union
def IoU(y_true, y_pred):  # 自己定义的损失函数
    y_1=y_true*tf.math.log(y_pred)
    print(y_1)
    #exit()
    y_2=(1-y_true)*tf.math.log(1-y_pred)
    print(y_2)
    #exit()
    y_3=y_1+y_2
    print(y_3)
    #exit()
    sum = -tf.reduce_sum(y_3)
    print(sum)
    #exit()
    return sum
    # with tf.compat.v1.Session() as sess:
    #
    #     array_y_true = y_true.eval(session=sess)
    #     array_y_pred = y_pred.eval(session=sess)
    #     exit()
    #
    # for i in range(Config.T_train):
    #     sum+= array_y_true[i] * np.log(array_y_pred[i])+(1-array_y_true[i])*np.log(1-array_y_pred[i])
    # return -sum


def input_Funtion(x):
    global input_x
    print(input_x.shape)
    input_x = tf.convert_to_tensor(input_x)
    LSTM_Layer_Output = tf.concat([x,input_x],2)
    print(LSTM_Layer_Output)
    # exit()
    # a=np.concatenate(x,input_x,axis=2)
    # print(a.shape)
    # exit()
    # for i in range(0, x.shape[1]):
    #     a = []
    #     for j in range(Config.dh):
    #         a.append(x[0][i][j])  # 把当前的隐藏层加进数组
    #     for j in range(Config.dh):
    #         a.append(input_x[0][i][j])  # 把隐藏层对应的输入加进去
    #     LSTM_Layer_Output.append(a)
    # LSTM_Layer_Output = np.array(LSTM_Layer_Output)
    # LSTM_Layer_Output = LSTM_Layer_Output[None, :, :]
    # print(LSTM_Layer_Output.shape)
    # print(LSTM_Layer_Output.shape)
    # print(type(tf.convert_to_tensor(LSTM_Layer_Output, tf.float32, name='t')))

    return LSTM_Layer_Output


def _output_shape(input_shape):
    shape=list(input_shape)
    shape[-1]*=2
    return tuple(shape)


def train_lstm(x_train=None, y_train=None):
    global input_x
    input_x=np.array(input_x)
    input_x=input_x[None,:,:]

    tf.compat.v1.disable_eager_execution()

    model = Sequential()
    # 获取LSTM层的信息
    model.add(LSTM(Config.dh, input_shape=(Config.T_train, Config.VEC_SIZE), return_sequences=True))
    model.add(Dropout(0.5))

    # 中间层和输入进行连接
    model.add(Lambda(input_Funtion, output_shape=_output_shape))
    #model.add(Flatten())

    model.add(Dense(Config.dy, input_shape=(), activation='relu'))

    model.add(Dense(1, input_shape=(), activation='sigmoid'))

    # 训练模型
    # es = EarlyStopping(monitor='val_acc', patience=5)
    model.compile(loss=IoU, optimizer="adam", metrics=['accuracy'])
    batch_size = 20
    epochs = 20
    print(x_train.shape,y_train.shape)
    print(type(x_train),type(y_train))
    print(type(x_train[0]), type(y_train[0]))
    print(type(x_train[0][0]), type(y_train[0][0]))
    print(type(x_train[0][0][0]))

    model.fit(x_train, y_train,
              batch_size=batch_size,
              epochs=epochs)# callbacks=[es]

    model.save(Config.ModelDir + Config.model_output)


def start():
    x_train, y_train = processing_data()
    train_lstm(x_train, y_train)
    model = keras.models.load_model(Config.ModelDir + Config.model_output,custom_objects={'IoU':IoU})
    feature = model.predict(np.array(x_train))
    print(feature)
    # mean_squared_error(np.array(y_train), feature)


start()
# IoU([[1,2,3]],[[1],[2],[3]])
