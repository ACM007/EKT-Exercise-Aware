from gensim.models import word2vec
import numpy as np
import nltk
from keras.callbacks import EarlyStopping
from keras.layers import LSTM, Dropout, Bidirectional, Flatten, Dense
from keras.models import Model, Sequential
from keras.layers import Embedding

nltk.download('punkt')


### 2.2. Word2vec 训练

# 用生成器的方式读取文件里的句子
# 适合读取大容量文件，而不用加载到内存
class MySentences(object):
    def __init__(self, fname):
        self.fname = fname

    def __iter__(self):
        for line in open(self.fname, 'r'):
            yield line.split()


# 模型训练函数并获取文字向量
def w2vTrain(Config):
    sentences = MySentences(Config.poetry_file)
    w2v_model = word2vec.Word2Vec(sentences,
                                  min_count=Config.MIN_COUNT,
                                  workers=Config.CPU_NUM,
                                  vector_size=Config.VEC_SIZE,
                                  window=Config.CONTEXT_WINDOW
                                  )
    w2v_model.save(Config.ModelDir + Config.model_output)
    word_vector_dict = {}
    for word in w2v_model.wv.index_to_key:
        word_vector_dict[word] = list(w2v_model.wv[word])
        print(word,word_vector_dict[word])
        print('\n')
    vector_file = "./ipynb_garbage_files/word_vector.txt"
    with open(vector_file, 'w', encoding='utf-8')as f:
        f.write(str(word_vector_dict))


# 数据预处理

# 数据文本
corpus = []


def preprocess_file(Config):
    # 语料文本内容
    with open(Config.poetry_file, 'r', encoding='utf-8') as f:
        for line in f:
            # 分词
            corpus.append(nltk.word_tokenize(line))


# 数据处理

# 窗口大小
sw_steps = 1


def preprocess_data():
    # 把文本数据筛选一边
    w2v_model = word2vec.Word2Vec.load(Config.ModelDir + Config.model_output)

    # 测试循环几次
    test_num=0
    with open(Config.file_txt, 'w') as f:
        for sent in corpus:
            if test_num!=5001:
                # word2vec后的新数据
                text_stream = []#过滤后的数据
                text_len = 0
                for word in sent:
                    if word in w2v_model.wv.index_to_key:#是否在word2vec生成的列表里
                        text_stream.append(word)
                        text_len += 1
                if text_len < 4: continue
                # 构造数据集

                # 训练数据
                x = []
                y = []
                for i in range(0, len(text_stream) - sw_steps):
                    given = text_stream[i:i+sw_steps]#步长为1，每一个单词对应下一个单词
                    predict = text_stream[i + sw_steps]
                    x.append(w2v_model.wv[given].tolist())
                    y.append(w2v_model.wv[predict].tolist())
                x = np.array(x)
                y = np.array(y)
                # print("!!!!!!!!!!!!")
                # print(y)

                # 生成模型
                model = Sequential()
                # model.add(Embedding(3800,32,input_length=380))
                # model.add(Dropout(0.5))
                model.add(Bidirectional(LSTM(40, input_shape=(x.shape[1], x.shape[2]), return_sequences=True),
                                        merge_mode='concat'))#双向lstm层
                model.add(Dropout(0.5))#Dropout层
                model.add(Flatten())#Flatten()层
                model.add(Dense(Config.VEC_SIZE, activation='sigmoid'))#Dense层

                # 训练模型
                es = EarlyStopping(monitor='val_acc', patience=5)
                model.compile(loss="mse", optimizer="adam", metrics=['accuracy'])

                batch_size = 64
                epochs = 20

                model.fit(x, y,
                          validation_split=0.1,
                          batch_size=batch_size,
                          epochs=epochs,
                          callbacks=[es],
                          shuffle=True)#训练模型
                # for index in range(4):
                #     layer=model.get_layer(index=index)
                #     print(layer)

                # 获得模型的隐藏层状态，进行最大化池，最终结果作为训练题目
                layer_model = Model(inputs=model.input, outputs=model.layers[0].output)#输出中间层
                feature = layer_model.predict(x)

                # maxpooling隐藏层，然后结合答题结果输出
                hide_i = feature.shape[0]
                hide_j = feature.shape[1]
                hide_k = feature.shape[2]

                for k in range(0, hide_k):
                    feature_maxn = feature[0][0][k]
                    for j in range(0, hide_j):
                        for i in range(0, hide_i):
                            feature_maxn = max(feature_maxn, feature[i][j][k])
                    #f.write(str(feature_maxn) + ' ')
                #f.write('\n')
            test_num+=1
    f.close()


class Config(object):
    '''
    模型参数配置。预先定义模型参数和加载语料以及模型保存名称
    '''
    poetry_file = "./bioCorpus_5000.txt"
    model_output = "test_w2v_model"
    ModelDir = "./ipynb_garbage_files/"
    file_txt = "./ipynb_garbage_files/file_data.txt"

    MIN_COUNT = 4
    CPU_NUM = 2  # 需要预先安装 Cython 以支持并行
    VEC_SIZE = 20
    CONTEXT_WINDOW = 5  # 提取目标词上下文距离最长5个词


# 训练

def build_model(self):
    '''
    建立模型
    :return:
    '''
    model = Sequential()
    # model.add(LSTM(1, input_shape=(timesteps, data_dim), return_sequences=True))

w2vTrain(Config)
preprocess_file(Config)
preprocess_data()


### 2.3. 查看结果

# 加载模型
w2v_model = word2vec.Word2Vec.load(Config.ModelDir + Config.model_output)

# print(w2v_model.syn1neg)

# print(w2v_model.wv.most_similar('body'))  # 结果一般

# print(w2v_model.wv.most_similar('heart'))  # 结果太差

# 数据集不够大时，停止词太多，解决方法：去除停止词

# 停止词
from nltk.corpus import stopwords

nltk.download('stopwords')
StopWords = stopwords.words('english')

StopWords = StopWords[:20]


# print(StopWords)

# 重新训练
# 模型训练函数
def w2vTrain_removeStopWords(Config):
    sentences = list(MySentences(Config.poetry_file))
    for idx, sentence in enumerate(sentences):
        sentence = [w for w in sentence if w not in StopWords]
        sentences[idx] = sentence
    w2v_model = word2vec.Word2Vec(sentences, min_count=Config.MIN_COUNT,
                                  workers=Config.CPU_NUM, vector_size=Config.VEC_SIZE)
    w2v_model.save(Config.ModelDir + Config.model_output)


w2vTrain_removeStopWords(Config)
w2v_model = word2vec.Word2Vec.load(Config.ModelDir + Config.model_output)

# print(w2v_model.wv.most_similar('heart'))  # 结果一般
