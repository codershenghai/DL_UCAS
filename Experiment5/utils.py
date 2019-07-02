from keras.preprocessing import text
from keras.preprocessing import sequence
from keras.callbacks import EarlyStopping
from keras import Input, Model
from keras.layers import Embedding, Dense, Conv1D, GlobalMaxPooling1D, Concatenate
import numpy as np


max_features = 5000
max_len = 400
train_path = "./dataset/train.txt"
val_path = "./dataset/validation.txt"
test_path = "./dataset/test.txt"


def data_gen():
    x_train, x_val, x_test = [], [], []
    y_train, y_val, y_test = [], [], []

    with open(train_path, encoding='utf-8') as f1:
        train_corpus = f1.readlines()
    with open(val_path, encoding='utf-8') as f2:
        val_corpus = f2.readlines()
    with open(test_path, encoding='utf-8') as f3:
        test_corpus = f3.readlines()

    for _corpus in train_corpus:
        x_train.append(_corpus[2:])
        y_train.append(_corpus[0])

    for _corpus in val_corpus:
        x_val.append(_corpus[2:])
        y_val.append(_corpus[0])

    for _corpus in test_corpus:
        x_test.append(_corpus[2:])
        y_test.append(_corpus[0])

    # 向量化文本，num_words为处理的最大单词数量
    token = text.Tokenizer(num_words=max_features)
    token.fit_on_texts(x_train)

    # 将影评映射到数字
    x_train = token.texts_to_sequences(x_train)
    x_val = token.texts_to_sequences(x_val)
    x_test = token.texts_to_sequences(x_test)
    # 让所有影评保持固定长度
    x_train = sequence.pad_sequences(x_train, maxlen=max_len)
    x_val = sequence.pad_sequences(x_val, maxlen=max_len)
    x_test = sequence.pad_sequences(x_test, maxlen=max_len)
    # 转为numpy数组
    y_train = np.array(y_train).astype(np.int64)
    y_val = np.array(y_val).astype(np.int64)
    y_test = np.array(y_test).astype(np.int64)

    return (x_train, y_train), (x_val, y_val), (x_test, y_test)


if __name__ == '__main__':
    data_gen()