from __future__ import division, print_function, absolute_import
import tensorflow as tf
import os
import numpy as np
import librosa
from random import shuffle


# 对音频文件特征MFCC进行提取
def read_files(files):
    labels = []
    features = []
    for ans, files in files.items():
        for file in files:
            wave, sr = librosa.load(file, mono=True)
            label = dense_to_one_hot(ans, 10)
            labels.append(label)
            mfcc = librosa.feature.mfcc(wave, sr)
            mfcc = np.pad(mfcc, ((0, 0), (0, 100 - len(mfcc[0]))), mode='constant', constant_values=0)
            features.append(np.array(mfcc))
    return np.array(features), np.array(labels)


def load_files(path="./numbersRec/recordings/"):
    files = os.listdir(path)
    cls_files = {}
    for wav in files:
        if not wav.endswith(".wav"): continue
        ans = int(wav[0])
        cls_files.setdefault(ans, [])
        cls_files[ans].append(path + wav)
    train_files = {}
    valid_files = {}
    test_files = {}
    for ans, file_list in cls_files.items():
        shuffle(file_list)
        all_len = len(file_list)
        train_len = int(all_len * 0.7)
        valid_len = int(all_len * 0.2)
        test_len = all_len - train_len - valid_len
        train_files[ans] = file_list[0:train_len]
        valid_files[ans] = file_list[train_len:train_len + valid_len]
        test_files[ans] = file_list[all_len - test_len:all_len]
    return train_files, valid_files, test_files


def batch_iter(X, Y, batch_size=128):
    data_len = len(X)
    num_batch = int((data_len - 1) / batch_size) + 1

    indices = np.random.permutation(np.arange(data_len))
    x_shuffle = X[indices]
    y_shuffle = Y[indices]

    for i in range(num_batch):
        start_id = i * batch_size
        end_id = min((i + 1) * batch_size, data_len)
        yield x_shuffle[start_id:end_id], y_shuffle[start_id:end_id]


def feed_data(cnn, x_batch, y_batch, keep_prob):
    feed_dict = {
        cnn.input_x: x_batch,
        cnn.input_y: y_batch,
        cnn.keep_prob: keep_prob
    }
    return feed_dict


# 对特征向量进行归一化处理
def dense_to_one_hot(labels_dense, num_classes=10):
    """Convert class labels from scalars to one-hot vectors."""
    return np.eye(num_classes)[labels_dense]


def mean_normalize(features):
    std_value = features.std()
    mean_value = features.mean()
    return (features - mean_value) / std_value


# 测试
def read_test_wave(path):
    files = os.listdir(path)
    feature = []
    label = []
    for wav in files:
        # print(wav)
        if not wav.endswith(".wav"): continue
        ans = int(wav[0])
        wave, sr = librosa.load(path + wav, mono=True)
        label.append(ans)
        # print("真实lable: %d" % ans)
        mfcc = librosa.feature.mfcc(wave, sr)
        mfcc = np.pad(mfcc, ((0, 0), (0, 100 - len(mfcc[0]))), mode='constant', constant_values=0)
        feature.append(np.array(mfcc))
    features = mean_normalize(np.array(feature))
    return features, label