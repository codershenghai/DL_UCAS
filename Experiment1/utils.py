import os
import numpy as np
from keras import callbacks
from keras.models import Sequential, model_from_yaml, load_model
from keras.layers import Dense, Conv2D, Flatten, Dropout, MaxPool2D
from keras.optimizers import Adam, SGD
from keras.preprocessing import image
from keras.utils import np_utils


def load_data(path, n_classes):
    images = []
    labels = []

    num_list = os.listdir(path)
    for i in num_list:
        num_path = os.path.join(path, i)
        img_list = os.listdir(num_path)
        for j in img_list:
            img_path = os.path.join(num_path, j)
            img = image.load_img(img_path, target_size=(20, 20))
            img_array = image.img_to_array(img)
            images.append(img_array)
            labels.append(int(i))

    data = np.array(images)
    labels = np.array(labels)
    labels = np_utils.to_categorical(labels, n_classes)

    return data/255, labels