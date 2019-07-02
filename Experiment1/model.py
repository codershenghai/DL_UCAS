from keras.models import Sequential
from keras.layers import Dense, Conv2D, Flatten, Dropout, MaxPool2D


def def_model(n_classes):
    model = Sequential()

    model.add(Conv2D(32, kernel_size=(5, 5), input_shape=(20, 20, 3), activation='relu', padding='same'))
    model.add(MaxPool2D())
    model.add(Dropout(0.3))

    model.add(Conv2D(64, kernel_size=(5, 5), activation='relu', padding='same'))
    model.add(MaxPool2D())
    model.add(Dropout(0.3))

    model.add(Conv2D(128, kernel_size=(5, 5), activation='relu', padding='same'))
    model.add(MaxPool2D())
    model.add(Dropout(0.5))

    model.add(Conv2D(256, kernel_size=(5, 5), activation='relu', padding='same'))
    model.add(MaxPool2D())
    model.add(Dropout(0.5))

    model.add(Flatten())
    model.add(Dense(512, activation='relu'))
    model.add(Dropout(0.5))

    model.add(Dense(n_classes, activation='softmax'))

    model.summary()

    return model